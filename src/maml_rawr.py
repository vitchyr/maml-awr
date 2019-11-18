import argparse
from copy import deepcopy
from typing import List, Optional
import os
import itertools

import higher
import numpy as np
import torch
import torch.autograd as A
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from src.envs import Env
from src.nn import MLP
from src.utils import ReplayBuffer, Experience


def copy_model_with_grads(from_model: nn.Module, to_model: nn.Module = None) -> nn.Module:
    with torch.no_grad():
        if to_model is None:
            to_model = deepcopy(from_model)

        for from_param, to_param in zip(from_model.parameters(), to_model.parameters()):
            to_param[:] = from_param[:]
            to_param.grad = from_param.grad

    return to_model


def kld_loss(mu, log_sigma):
    return -0.5 * (1 + 2 * log_sigma - mu.pow(2) - log_sigma.exp().pow(2)).sum()


def env_action_dim(env):
    action_space = env.action_space.shape
    return action_space[0] if len(action_space) > 0 else 1


class MAMLRAWR(object):
    def __init__(self, args: argparse.Namespace, envs: List[Env], log_dir: str, name: str = None,
                 policy_hidden_layers: List[int] = [32, 32], value_function_hidden_layers: List[int] = [32, 32],
                 training_iterations: int = 20000, batch_size: int = 64,
                 alpha1: float = 0.1, alpha2: float = 2.5e-4, eta1: float = 0.1, eta2: float = 1e-4, mu: float = 1e-4,
                 adaptation_temperature: float = 0.05, exploration_temperature: float = 0.05,
                 initial_trajectories: int = 40, device: str = 'cuda:0', maml_steps: int = 1,
                 test_samples: int = 10, weight_clamp: float = 20.0, action_sigma: float = 0.2,
                 visualization_interval: int = 100, silent: bool = False, replay_buffer_length: int = 1000,
                 inline_render: bool = False, gradient_steps_per_iteration: int = 1, discount_factor: float = 0.99,
                 offline_inner_loop: bool = False, grad_clip: float = 100., inner_batch_size: int = 256, bias_linear: bool = False):
        self._envs = envs
        self._log_dir = log_dir
        self._name = name if name is not None else 'throwaway_test_run'
        self._args = args
        
        if len(envs) == 1:
            alpha1 = 0
            eta1 = 0
            
        example_env = self._envs[0]

        self._observation_dim = example_env.observation_space.shape[0]
        self._action_dim = env_action_dim(example_env)

        policy_head = [32, 1] if args.advantage_head_coef is not None else None
        
        self._adaptation_policy = MLP([self._observation_dim] +
                                      policy_hidden_layers +
                                      [self._action_dim],
                                      final_activation=torch.tanh,
                                      bias_linear=bias_linear,
                                      extra_head_layers=policy_head).to(device)
        self._exploration_policy = MLP([self._observation_dim] +
                                       policy_hidden_layers +
                                       [self._action_dim],
                                       final_activation=torch.tanh).to(device)
        self._value_function = MLP([self._observation_dim] + value_function_hidden_layers + [1],
                                   bias_linear=bias_linear).to(device)

        self._advantage_coef = nn.Parameter(torch.ones(1).to(device))
        
        self._adaptation_policy_optimizer = O.Adam(self._adaptation_policy.parameters(), lr=alpha2)
        self._value_function_optimizer = O.Adam(self._value_function.parameters(), lr=eta2)
        self._exploration_policy_optimizer = O.Adam(self._adaptation_policy.parameters(), lr=mu)
        self._advantage_coef_opt = O.SGD([self._advantage_coef], lr=1e-4, momentum=0.9)
        
        if args.vf_archive is not None:
            print(f'Loading value function archive from: {args.vf_archive}')
            self._value_function.load_state_dict(torch.load(args.vf_archive, map_location=device))
        if args.ap_archive is not None:
            print(f'Loading policy archive from: {args.ap_archive}')
            self._adaptation_policy.load_state_dict(torch.load(args.ap_archive, map_location=device))
        
        self._adapted_buffers = [ReplayBuffer(env._max_episode_steps, env.observation_space.shape[0], env_action_dim(env),
                                            max_trajectories=replay_buffer_length, discount_factor=discount_factor, immutable=offline_inner_loop or args.eval)
                               for env in self._envs]

        self._buffers = [ReplayBuffer(env._max_episode_steps, env.observation_space.shape[0], env_action_dim(env),
                                      max_trajectories=replay_buffer_length, discount_factor=discount_factor)
                         for env in self._envs]
        self._pre_adapted_buffers = [ReplayBuffer(env._max_episode_steps, env.observation_space.shape[0], env_action_dim(env),
                                                  max_trajectories=replay_buffer_length, discount_factor=discount_factor)
                                     for env in self._envs]
        
        self._training_iterations = training_iterations
        self._batch_size = batch_size
        self._alpha1, self._alpha2, self._eta1, self._eta2, self._mu = alpha1, alpha2, eta1, eta2, mu
        self._adaptation_temperature, self._exploration_temperature = adaptation_temperature, exploration_temperature
        self._device = torch.device(device)
        self._initial_trajectories = initial_trajectories
        self._maml_steps = maml_steps
        self._cpu = torch.device('cpu')
        self._test_samples = test_samples
        self._advantage_clamp = np.log(weight_clamp)
        self._action_sigma = action_sigma
        self._visualization_interval = visualization_interval
        self._silent = silent
        self._inline_render = inline_render
        self._gradient_steps_per_iteration = gradient_steps_per_iteration
        self._offline_inner_loop = offline_inner_loop
        self._grad_clip = grad_clip
        self._inner_batch_size = inner_batch_size
        
    #################################################################
    ################# SUBROUTINES FOR TRAINING ######################
    #################################################################
    def _rollout_policy(self, policy: MLP, env: Env, test: bool = False, random: bool = False, render: bool = False) -> List[Experience]:
        trajectory = []
        cpu_policy = deepcopy(policy).to(self._cpu)
        state = env.reset()
        if render:
            env.render()
        done = False
        total_reward = 0
        episode_t = 0
        while not done:
            if not random:
                with torch.no_grad():
                    mu = cpu_policy(torch.tensor(state).unsqueeze(0).float())#.squeeze().numpy()
                    if test:
                        action = mu
                    else:
                        action = mu + torch.empty_like(mu).normal_() * sigma

                    log_prob = torch.distributions.MultivariateNormal(mu, torch.empty_like(mu).fill_(self._action_sigma)).log_prob(action).numpy()
                    action = action.squeeze().numpy().clip(min=env.action_space.low, max=env.action_space.high)
            else:
                action = env.action_space.sample()
                log_prob = 1 / 2 ** env.action_space.shape[0]

            next_state, reward, done, info_dict = env.step(action)
            if render:
                env.render()
            trajectory.append(Experience(state, action, next_state, reward, done, log_prob))
            state = next_state
            total_reward += reward
            episode_t += 1
            if episode_t >= env._max_episode_steps:
                break
            
        return trajectory, total_reward

    def mc_value_estimates_on_batch(self, value_function, batch):
        with torch.no_grad():
            terminal_state_value_estimates = value_function(batch[:,self._observation_dim * 2 + self._action_dim:self._observation_dim * 3 + self._action_dim])
            terminal_factors = batch[:, -4:-3] # I know this magic number indexing is heinous... I'm sorry
            mc_value_estimates = batch[:,-1:] + terminal_factors * terminal_state_value_estimates

        return mc_value_estimates
    
    def value_function_loss_on_batch(self, value_function, batch, inner: bool = False):
        value_estimates = value_function(batch[:,:self._observation_dim])
        with torch.no_grad():
            mc_value_estimates = self.mc_value_estimates_on_batch(value_function, batch)

        losses = (value_estimates - mc_value_estimates).pow(2)
        if inner:
            advantages = (mc_value_estimates - value_estimates)
            original_action = batch[:,self._observation_dim:self._observation_dim + self._action_dim]

        # return (value_estimates - mc_value_estimates).pow(2).mean() / (mc_value_estimates.var() + 1e-3), mc_value_estimates.mean(), mc_value_estimates.std()
        return losses.mean(), mc_value_estimates.mean(), mc_value_estimates.std()

    def adaptation_policy_loss_on_batch(self, policy, value_function, batch, inner: bool = False):
        with torch.no_grad():
            value_estimates = value_function(batch[:,:self._observation_dim])
            mc_value_estimates = self.mc_value_estimates_on_batch(value_function, batch)
            
            advantages = (mc_value_estimates - value_estimates)
            normalized_advantages = (1 / self._adaptation_temperature) * (advantages - advantages.mean()) / advantages.std()
            weights = advantages.clamp(max=self._advantage_clamp).exp()

        original_action = batch[:,self._observation_dim:self._observation_dim + self._action_dim]
        if self._args.advantage_head_coef is not None:
            action_mu, advantage_prediction = policy(batch[:,:self._observation_dim], batch[:,self._observation_dim:self._observation_dim+self._action_dim])
        else:
            action_mu = policy(batch[:,:self._observation_dim])
        action_sigma = torch.empty_like(action_mu).fill_(self._action_sigma)
        action_distribution = Normal(action_mu, action_sigma)
        action_log_probs = action_distribution.log_prob(batch[:,self._observation_dim:self._observation_dim + self._action_dim])

        losses = -(action_log_probs * weights)

        if inner:
            # Use learnable learning rate
            if self._args.advantage_head_coef is not None:
                losses = losses + F.softplus(self._advantage_coef) * self._args.advantage_head_coef * (advantage_prediction - advantages) ** 2

        return losses.mean()

    #################################################################
    #################################################################

    # This function is the body of the main training loop [L4]
    # At every iteration, it adds rollouts from the exploration policy and one of the adapted policies
    #  to the replay buffer. It also updates the adaptation value function, adaptation policy, and
    #  exploration policy
    def train_step(self, train_step_idx: int, writer: Optional[SummaryWriter] = None):
        batches = []
        value_functions = []
        adaptation_policies = []
        meta_value_grads = []
        meta_policy_grads = []
        test_rewards = []
        train_rewards = []
        rollouts = []
        for i, (env, adapted_buffer, buffer, pre_adapted_buffer) in enumerate(zip(self._envs, self._adapted_buffers, self._buffers, self._pre_adapted_buffers)):
            # Sample an exploration trajectory and add to buffer i [L6]
            # buffer.add_trajectory(self._rollout_policy(self._exploration_policy, env)[0])

            # Sample J training batches for independent adaptations [L7]
            inner_buffer = pre_adapted_buffer if self._args.pre_adapted else adapted_buffer
            np_batch = inner_buffer.sample(self._inner_batch_size * self._maml_steps).reshape(
                (self._args.n_adaptations, self._maml_steps, self._inner_batch_size // self._args.n_adaptations, -1))
            pyt_batch = torch.tensor(np_batch, requires_grad=False).to(self._device)
            batches.append(pyt_batch)
            meta_batch = torch.tensor(buffer.sample(self._batch_size), requires_grad=False).to(self._device)

            value_functions_i = []
            adaptation_policies_i = []
            meta_value_grads_i = []
            adv_coef_grads = []
            meta_policy_grads_i = []
            inner_value_losses = []
            meta_value_losses = []
            inner_policy_losses = []
            meta_policy_losses = []
            inner_mc_means, inner_mc_stds = [], []
            outer_mc_means, outer_mc_stds = [], []
            for j, batch in enumerate(pyt_batch):
                ##################################################################################################
                # Adapt value function and collect meta-gradients
                ##################################################################################################
                value_function_j = deepcopy(self._value_function)
                opt = O.SGD(value_function_j.parameters(), lr=self._eta1)
                with higher.innerloop_ctx(value_function_j, opt) as (f_value_function_j, diff_value_opt):
                    if self._eta1 > 0:
                        for inner_batch in batch:
                            # Compute loss and adapt value function [L9]
                            loss, mc_inner, mc_std_inner = self.value_function_loss_on_batch(f_value_function_j, inner_batch, inner=True)
                            inner_mc_means.append(mc_inner.item())
                            inner_mc_stds.append(mc_std_inner.item())
                            diff_value_opt.step(loss)
                            inner_value_losses.append(loss.item())

                    # Collect grads for the value function update in the outer loop [L14],
                    #  which is not actually performed here
                    meta_value_function_loss, mc, mc_std = self.value_function_loss_on_batch(f_value_function_j, meta_batch)
                    meta_value_grad_j = A.grad(meta_value_function_loss, f_value_function_j.parameters(time=0))

                    outer_mc_means.append(mc.item())
                    outer_mc_stds.append(mc_std.item())
                    meta_value_losses.append(meta_value_function_loss.item())
                    meta_value_grads_i.append(meta_value_grad_j)
                    copy_model_with_grads(f_value_function_j, value_function_j)
                    value_functions_i.append(value_function_j)

                ##################################################################################################

                ##################################################################################################
                # Adapt policy and collect meta-gradients [
                ##################################################################################################
                adapted_value_function = value_functions_i[-1]
                adaptation_policy_j = deepcopy(self._adaptation_policy)
                opt = O.SGD(adaptation_policy_j.parameters(), lr=self._alpha1)
                with higher.innerloop_ctx(adaptation_policy_j, opt) as (f_adaptation_policy_j, diff_policy_opt):
                    if self._alpha1 > 0:
                        for inner_batch in batch:
                            # Compute loss and adapt policy [L10]
                            loss = self.adaptation_policy_loss_on_batch(f_adaptation_policy_j, adapted_value_function, inner_batch, inner=True)
                            diff_policy_opt.step(loss)
                            inner_policy_losses.append(loss.item())
                            
                    meta_policy_loss = self.adaptation_policy_loss_on_batch(f_adaptation_policy_j, adapted_value_function, meta_batch)
                    meta_policy_grad_j = A.grad(meta_policy_loss, f_adaptation_policy_j.parameters(time=0), retain_graph=True)
                    if self._args.advantage_head_coef is not None:
                        adv_coef_grads.append(A.grad(meta_policy_loss, self._advantage_coef)[0])
                    # Collect grads for the adaptation policy update in the outer loop [L15],
                    #  which is not actually performed here
                    meta_policy_losses.append(meta_policy_loss.item())
                    meta_policy_grads_i.append(meta_policy_grad_j)
                    copy_model_with_grads(f_adaptation_policy_j, adaptation_policy_j)
                    adaptation_policies_i.append(adaptation_policy_j)
                ##################################################################################################
            
            value_functions.append(value_functions_i)
            meta_value_grads.append(meta_value_grads_i)
            adaptation_policies.append(adaptation_policies_i)
            meta_policy_grads.append(meta_policy_grads_i)

            # Sample adapted policy trajectory, add to replay buffer i [L12]
            if train_step_idx % self._gradient_steps_per_iteration == 0:
                train_trajectory, train_reward = self._rollout_policy(adaptation_policies_i[0], env, test=False)
                if not (self._offline_inner_loop or self._args.eval):
                    adapted_buffer.add_trajectory(train_trajectory)
                buffer.add_trajectory(train_trajectory)

                if self._args.pre_adapted:
                    train_trajectory, _ = self._rollout_policy(self._adaptation_policy, env, test=False)
                    if not (self._offline_inner_loop or self._args.eval):
                        pre_adapted_buffer.add_trajectory(train_trajectory)

                train_rewards.append(train_reward)
            
            if train_step_idx % self._visualization_interval == 0:
                if self._inline_render:
                    print(f'Visualizing task {i}, test rollout')
                test_trajectory, test_reward = self._rollout_policy(adaptation_policies_i[0], env, test=True,
                                                                    render=self._inline_render)
                rollouts.append(test_trajectory)
                test_rewards.append(test_reward)

            if writer is not None:
                if len(inner_value_losses):
                    writer.add_scalar(f'Loss_Value_Inner/Task_{i}', np.mean(inner_value_losses), train_step_idx)
                    writer.add_scalar(f'Loss_Policy_Inner/Task_{i}', np.mean(inner_policy_losses), train_step_idx)
                    writer.add_scalar(f'MC_Mean_Inner/Task_{i}', np.mean(inner_mc_means), train_step_idx)
                    writer.add_scalar(f'MC_std_Inner/Task_{i}', np.mean(inner_mc_stds), train_step_idx)
                writer.add_scalar(f'MC_Mean_Outer/Task_{i}', np.mean(outer_mc_means), train_step_idx)
                writer.add_scalar(f'MC_std_Outer/Task_{i}', np.mean(outer_mc_stds), train_step_idx)
                writer.add_scalar(f'Loss_Value_Outer/Task_{i}', np.mean(meta_value_losses), train_step_idx)
                writer.add_scalar(f'Loss_Policy_Outer/Task_{i}', np.mean(meta_policy_losses), train_step_idx)
                if train_step_idx % self._visualization_interval == 0:
                    writer.add_scalar(f'Reward_Test/Task_{i}', test_reward, train_step_idx)
                if train_step_idx % self._gradient_steps_per_iteration == 0:
                    writer.add_scalar(f'Reward_Train/Task_{i}', train_reward, train_step_idx)

                if self._args.advantage_head_coef is not None:
                    writer.add_scalar(f'Advantage_Coef', F.softplus(self._advantage_coef), train_step_idx)

        if self._args.eval:
            return rollouts, test_rewards, train_rewards, meta_value_losses, meta_policy_losses

        ############################################################################3
        # Meta-update value function [L14]
        ############################################################################3
        for idx, parameter in enumerate(self._value_function.parameters()):
            grads = []
            for i in range(len(meta_value_grads)):
                for j in range(len(meta_value_grads[i])):
                    grads.append(meta_value_grads[i][j][idx])
            parameter.grad = sum(grads) / len(grads)
        if self._grad_clip is not None:
            grad = torch.nn.utils.clip_grad_norm_(self._value_function.parameters(), self._grad_clip)
            writer.add_scalar(f'Value_Outer_Grad', grad, train_step_idx)
        self._value_function_optimizer.step()
        self._value_function_optimizer.zero_grad()
        ############################################################################3

        ############################################################################3
        # Meta-update adaptation policy [L15]
        ############################################################################3
        for idx, parameter in enumerate(self._adaptation_policy.parameters()):
            grads = []
            for i in range(len(meta_policy_grads)):
                for j in range(len(meta_policy_grads[i])):
                    grads.append(meta_policy_grads[i][j][idx])
            parameter.grad = sum(grads) / len(grads)
        if self._grad_clip is not None:
            grad = torch.nn.utils.clip_grad_norm_(self._adaptation_policy.parameters(), self._grad_clip)
            writer.add_scalar(f'Policy_Outer_Grad', grad, train_step_idx)
        self._adaptation_policy_optimizer.step()
        self._adaptation_policy_optimizer.zero_grad()
        ############################################################################3

        ###################
        if self._args.advantage_head_coef is not None:
            self._advantage_coef.grad = sum(adv_coef_grads) / len(adv_coef_grads)
            self._advantage_coef_opt.step()
            self._advantage_coef_opt.zero_grad()
        ###################
        
        ############################################################################3
        # Update exploration policy [WIP] [L16]
        ############################################################################3
        for i, batches_i in enumerate(batches):
            for j, batch_ij in enumerate(batches_i):
                batch = batch_ij.view(self._maml_steps * self._batch_size // self._args.n_adaptations, -1)

                with torch.no_grad():
                    value_estimates = value_functions[i][j](batch[:,:self._observation_dim])
                    mc_value_estimates = self.mc_value_estimates_on_batch(value_functions[i][j], batch)
                    
                    advantages = ((1 / self._exploration_temperature) * (mc_value_estimates - value_estimates))
                    weight = advantages.mean().clamp(max=self._advantage_clamp).exp()

                action_mu = self._exploration_policy(batch[:,:self._observation_dim])
                action_sigma = torch.empty_like(action_mu).fill_(self._action_sigma)
                action_distribution = Normal(action_mu, action_sigma)
                action_log_probs = action_distribution.log_prob(batch[:,self._observation_dim:self._observation_dim + self._action_dim])

                exploration_policy_loss = -(action_log_probs.sum() * weight)
                exploration_policy_loss.backward()
        self._exploration_policy_optimizer.step()
        self._exploration_policy_optimizer.zero_grad()
        ############################################################################3

        return rollouts, test_rewards, train_rewards, meta_value_losses, meta_policy_losses

    def train(self):
        log_path = f'{self._log_dir}/{self._name}'
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        tensorboard_log_path = f'{log_path}/tb'
        if not os.path.exists(tensorboard_log_path):
            os.makedirs(tensorboard_log_path)
        summary_writer = SummaryWriter(tensorboard_log_path)
            
        # Gather initial trajectory rollouts
        for i, (env, adapted_buffer, buffer, pre_adapted_buffer) in enumerate(zip(self._envs, self._adapted_buffers, self._buffers, self._pre_adapted_buffers)):
            trajectories = [self._rollout_policy(self._adaptation_policy, env, random=True)[0] for _ in range(self._initial_trajectories)]
            adapted_buffer.add_trajectories(trajectories, force=True)
            pre_adapted_buffer.add_trajectories(trajectories, force=True)
            buffer.add_trajectories(trajectories)

        for t in range(self._training_iterations):
            rollouts, test_rewards, train_rewards, value, policy = self.train_step(t, summary_writer)

            if not self._silent:
                if len(test_rewards):
                    print(f'{t}: {test_rewards}, {np.mean(value)}, {np.mean(policy)}')

            if len(test_rewards):
                summary_writer.add_scalar(f'Reward_Test/Mean', np.mean(test_rewards), t)
            if len(train_rewards):
                summary_writer.add_scalar(f'Reward_Train/Mean', np.mean(train_rewards), t)
            if t % self._visualization_interval == 0:
                try:
                    for idx, (env, rollout) in enumerate(zip(self._envs, rollouts)):
                        image = env.render_rollout(rollout, f'{log_path}/{t}_{idx}.png')
                except Exception as e:
                    pass

                torch.save(self._value_function.state_dict(), f'{log_path}/vf_LATEST.pt')
                torch.save(self._value_function.state_dict(), f'{log_path}/vf_LATEST_.pt')
                torch.save(self._adaptation_policy.state_dict(), f'{log_path}/ap_LATEST.pt')
                torch.save(self._adaptation_policy.state_dict(), f'{log_path}/ap_LATEST_.pt')
                #torch.save(self._exploration_policy.state_dict(), f'{log_path}/ep_LATEST.pt')
                #torch.save(self._exploration_policy.state_dict(), f'{log_path}/ep_LATEST_.pt')
