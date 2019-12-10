import argparse
from copy import deepcopy
from typing import List, Optional
import os
import itertools
import math
import random

import higher
import numpy as np
import torch
import torch.autograd as A
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as O
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter

from src.envs import Env
from src.nn import MLP, CVAE
from src.utils import ReplayBuffer, Experience, argmax, kld


def copy_model_with_grads(from_model: nn.Module, to_model: nn.Module = None) -> nn.Module:
    with torch.no_grad():
        if to_model is None:
            to_model = deepcopy(from_model)

        for from_param, to_param in zip(from_model.parameters(), to_model.parameters()):
            to_param[:] = from_param[:]
            to_param.grad = from_param.grad

    return to_model


def env_action_dim(env):
    action_space = env.action_space.shape
    return action_space[0] if len(action_space) > 0 else 1


class MAMLRAWR(object):
    def __init__(self, args: argparse.Namespace, env: Env, log_dir: str, name: str = None,
                 policy_hidden_layers: List[int] = [32, 32], value_function_hidden_layers: List[int] = [32, 32],
                 training_iterations: int = 20000, batch_size: int = 64,
                 initial_trajectories: int = 40, device: str = 'cuda:0', maml_steps: int = 1,
                 test_samples: int = 10, action_sigma: float = 0.2,
                 visualization_interval: int = 100, silent: bool = False, replay_buffer_length: int = 1000,
                 gradient_steps_per_iteration: int = 1, discount_factor: float = 0.99, grad_clip: float = 100.,
                 inner_batch_size: int = 256, bias_linear: bool = False):
        self._env = env
        self._log_dir = log_dir
        self._name = name if name is not None else 'throwaway_test_run'
        self._args = args
        
        self._observation_dim = env.observation_space.shape[0]
        self._action_dim = env_action_dim(env)

        policy_head = [32, 1] if args.advantage_head_coef is not None else None
        
        self._adaptation_policy = MLP([self._observation_dim] +
                                      policy_hidden_layers +
                                      [self._action_dim],
                                      final_activation=torch.tanh,
                                      bias_linear=bias_linear,
                                      extra_head_layers=policy_head).to(device)
        if args.cvae:
            self._exploration_policy = CVAE(self._observation_dim, self._action_dim,
                                            self._env.task_description_dim() if not args.unconditional else 0,
                                            args.latent_dim).to(device)
        else:
            self._exploration_policy = MLP([self._observation_dim] +
                                           policy_hidden_layers +
                                           [self._action_dim],
                                           final_activation=torch.tanh).to(device)

        self._value_function = MLP([self._observation_dim] + value_function_hidden_layers + [1],
                                   bias_linear=bias_linear).to(device)
        print(self._adaptation_policy.seq[0]._linear.weight.mean())
        
        self._adaptation_policy_optimizer = O.Adam(self._adaptation_policy.parameters(), lr=args.outer_policy_lr)
        self._value_function_optimizer = O.Adam(self._value_function.parameters(), lr=args.outer_value_lr)
        self._exploration_policy_optimizer = O.Adam(self._exploration_policy.parameters(), lr=args.exploration_lr)
        
        if args.vf_archive is not None:
            print(f'Loading value function archive from: {args.vf_archive}')
            self._value_function.load_state_dict(torch.load(args.vf_archive, map_location=device))
        if args.ap_archive is not None:
            print(f'Loading policy archive from: {args.ap_archive}')
            self._adaptation_policy.load_state_dict(torch.load(args.ap_archive, map_location=device))
        if args.ep_archive is not None:
            print(f'Loading exploration policy archive from: {args.ep_archive}')
            self._exploration_policy.load_state_dict(torch.load(args.ep_archive, map_location=device))
            
        inner_buffer = args.buffer_paths if args.load_inner_buffer else [None for _ in self._env.tasks]
        outer_buffer = args.buffer_paths if args.load_outer_buffer else [None for _ in self._env.tasks]
        self._inner_buffers = [ReplayBuffer(self._env._max_episode_steps, self._env.observation_space.shape[0], env_action_dim(self._env),
                                            max_trajectories=replay_buffer_length, discount_factor=discount_factor,
                                            immutable=args.offline or args.offline_inner, load_from=inner_buffer[i])
                               for i, task in enumerate(self._env.tasks)]
        self._outer_buffers = [ReplayBuffer(self._env._max_episode_steps, self._env.observation_space.shape[0], env_action_dim(self._env),
                                            max_trajectories=replay_buffer_length, discount_factor=discount_factor,
                                            immutable=args.offline or args.offline_outer, load_from=outer_buffer[i])
                               for i, task in enumerate(self._env.tasks)]

        self._training_iterations = training_iterations
        self._batch_size = batch_size
        self._inner_policy_lr, self._inner_value_lr = args.inner_policy_lr, args.inner_value_lr
        self._adaptation_temperature, self._exploration_temperature = args.adaptation_temp, args.exploration_temp
        self._device = torch.device(device)
        self._initial_trajectories = initial_trajectories
        self._maml_steps = maml_steps
        self._cpu = torch.device('cpu')
        self._test_samples = test_samples
        self._advantage_clamp = np.log(args.exp_advantage_clip)
        self._action_sigma = action_sigma
        self._visualization_interval = visualization_interval
        self._silent = silent
        self._gradient_steps_per_iteration = gradient_steps_per_iteration
        self._grad_clip = grad_clip
        self._inner_batch_size = inner_batch_size
        self._env_seeds = np.random.randint(1e10, size=(int(1e7),))
        self._rollout_counter = 0
        
    #################################################################
    ################# SUBROUTINES FOR TRAINING ######################
    #################################################################
    def _rollout_policy(self, policy: MLP, env: Env, test: bool = False, random: bool = False, render: bool = False) -> List[Experience]:
        env.seed(self._env_seeds[self._rollout_counter].item())
        self._rollout_counter += 1

        trajectory = []
        cpu_policy = deepcopy(policy).to(self._cpu)
        state = env.reset()
        if render:
            env.render()
        done = False
        total_reward = 0
        episode_t = 0
        if isinstance(cpu_policy, CVAE) and not self._args.unconditional:
            desc = torch.tensor(self.task_description(env, batch=1), device=self._cpu)
        while not done:
            if not random:
                with torch.no_grad():
                    action_sigma = self._action_sigma
                    if isinstance(cpu_policy, CVAE):
                        if self._args.unconditional:
                            mu, action_sigma = cpu_policy(torch.tensor(state).unsqueeze(0).float())
                        else:
                            mu, action_sigma = cpu_policy(torch.tensor(state).unsqueeze(0).float(), desc)
                        mu = mu.squeeze()
                        action_sigma = action_sigma.squeeze()
                    else:
                        mu = cpu_policy(torch.tensor(state).unsqueeze(0).float()).squeeze()

                    if test:
                        action = mu
                    else:
                        action = mu + torch.empty_like(mu).normal_() * action_sigma

                    log_prob = D.Normal(mu, torch.empty_like(mu).fill_(self._action_sigma)).log_prob(action).numpy().sum()
                    action = action.squeeze().numpy().clip(min=env.action_space.low, max=env.action_space.high)
            else:
                action = env.action_space.sample()
                log_prob = math.log(1 / 2 ** env.action_space.shape[0])

            next_state, reward, done, info_dict = env.step(action)
            reward += self._args.reward_offset
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

        return losses.mean(), value_estimates.mean(), mc_value_estimates.mean(), mc_value_estimates.std()

    def adaptation_policy_loss_on_batch(self, policy, value_function, batch, inner: bool = False):
        with torch.no_grad():
            value_estimates = value_function(batch[:,:self._observation_dim])
            mc_value_estimates = self.mc_value_estimates_on_batch(value_function, batch)

            advantages = (mc_value_estimates - value_estimates).squeeze(-1)
            normalized_advantages = (1 / self._adaptation_temperature) * (advantages - advantages.mean()) / advantages.std()
            weights = normalized_advantages.clamp(max=self._advantage_clamp).exp()

        original_action = batch[:,self._observation_dim:self._observation_dim + self._action_dim]
        if self._args.advantage_head_coef is not None:
            action_mu, advantage_prediction = policy(batch[:,:self._observation_dim], batch[:,self._observation_dim:self._observation_dim+self._action_dim])
        else:
            action_mu = policy(batch[:,:self._observation_dim])
        action_sigma = torch.empty_like(action_mu).fill_(self._action_sigma)
        action_distribution = D.Normal(action_mu, action_sigma)
        action_log_probs = action_distribution.log_prob(batch[:,self._observation_dim:self._observation_dim + self._action_dim]).sum(-1)

        losses = -(action_log_probs * weights)

        if inner:
            if self._args.advantage_head_coef is not None:
                losses = losses + self._args.advantage_head_coef * (advantage_prediction.squeeze() - advantages) ** 2

        return losses.mean()

    def update_model_with_grads(self, model: nn.Module, grads: list, optimizer: torch.optim.Optimizer, clip: float, step: bool = True):
        for idx, parameter in enumerate(model.parameters()):
            grads_ = []
            for i in range(len(grads)):
                for j in range(len(grads[i])):
                    grads_.append(grads[i][j][idx])
            parameter.grad = sum(grads_) / len(grads_)

        if self._grad_clip is not None:
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        else:
            grad = None

        if step:
            optimizer.step()
            optimizer.zero_grad()
        
        return grad

    def policy_advantage_on_batch(self, policy: nn.Module, value_function: nn.Module, batch: torch.tensor):
        with torch.no_grad():
            value_estimates = value_function(batch[:,:self._observation_dim])
            mc_value_estimates = self.mc_value_estimates_on_batch(value_function, batch)
            
            exp_clipped_advantages = (mc_value_estimates - value_estimates).squeeze(-1).clamp(max=self._advantage_clamp).exp()
            
            original_obs = batch[:,:self._observation_dim]
            original_action = batch[:,self._observation_dim:self._observation_dim + self._action_dim]
            original_log_probs = batch[:,-5]
            
            action = policy(original_obs)
            log_probs = D.Normal(action, torch.empty_like(action).fill_(self._action_sigma)).log_prob(original_action).sum(-1)
            ratio = (log_probs - original_log_probs).exp()
            clipped_ratio = ratio.clamp(min=1-self._args.ratio_clip, max=1+self._args.ratio_clip)

        if clipped_ratio.dim() > 1 or exp_clipped_advantages.dim() > 1:
            raise RuntimeError()

        return (clipped_ratio * exp_clipped_advantages).mean().item()

    #################################################################
    #################################################################

    def train_awr_exploration(self, train_step_idx: int, batches: torch.tensor, meta_batches: torch.tensor,
                              adaptation_policies: list, value_functions: list, writer: SummaryWriter):
        exploration_loss = 0
        for i, (meta_batch, pyt_batch) in enumerate(zip(meta_batches, batches)):
            weights = []
            for policy, vf in zip(adaptation_policies[i], value_functions[i]):
                weight = self.policy_advantage_on_batch(policy, vf, meta_batch)
                weights.append(weight)

            print(i, train_step_idx, weights)

            for j, batch in enumerate(pyt_batch):
                batch = batch.view(-1, batch.shape[-1])
                original_action = batch[:,self._observation_dim:self._observation_dim + self._action_dim]
                original_obs = batch[:,:self._observation_dim]
                action = self._exploration_policy(original_obs)
                log_probs = D.Normal(action, torch.empty_like(action).fill_(self._action_sigma)).log_prob(original_action).sum(-1)

                loss = -log_probs.mean() * weights[j]
                exploration_loss = exploration_loss + loss

        exploration_loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self._exploration_policy.parameters(), 1000)
        writer.add_scalar(f'Explore_Grad', grad, train_step_idx)
        writer.add_scalar(f'Explore_Loss', exploration_loss.item(), train_step_idx)

        self._exploration_policy_optimizer.step()
        self._exploration_policy_optimizer.zero_grad()

    def train_vae_exploration(self, train_step_idx: int, batches: torch.tensor, meta_batches: torch.tensor,
                              adaptation_policies: list, value_functions: list, writer: SummaryWriter):
        kld_loss = 0
        recon_loss = 0
        for i, (meta_batch, batch) in enumerate(zip(meta_batches, batches)):
            self._env.set_task_idx(i)
            if self._args.unconditional:
                task = None
            else:
                task = torch.tensor(self._env.task_description(batch=self._args.inner_batch_size), device=self._device)
            batch = batch.view(-1, batch.shape[-1])
            obs = batch[:,:self._observation_dim]
            action = batch[:,self._observation_dim:self._observation_dim + self._action_dim]

            pz_wxy, z = self._exploration_policy.encode(obs, action, task, sample=True)
            pz_wx = self._exploration_policy.prior(obs, task)

            py_wx = self._exploration_policy.decode(z, obs, task)
            mu_y = py_wx[:,:py_wx.shape[-1]//2]
            std_y = (py_wx[:,py_wx.shape[-1]//2:] / 2).exp()

            d_y = D.Normal(mu_y, std_y)
            
            kld_loss += kld(pz_wxy, pz_wx).mean()
            recon_loss += -d_y.log_prob(action).sum(-1).mean()

        exploration_loss = kld_loss + recon_loss
        exploration_loss.backward()
        
        grad = torch.nn.utils.clip_grad_norm_(self._exploration_policy.parameters(), 1000)

        writer.add_scalar(f'Explore_Grad', grad, train_step_idx)
        writer.add_scalar(f'Explore_Loss', exploration_loss.item(), train_step_idx)
        writer.add_scalar(f'Explore_Loss_Recon', recon_loss.item(), train_step_idx)
        writer.add_scalar(f'Explore_Loss_KLD', kld_loss.item(), train_step_idx)
        
        self._exploration_policy_optimizer.step()
        self._exploration_policy_optimizer.zero_grad()
            
    # This function is the body of the main training loop [L4]
    # At every iteration, it adds rollouts from the exploration policy and one of the adapted policies
    #  to the replay buffer. It also updates the adaptation value function, adaptation policy, and
    #  exploration policy
    def train_step(self, train_step_idx: int, writer: Optional[SummaryWriter] = None):
        batches = []
        meta_batches = []
        value_functions = []
        adaptation_policies = []
        meta_value_grads = []
        meta_policy_grads = []
        test_rewards = []
        train_rewards = []
        rollouts = []
        for i, (inner_buffer, outer_buffer) in enumerate(zip(self._inner_buffers, self._outer_buffers)):
            self._env.set_task_idx(i)
            
            # Sample J training batches for independent adaptations [L7]
            np_batch = inner_buffer.sample(self._inner_batch_size * self._maml_steps).reshape(
                (self._args.n_adaptations, self._maml_steps, self._inner_batch_size // self._args.n_adaptations, -1))
            pyt_batch = torch.tensor(np_batch, requires_grad=False).to(self._device)
            batches.append(pyt_batch)

            meta_batch = torch.tensor(outer_buffer.sample(self._batch_size), requires_grad=False).to(self._device)
            meta_batches.append(meta_batch)
            
            value_functions_i = []
            adaptation_policies_i = []
            meta_value_grads_i = []
            meta_policy_grads_i = []
            inner_value_losses = []
            meta_value_losses = []
            inner_policy_losses = []
            meta_policy_losses = []
            inner_mc_means, inner_mc_stds = [], []
            outer_mc_means, outer_mc_stds = [], []
            inner_values, outer_values = [], []
            for j, batch in enumerate(pyt_batch):
                ##################################################################################################
                # Adapt value function and collect meta-gradients
                ##################################################################################################
                value_function_j = deepcopy(self._value_function)
                opt = O.SGD(value_function_j.parameters(), lr=self._inner_value_lr)
                with higher.innerloop_ctx(value_function_j, opt) as (f_value_function_j, diff_value_opt):
                    if self._inner_value_lr > 0 and len(self._env.tasks) > 1:
                        for inner_batch in batch:
                            # Compute loss and adapt value function [L9]
                            loss, value_inner, mc_inner, mc_std_inner = self.value_function_loss_on_batch(f_value_function_j, inner_batch, inner=True)
                            inner_values.append(value_inner.item())
                            inner_mc_means.append(mc_inner.item())
                            inner_mc_stds.append(mc_std_inner.item())
                            diff_value_opt.step(loss)
                            inner_value_losses.append(loss.item())

                    # Collect grads for the value function update in the outer loop [L14],
                    #  which is not actually performed here
                    meta_value_function_loss, value, mc, mc_std = self.value_function_loss_on_batch(f_value_function_j, meta_batch)
                    meta_value_grad_j = A.grad(meta_value_function_loss, f_value_function_j.parameters(time=0))

                    outer_values.append(value.item())
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
                opt = O.SGD(adaptation_policy_j.parameters(), lr=self._inner_policy_lr)
                with higher.innerloop_ctx(adaptation_policy_j, opt) as (f_adaptation_policy_j, diff_policy_opt):
                    if self._inner_policy_lr > 0 and len(self._env.tasks) > 1:
                        for inner_batch in batch:
                            # Compute loss and adapt policy [L10]
                            loss = self.adaptation_policy_loss_on_batch(f_adaptation_policy_j, adapted_value_function, inner_batch, inner=True)
                            diff_policy_opt.step(loss)
                            inner_policy_losses.append(loss.item())
                            
                    meta_policy_loss = self.adaptation_policy_loss_on_batch(f_adaptation_policy_j, adapted_value_function, meta_batch)
                    meta_policy_grad_j = A.grad(meta_policy_loss, f_adaptation_policy_j.parameters(time=0), retain_graph=True)

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
                adapted_trajectory, adapted_reward = self._rollout_policy(adaptation_policies_i[0], self._env, test=False)
                train_rewards.append(adapted_reward)

                if not (self._args.offline or self._args.offline_inner):
                    if self._args.explore:
                        exploration_trajectory, _ = self._rollout_policy(self._exploration_policy, self._env, test=False)
                        inner_buffer.add_trajectory(exploration_trajectory)
                    else:
                        inner_buffer.add_trajectory(adapted_trajectory)
                if not (self._args.offline or self._args.offline_outer):
                    outer_buffer.add_trajectory(adapted_trajectory)

            if train_step_idx % self._visualization_interval == 0:
                if self._args.render:
                    print(f'Visualizing task {i}, test rollout')
                test_trajectory, test_reward = self._rollout_policy(adaptation_policies_i[0], self._env, test=False,
                                                                    render=self._args.render)
                if self._args.render:
                    print(f'Reward: {test_reward}')
                rollouts.append(test_trajectory)
                test_rewards.append(test_reward)

            if writer is not None:
                if len(inner_value_losses):
                    writer.add_scalar(f'Loss_Value_Inner/Task_{i}', np.mean(inner_value_losses), train_step_idx)
                    writer.add_scalar(f'Loss_Policy_Inner/Task_{i}', np.mean(inner_policy_losses), train_step_idx)
                    writer.add_scalar(f'Value_Mean_Inner/Task_{i}', np.mean(inner_values), train_step_idx)
                    writer.add_scalar(f'MC_Mean_Inner/Task_{i}', np.mean(inner_mc_means), train_step_idx)
                    writer.add_scalar(f'MC_std_Inner/Task_{i}', np.mean(inner_mc_stds), train_step_idx)
                writer.add_scalar(f'Value_Mean_Outer/Task_{i}', np.mean(outer_values), train_step_idx)
                writer.add_scalar(f'MC_Mean_Outer/Task_{i}', np.mean(outer_mc_means), train_step_idx)
                writer.add_scalar(f'MC_std_Outer/Task_{i}', np.mean(outer_mc_stds), train_step_idx)
                writer.add_scalar(f'Loss_Value_Outer/Task_{i}', np.mean(meta_value_losses), train_step_idx)
                writer.add_scalar(f'Loss_Policy_Outer/Task_{i}', np.mean(meta_policy_losses), train_step_idx)
                if train_step_idx % self._visualization_interval == 0:
                    writer.add_scalar(f'Reward_Test/Task_{i}', test_reward, train_step_idx)
                if train_step_idx % self._gradient_steps_per_iteration == 0:
                    writer.add_scalar(f'Reward_Train/Task_{i}', adapted_reward, train_step_idx)

        if self._args.eval:
            return rollouts, test_rewards, train_rewards, meta_value_losses, meta_policy_losses, value_functions_i[-1]

        # Meta-update value function [L14]
        grad = self.update_model_with_grads(self._value_function, meta_value_grads, self._value_function_optimizer, self._grad_clip)
        if grad is not None:
            writer.add_scalar(f'Value_Outer_Grad', grad, train_step_idx)

        # Meta-update adaptation policy [L15]
        grad = self.update_model_with_grads(self._adaptation_policy, meta_policy_grads, self._adaptation_policy_optimizer, self._grad_clip)
        if grad is not None:
            writer.add_scalar(f'Policy_Outer_Grad', grad, train_step_idx)

        ############################################################################3
        # Update exploration policy [WIP] [L16]
        ############################################################################3
        if self._args.explore:
            if self._args.cvae:
                self.train_vae_exploration(train_step_idx, batches, meta_batches, adaptation_policies, value_functions, writer)
            else:
                self.train_awr_exploration(train_step_idx, batches, meta_batches, adaptation_policies, value_functions, writer)
        ############################################################################3

        return rollouts, test_rewards, train_rewards, meta_value_losses, meta_policy_losses, [vfs[-1] for vfs in value_functions]

    def train(self):
        log_path = f'{self._log_dir}/{self._name}'
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        tensorboard_log_path = f'{log_path}/tb'
        if not os.path.exists(tensorboard_log_path):
            os.makedirs(tensorboard_log_path)
        summary_writer = SummaryWriter(tensorboard_log_path)

        # Gather initial trajectory rollouts
        if not self._args.load_inner_buffer or not self._args.load_outer_buffer:
            print('Gathering initial trajectories...')
            exploration_rewards = np.zeros((self._initial_trajectories, len(self._env.tasks)))
            for j in range(self._initial_trajectories):
                for i, (inner_buffer, outer_buffer) in enumerate(zip(self._inner_buffers, self._outer_buffers)):
                    self._env.set_task_idx(i)
                    if self._args.render_exploration:
                        print(f'Task {i}, trajectory {j}')
                    trajectory, reward = self._rollout_policy(self._exploration_policy, self._env, random=self._args.random, render=self._args.render_exploration)
                    exploration_rewards[j,i] = reward
                    if self._args.render_exploration:
                        print(f'Reward: {reward}')
                    inner_buffer.add_trajectory(trajectory, force=True)
                    if not self._args.load_outer_buffer:
                        outer_buffer.add_trajectory(trajectory, force=True)

            if self._args.debug:
                print(f'Mean exploration rewards: {exploration_rewards.mean(0)}')
                print(f'Positive exploration rewards: {(exploration_rewards>0).mean(0)}')
                    
        for t in range(self._training_iterations):
            rollouts, test_rewards, train_rewards, value, policy, vfs = self.train_step(t, summary_writer)

            if not self._silent:
                if len(train_rewards):
                    print(f'{t}: {train_rewards}, {np.mean(value)}, {np.mean(policy)}')
                    if self._args.eval:
                        for idx, vf in enumerate(vfs):
                            print(idx, argmax(vf, torch.zeros(self._observation_dim, device=self._device)))

            if len(test_rewards):
                summary_writer.add_scalar(f'Reward_Test/Mean', np.mean(test_rewards), t)
            if len(train_rewards):
                summary_writer.add_scalar(f'Reward_Train/Mean', np.mean(train_rewards), t)

                if self._args.target_reward is not None:
                    if np.mean(train_rewards) > self._args.target_reward:
                        print('Target reward reached; breaking')
                        break
                
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
                if self._args.explore:
                    torch.save(self._exploration_policy.state_dict(), f'{log_path}/ep_LATEST.pt')
                    torch.save(self._exploration_policy.state_dict(), f'{log_path}/ep_LATEST_.pt')
                if self._args.save_buffers:
                    for i, (inner_buffer, outer_buffer) in enumerate(zip(self._inner_buffers, self._outer_buffers)):
                        inner_buffer.save(f'{log_path}/inner_buffer_{i}')
                        outer_buffer.save(f'{log_path}/outer_buffer_{i}')
                    
