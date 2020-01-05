import argparse
from copy import deepcopy
from typing import List, Optional
import os
import itertools
import math
import random
import time
import json

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
from src.utils import ReplayBuffer, Experience, argmax, kld, RunningEstimator


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


def print_(s: str, c: bool, end=None):
    if not c:
        if end is not None:
            print(s, end=end)
        else:
            print(s)


class MAMLRAWR(object):
    def __init__(self, args: argparse.Namespace, env: Env, log_dir: str, name: str = None,
                 policy_hidden_layers: List[int] = [32, 32], value_function_hidden_layers: List[int] = [32, 32],
                 training_iterations: int = 20000, action_sigma: float = 0.2,
                 visualization_interval: int = 100, silent: bool = False, replay_buffer_length: int = 1000,
                 gradient_steps_per_iteration: int = 1, discount_factor: float = 0.99, grad_clip: float = 100.,
                 bias_linear: bool = False):
        self._env = env
        self._log_dir = log_dir
        self._name = name if name is not None else 'throwaway_test_run'
        self._args = args
        self._start_time = time.time()
        self._observation_dim = env.observation_space.shape[0]
        self._action_dim = env_action_dim(env)

        policy_head = [32, 1] if args.advantage_head_coef is not None else None
        
        self._adaptation_policy = MLP([self._observation_dim] +
                                      policy_hidden_layers +
                                      [self._action_dim],
                                      final_activation=torch.tanh,
                                      bias_linear=bias_linear,
                                      extra_head_layers=policy_head).to(args.device)
        if args.cvae:
            self._exploration_policy = CVAE(self._observation_dim, self._action_dim, args.latent_dim).to(args.device)
        else:
            self._exploration_policy = MLP([self._observation_dim] +
                                           policy_hidden_layers +
                                           [self._action_dim],
                                           final_activation=torch.tanh).to(args.device)

        self._q_function = MLP([self._observation_dim + self._action_dim] + value_function_hidden_layers + [1],
                                   bias_linear=bias_linear).to(args.device)
        self._value_function = MLP([self._observation_dim] + value_function_hidden_layers + [1],
                                   bias_linear=bias_linear).to(args.device)
        self._policy_lrs = [nn.Parameter(torch.tensor(args.inner_policy_lr, device=args.device, requires_grad=True)) for _ in self._adaptation_policy.parameters()]
        self._value_lrs = [nn.Parameter(torch.tensor(args.inner_value_lr, device=args.device, requires_grad=True)) for _ in self._value_function.parameters()]
        print(self._adaptation_policy.seq[0]._linear.weight.mean())
        
        self._adaptation_policy_optimizer = O.Adam(self._adaptation_policy.parameters(), lr=args.outer_policy_lr)
        self._q_function_optimizer = O.Adam(self._q_function.parameters(), lr=args.outer_value_lr)
        self._value_function_optimizer = O.Adam(self._value_function.parameters(), lr=args.outer_value_lr)
        if args.train_exploration or args.sample_exploration_inner:
            self._exploration_policy_optimizer = O.Adam(self._exploration_policy.parameters(), lr=args.exploration_lr)
        
        if args.vf_archive is not None:
            print_(f'Loading value function archive from: {args.vf_archive}', silent)
            self._value_function.load_state_dict(torch.load(args.vf_archive, map_location=args.device))
        if args.ap_archive is not None:
            print_(f'Loading policy archive from: {args.ap_archive}', silent)
            self._adaptation_policy.load_state_dict(torch.load(args.ap_archive, map_location=args.device))
        if args.ep_archive is not None:
            print_(f'Loading exploration policy archive from: {args.ep_archive}', silent)
            self._exploration_policy.load_state_dict(torch.load(args.ep_archive, map_location=args.device))
            
        inner_buffer = args.buffer_paths if args.load_inner_buffer else [None for _ in self._env.tasks]
        outer_buffer = args.buffer_paths if args.load_outer_buffer else [None for _ in self._env.tasks]
        self._inner_buffers = [ReplayBuffer(self._env._max_episode_steps, self._env.observation_space.shape[0], env_action_dim(self._env),
                                            max_trajectories=replay_buffer_length, discount_factor=discount_factor,
                                            immutable=args.offline or args.offline_inner, load_from=inner_buffer[i], silent=silent,
                                            trim_suffix=args.trim_suffix)
                               for i, task in enumerate(self._env.tasks)]
        self._outer_buffers = [ReplayBuffer(self._env._max_episode_steps, self._env.observation_space.shape[0], env_action_dim(self._env),
                                            max_trajectories=replay_buffer_length, discount_factor=discount_factor,
                                            immutable=args.offline or args.offline_outer, load_from=outer_buffer[i], silent=silent,
                                            trim_suffix=args.trim_suffix)
                               for i, task in enumerate(self._env.tasks)]

        self._training_iterations = training_iterations
        self._inner_policy_lr, self._inner_value_lr = args.inner_policy_lr, args.inner_value_lr
        self._adaptation_temperature, self._exploration_temperature = args.adaptation_temp, args.exploration_temp
        self._device = torch.device(args.device)
        self._cpu = torch.device('cpu')
        self._advantage_clamp = np.log(args.exp_advantage_clip)
        self._action_sigma = action_sigma
        self._visualization_interval = visualization_interval
        self._silent = silent
        self._gradient_steps_per_iteration = gradient_steps_per_iteration
        self._grad_clip = grad_clip
        self._env_seeds = np.random.randint(1e10, size=(int(1e7),))
        self._rollout_counter = 0
        self._value_estimators = [RunningEstimator() for _ in self._env.tasks]
        self._q_estimators = [RunningEstimator() for _ in self._env.tasks]
        
    #################################################################
    ################# SUBROUTINES FOR TRAINING ######################
    #################################################################
    #@profile
    def _rollout_policy(self, policy: MLP, env: Env, test: bool = False, random: bool = False, render: bool = False) -> List[Experience]:
        env.seed(self._env_seeds[self._rollout_counter].item())
        self._rollout_counter += 1
        device = self._cpu
        trajectory = []
        old_device = list(policy.parameters())[0].device
        if device is not self._device:
            policy.to(self._cpu)
        state = env.reset()
        if render:
            env.render()
        done = False
        total_reward = 0
        episode_t = 0

        if isinstance(policy, CVAE):
            policy.fix()

        while not done:
            if not random:
                with torch.no_grad():
                    action_sigma = self._action_sigma
                    if isinstance(policy, CVAE):
                        mu, action_sigma = policy(torch.tensor(state, device=device).unsqueeze(0).float())
                        mu = mu.squeeze()
                        action_sigma = action_sigma.squeeze().clamp(max=0.5)
                    else:
                        mu = policy(torch.tensor(state, device=device).unsqueeze(0).float()).squeeze()

                    if test:
                        action = mu
                    else:
                        action = mu + torch.empty_like(mu).normal_() * action_sigma

                    log_prob = D.Normal(mu, torch.empty_like(mu).fill_(self._action_sigma)).log_prob(action).to(self._cpu).numpy().sum()
                    action = action.squeeze().to(self._cpu).numpy().clip(min=env.action_space.low, max=env.action_space.high)
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

        if isinstance(policy, CVAE):
            policy.unfix()
        policy.to(old_device)
        return trajectory, total_reward

    #@profile
    def mc_value_estimates_on_batch(self, value_function, batch):
        with torch.no_grad():
            if self._args.no_bootstrap:
                mc_value_estimates = batch[:,-1:]
            else:
                terminal_state_value_estimates = value_function(batch[:,self._observation_dim * 2 + self._action_dim:self._observation_dim * 3 + self._action_dim])
                terminal_factors = batch[:, -4:-3] # I know this magic number indexing is heinous... I'm sorry
                mc_value_estimates = batch[:,-1:] + terminal_factors * terminal_state_value_estimates

        return mc_value_estimates

    #@profile
    def q_function_loss_on_batch(self, q_function, value_function, batch, inner: bool = False, task_idx: int = None):
        q_estimates = q_function(torch.cat((batch[:,:self._observation_dim], batch[:,self._observation_dim:self._observation_dim+self._action_dim]), -1))
        with torch.no_grad():
            value_estimates = value_function(batch[:,self._observation_dim + self._action_dim:self._observation_dim * 2 + self._action_dim])
            #mc_value_estimates = self.mc_value_estimates_on_batch(value_function, batch)

        targets = batch[:,-2] + value_estimates

        if self._args.normalize_values or (self._args.normalize_values_outer and not inner):
            if task_idx is not None:
                self._q_estimators[task_idx].add(targets)
                factor = self._q_estimators[task_idx].std() + 1
            else:
                factor = targets.std() + 1
        else:
            factor = 1

        return (q_estimates - targets).div(factor).pow(2).mean()

    def exploration_weights(self, batch: List[torch.tensor], clamp: bool = True):
        lens = [traj.shape[0] for traj in batch]
        cumlens = [0] + np.cumsum(lens)[:-1].tolist()
        raveled_batch = torch.cat(batch)

        original_log_probs = raveled_batch[:,-5]
        
        exploration_mu = self._exploration_policy(raveled_batch[:,:self._observation_dim])
        exploration_sigma = torch.empty_like(exploration_mu).fill_(self._action_sigma)
        exploration_distribution = D.Normal(exploration_mu, exploration_sigma)
        exploration_log_probs = exploration_distribution.log_prob(raveled_batch[:,self._observation_dim:self._observation_dim + self._action_dim]).sum(-1)

        original_trajectory_log_probs = torch.cat([original_log_probs[start:start+len_-1].sum().unsqueeze(0) for start, len_ in zip(cumlens, lens)])
        exploration_trajectory_log_probs = torch.cat([exploration_log_probs[start:start+len_-1].sum().unsqueeze(0) for start, len_ in zip(cumlens, lens)])
        original_trajectory_action_log_probs = torch.cat([original_log_probs[start+len_-1].unsqueeze(0) for start, len_ in zip(cumlens, lens)])
        exploration_trajectory_action_log_probs = torch.cat([exploration_log_probs[start+len_-1].unsqueeze(0) for start, len_ in zip(cumlens, lens)])
        exploration_weights_no_action_logits_ = exploration_trajectory_log_probs - original_trajectory_log_probs
        exploration_weights_logits_ = exploration_weights_no_action_logits_ + exploration_trajectory_action_log_probs - original_trajectory_action_log_probs

        if clamp:
            exploration_weights_logits = exploration_weights_logits_.clamp(min=-1,max=1)
            exploration_weights_no_action_logits = exploration_weights_no_action_logits_.clamp(min=-1,max=1)
        else:
            exploration_weights_logits = exploration_weights_logits_
            exploration_weights_no_action_logits = exploration_weights_no_action_logits_
        exploration_weights = exploration_weights_logits.exp()
        exploration_weights_no_action = exploration_weights_no_action_logits.exp()

        return exploration_weights, exploration_weights_no_action, exploration_weights_logits_

    #@profile
    def value_function_loss_on_batch(self, value_function, batch, inner: bool = False, task_idx: int = None, iweights: torch.tensor = None):
        value_estimates = value_function(batch[:,:self._observation_dim])
        with torch.no_grad():
            mc_value_estimates = self.mc_value_estimates_on_batch(value_function, batch)

        targets = mc_value_estimates
        if self._args.normalize_values or (self._args.normalize_values_outer and not inner):
            if task_idx is not None:
                self._value_estimators[task_idx].add(targets)
                factor = self._value_estimators[task_idx].std() + 1
            else:
                factor = targets.std() + 1
        else:
            factor = 1

        if self._args.huber and not inner:
            losses = F.smooth_l1_loss(value_estimates / factor, targets / factor, reduction='none')
        else:
            losses = (value_estimates - targets).div(factor).pow(2)

        if iweights is not None:
            losses = losses * iweights
        
        return losses.mean(), value_estimates.mean(), mc_value_estimates.mean(), mc_value_estimates.std()

    #@profile
    def adaptation_policy_loss_on_batch(self, policy, q_function, value_function, batch, inner: bool = False, iweights: torch.tensor = None):
        with torch.no_grad():
            value_estimates = value_function(batch[:,:self._observation_dim])
            if q_function is not None:
                action_value_estimates = q_function(torch.cat((batch[:,:self._observation_dim], batch[:,self._observation_dim:self._observation_dim+self._action_dim]), -1))
            else:
                action_value_estimates = self.mc_value_estimates_on_batch(value_function, batch)

            advantages = (action_value_estimates - value_estimates).squeeze(-1)
            if self._args.no_norm:
                weights = advantages.clamp(max=self._advantage_clamp).exp()
            else:
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

        #if self._args.iw_exploration and inner:
        if iweights is not None:
            losses = losses * iweights
        
        if inner:
            if self._args.advantage_head_coef is not None:
                losses = losses + self._args.advantage_head_coef * (advantage_prediction.squeeze() - advantages) ** 2

        return losses.mean(), advantages.mean(), weights.mean()

    def update_model_with_grads(self, model: nn.Module, grads: list, optimizer: torch.optim.Optimizer, clip: float = None, extra_grad: list = None):
        for idx, parameter in enumerate(model.parameters()):
            grads_ = []
            for i in range(len(grads)):
                grads_.append(grads[i][idx])
            parameter.grad = sum(grads_) / len(grads_)
            if extra_grad is not None:
                parameter.grad += extra_grad[idx]

        if clip is not None:
            grad = torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        else:
            grad = None

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

            print_(i, train_step_idx, weights, self._silent)

            for j, batch in enumerate(pyt_batch):
                batch = batch.view(-1, batch.shape[-1])
                original_action = batch[:,self._observation_dim:self._observation_dim + self._action_dim]
                original_obs = batch[:,:self._observation_dim]
                action = self._exploration_policy(original_obs)
                log_probs = D.Normal(action, torch.empty_like(action).fill_(self._action_sigma)).log_prob(original_action).sum(-1)

                loss = -log_probs.mean() * weights[j]
                exploration_loss = exploration_loss + loss

        exploration_loss.backward()
        grad = torch.nn.utils.clip_grad_norm_(self._exploration_policy.parameters(), self._grad_clip)
        writer.add_scalar(f'Explore_Grad', grad, train_step_idx)
        writer.add_scalar(f'Explore_Loss', exploration_loss.item(), train_step_idx)

        self._exploration_policy_optimizer.step()
        self._exploration_policy_optimizer.zero_grad()


    #@profile
    def train_vae_exploration(self, train_step_idx: int, writer: SummaryWriter):
        kld_loss = 0
        recon_loss = 0
        for i, buf in enumerate(self._outer_buffers):
            batch, trajectories = buf.sample(self._args.exploration_batch_size, trajectory=True, complete=True)
            batch = torch.tensor(batch, device=self._device)

            trajectories = torch.tensor(trajectories, device=self._device)
            traj = trajectories[:,:,:self._observation_dim + self._action_dim].permute(0,2,1)[:,:,::self._args.cvae_skip]

            obs = batch[:,:self._observation_dim]
            action = batch[:,self._observation_dim:self._observation_dim + self._action_dim]
            
            pz_t, z = self._exploration_policy.encode(traj, sample=True)
            pz = self._exploration_policy.prior()

            py_zs = self._exploration_policy.decode(z, obs)
            mu_y = py_zs[:,:py_zs.shape[-1]//2]
            std_y = (py_zs[:,py_zs.shape[-1]//2:] / 2).exp()

            d_y = D.Normal(mu_y, std_y)

            kld_loss += kld(pz_t, pz).mean()
            recon_loss += -d_y.log_prob(action).sum(-1).mean()

        kld_coef = (torch.tensor(train_step_idx).float() / 10000 - 10).sigmoid().to(self._device)
        exploration_loss = kld_coef * kld_loss + recon_loss
        exploration_loss.backward()
        
        grad = torch.nn.utils.clip_grad_norm_(self._exploration_policy.parameters(), self._grad_clip)

        writer.add_scalar(f'Explore_Grad', grad, train_step_idx)
        writer.add_scalar(f'Explore_Loss', exploration_loss.item(), train_step_idx)
        writer.add_scalar(f'Explore_Loss_Recon', recon_loss.item(), train_step_idx)
        writer.add_scalar(f'Explore_Loss_KLD', kld_loss.item(), train_step_idx)
        writer.add_histogram(f'Explore_Mu', mu_y.detach().cpu().numpy(), train_step_idx)
        writer.add_histogram(f'Explore_Sigma', std_y.detach().cpu().numpy(), train_step_idx)

        self._exploration_policy_optimizer.step()
        self._exploration_policy_optimizer.zero_grad()

    # This function is the body of the main training loop [L4]
    # At every iteration, it adds rollouts from the exploration policy and one of the adapted policies
    #  to the replay buffer. It also updates the adaptation value function, adaptation policy, and
    #  exploration policy
    #@profile
    def train_step(self, train_step_idx: int, writer: Optional[SummaryWriter] = None):
        batches = []
        meta_batches = []
        q_functions = []
        value_functions = []
        adaptation_policies = []
        meta_q_grads = []
        meta_value_grads = []
        meta_policy_grads = []
        exploration_grads = []
        test_rewards = []
        train_rewards = []
        rollouts = []
        for i, (inner_buffer, outer_buffer) in enumerate(zip(self._inner_buffers, self._outer_buffers)):
            self._env.set_task_idx(i)
            
            # Sample J training batches for independent adaptations [L7]
            if self._args.iw_exploration:
                np_batch, np_trajectories = inner_buffer.sample(self._args.inner_batch_size * self._args.maml_steps, trajectory=True)
                pyt_trajectories = [torch.tensor(np_traj, requires_grad=False).to(self._device) for np_traj in np_trajectories]
            else:
                np_batch = inner_buffer.sample(self._args.inner_batch_size * self._args.maml_steps, trajectory=False)
            np_batch = np_batch.reshape((self._args.maml_steps, self._args.inner_batch_size, -1))
            pyt_batch = torch.tensor(np_batch, requires_grad=False).to(self._device)
            batches.append(pyt_batch)

            meta_batch = torch.tensor(outer_buffer.sample(self._args.batch_size), requires_grad=False).to(self._device)
            meta_batches.append(meta_batch)

            value_exploration_grads = []
            policy_exploration_grads = []
            inner_q_losses = []
            meta_q_losses = []
            inner_value_losses = []
            meta_value_losses = []
            inner_policy_losses = []
            meta_policy_losses = []
            inner_mc_means, inner_mc_stds = [], []
            outer_mc_means, outer_mc_stds = [], []
            inner_values, outer_values = [], []
            inner_weights, outer_weights = [], []
            inner_advantages, outer_advantages = [], []
            if self._args.iw_exploration:
                iweights_, iweights_no_action_, iweight_logits = self.exploration_weights(pyt_trajectories, clamp=not self._args.noclamp)
                iweights = iweights_.detach().cpu().numpy()
                if train_step_idx % self._visualization_interval == 0:
                    writer.add_histogram(f'IW_Hist/Task_{i}', iweights, train_step_idx)
                writer.add_scalar(f'IW_Mean/Task_{i}', np.mean(iweights), train_step_idx)
                writer.add_scalar(f'IW_STD/Task_{i}', np.std(iweights), train_step_idx)
                writer.add_scalar(f'IW_Median/Task_{i}', np.median(iweights), train_step_idx)
            else:
                iweights_ = None
                iweights_no_action_ = None

            ##################################################################################################
            # Adapt value function and collect meta-gradients
            ##################################################################################################
            opt = O.SGD(self._value_function.parameters(), lr=self._inner_value_lr)
            with higher.innerloop_ctx(self._value_function, opt) as (f_value_function, diff_value_opt):
                if self._inner_value_lr > 0 and len(self._env.tasks) > 1:
                    for inner_batch in pyt_batch:
                        # Compute loss and adapt value function [L9]
                        loss, value_inner, mc_inner, mc_std_inner = self.value_function_loss_on_batch(f_value_function, inner_batch, inner=True, task_idx=i)#, iweights=iweights_no_action_)
                        inner_values.append(value_inner.item())
                        inner_mc_means.append(mc_inner.item())
                        inner_mc_stds.append(mc_std_inner.item())
                        diff_value_opt.step(loss)
                        inner_value_losses.append(loss.item())

                # Collect grads for the value function update in the outer loop [L14],
                #  which is not actually performed here
                meta_value_function_loss, value, mc, mc_std = self.value_function_loss_on_batch(f_value_function, meta_batch, task_idx=i)
                meta_value_grad = A.grad(meta_value_function_loss, f_value_function.parameters(time=0), retain_graph=self._args.iw_exploration)
                if self._args.iw_exploration:
                    value_exploration_grad = A.grad(meta_value_function_loss, self._exploration_policy.parameters(), retain_graph=True)

                outer_values.append(value.item())
                outer_mc_means.append(mc.item())
                outer_mc_stds.append(mc_std.item())
                meta_value_losses.append(meta_value_function_loss.item())
                meta_value_grads.append(meta_value_grad)
                if self._args.iw_exploration:
                    value_exploration_grads.append(value_exploration_grad)
                value_functions.append(f_value_function)
            ##################################################################################################

            ##################################################################################################
            # Adapt value function and collect meta-gradients
            ##################################################################################################
            if self._args.q:
                q_opt = O.SGD(self._q_function.parameters(), lr=self._inner_value_lr)
                with higher.innerloop_ctx(self._q_function, q_opt) as (f_q_function, diff_q_opt):
                    if self._inner_value_lr > 0 and len(self._env.tasks) > 1:
                        for inner_batch in pyt_batch:
                            # Compute loss and adapt value function [L9]
                            loss = self.q_function_loss_on_batch(f_q_function, value_functions[-1], inner_batch, inner=True, task_idx=i)
                            diff_value_opt.step(loss)
                            inner_q_losses.append(loss.item())

                    # Collect grads for the value function update in the outer loop [L14],
                    #  which is not actually performed here
                    meta_q_function_loss = self.q_function_loss_on_batch(f_q_function, value_functions[-1], meta_batch, task_idx=i)
                    meta_q_grad = A.grad(meta_q_function_loss, f_q_function.parameters(time=0))

                    meta_q_losses.append(meta_q_function_loss.item())
                    meta_q_grads.append(meta_q_grad)
                    q_functions.append(f_q_function)

            ##################################################################################################
            # Adapt policy and collect meta-gradients
            ##################################################################################################
            adapted_value_function = value_functions[-1]
            adapted_q_function = q_functions[-1] if self._args.q else None
            opt = O.SGD(self._adaptation_policy.parameters(), lr=self._inner_policy_lr)
            with higher.innerloop_ctx(self._adaptation_policy, opt) as (f_adaptation_policy, diff_policy_opt):
                if self._inner_policy_lr > 0 and len(self._env.tasks) > 1:
                    for inner_batch in pyt_batch:
                        # Compute loss and adapt policy [L10]
                        loss, adv, weight = self.adaptation_policy_loss_on_batch(f_adaptation_policy, adapted_q_function,
                                                                                 adapted_value_function, inner_batch, inner=True, iweights=iweights_)
                        diff_policy_opt.step(loss)
                        inner_policy_losses.append(loss.item())
                        inner_advantages.append(adv.item())
                        inner_weights.append(weight.item())

                meta_policy_loss, outer_adv, outer_weight = self.adaptation_policy_loss_on_batch(f_adaptation_policy, adapted_q_function,
                                                                                                 adapted_value_function, meta_batch)
                meta_policy_grad = A.grad(meta_policy_loss, f_adaptation_policy.parameters(time=0), retain_graph=self._args.iw_exploration)
                if self._args.iw_exploration:
                    policy_exploration_grad = A.grad(meta_policy_loss, self._exploration_policy.parameters(), retain_graph=self._args.exploration_reg is not None)

                # Collect grads for the adaptation policy update in the outer loop [L15],
                #  which is not actually performed here
                outer_weights.append(outer_weight.item())
                outer_advantages.append(outer_adv.item())
                meta_policy_losses.append(meta_policy_loss.item())
                meta_policy_grads.append(meta_policy_grad)
                if self._args.iw_exploration:
                    policy_exploration_grads.append(policy_exploration_grad)
                adaptation_policies.append(f_adaptation_policy)
            ##################################################################################################
            
            if self._args.iw_exploration:
                exploration_grads.append(value_exploration_grads)
                exploration_grads.append(policy_exploration_grads)

            # Sample adapted policy trajectory, add to replay buffer i [L12]
            if train_step_idx % self._gradient_steps_per_iteration == 0:
                adapted_trajectory, adapted_reward = self._rollout_policy(adaptation_policies[-1], self._env, test=False)
                train_rewards.append(adapted_reward)

                if not (self._args.offline or self._args.offline_inner):
                    if self._args.sample_exploration_inner:
                        exploration_trajectory, _ = self._rollout_policy(self._exploration_policy, self._env, test=False)
                        inner_buffer.add_trajectory(exploration_trajectory)
                    else:
                        inner_buffer.add_trajectory(adapted_trajectory)
                if not (self._args.offline or self._args.offline_outer):
                    outer_buffer.add_trajectory(adapted_trajectory)

            if train_step_idx % self._visualization_interval == 0:
                if self._args.render:
                    print_(f'Visualizing task {i}, test rollout', self._silent)
                test_trajectory, test_reward = self._rollout_policy(adaptation_policies[-1], self._env, test=False,
                                                                    render=self._args.render)
                if self._args.render:
                    print_(f'Reward: {test_reward}', self._silent)
                rollouts.append(test_trajectory)
                test_rewards.append(test_reward)

            if writer is not None:
                if len(inner_value_losses):
                    if self._args.q:
                        writer.add_scalar(f'Loss_Q_Inner/Task_{i}', np.mean(inner_q_losses), train_step_idx)
                    writer.add_scalar(f'Loss_Value_Inner/Task_{i}', np.mean(inner_value_losses), train_step_idx)
                    writer.add_scalar(f'Loss_Policy_Inner/Task_{i}', np.mean(inner_policy_losses), train_step_idx)
                    writer.add_scalar(f'Value_Mean_Inner/Task_{i}', np.mean(inner_values), train_step_idx)
                    writer.add_scalar(f'Advantage_Mean_Inner/Task_{i}', np.mean(inner_advantages), train_step_idx)
                    writer.add_scalar(f'Weight_Mean_Inner/Task_{i}', np.mean(inner_weights), train_step_idx)
                    writer.add_scalar(f'MC_Mean_Inner/Task_{i}', np.mean(inner_mc_means), train_step_idx)
                    writer.add_scalar(f'MC_std_Inner/Task_{i}', np.mean(inner_mc_stds), train_step_idx)
                writer.add_scalar(f'Value_Mean_Outer/Task_{i}', np.mean(outer_values), train_step_idx)
                writer.add_scalar(f'Weight_Mean_Outer/Task_{i}', np.mean(outer_weights), train_step_idx)
                writer.add_scalar(f'Advantage_Mean_Outer/Task_{i}', np.mean(outer_advantages), train_step_idx)
                writer.add_scalar(f'MC_Mean_Outer/Task_{i}', np.mean(outer_mc_means), train_step_idx)
                writer.add_scalar(f'MC_std_Outer/Task_{i}', np.mean(outer_mc_stds), train_step_idx)
                if train_step_idx % 100 == 0:
                    writer.add_histogram(f'Value_LR', np.asarray(self._value_lrs), train_step_idx)
                    writer.add_histogram(f'Policy_LR', np.asarray(self._policy_lrs), train_step_idx)
                if self._args.q:
                    writer.add_scalar(f'Loss_Q_Outer/Task_{i}', np.mean(meta_q_losses), train_step_idx)
                writer.add_scalar(f'Loss_Value_Outer/Task_{i}', np.mean(meta_value_losses), train_step_idx)
                writer.add_scalar(f'Loss_Policy_Outer/Task_{i}', np.mean(meta_policy_losses), train_step_idx)
                if train_step_idx % self._visualization_interval == 0:
                    writer.add_scalar(f'Reward_Test/Task_{i}', test_reward, train_step_idx)
                if train_step_idx % self._gradient_steps_per_iteration == 0:
                    writer.add_scalar(f'Reward_Train/Task_{i}', adapted_reward, train_step_idx)

        if self._args.eval:
            return rollouts, test_rewards, train_rewards, meta_value_losses, meta_policy_losses, value_functions[-1]

        # Meta-update value function [L14]
        grad = self.update_model_with_grads(self._value_function, meta_value_grads, self._value_function_optimizer, self._grad_clip)
        if grad is not None:
            writer.add_scalar(f'Value_Outer_Grad', grad, train_step_idx)

        # Meta-update Q function [L14]
        if self._args.q:
            grad = self.update_model_with_grads(self._q_function, meta_q_grads, self._q_function_optimizer, self._grad_clip)
            if grad is not None:
                writer.add_scalar(f'Q_Outer_Grad', grad, train_step_idx)

        # Meta-update adaptation policy [L15]
        grad = self.update_model_with_grads(self._adaptation_policy, meta_policy_grads, self._adaptation_policy_optimizer, self._grad_clip)
        if grad is not None:
            writer.add_scalar(f'Policy_Outer_Grad', grad, train_step_idx)

        ############################################################################3
        # Update exploration policy [WIP] [L16]
        ############################################################################3
        if self._args.train_exploration:
            if self._args.cvae:
                self.train_vae_exploration(train_step_idx, writer)
            elif self._args.iw_exploration:
                if self._args.exploration_reg is not None:
                    exploration_reg_loss = self._args.exploration_reg * iweight_logits.pow(2).mean()
                    extra_grad = A.grad(exploration_reg_loss, self._exploration_policy.parameters())
                    writer.add_scalar(f'Exploration_Reg_Loss', exploration_reg_loss.detach().item(), train_step_idx)
                    writer.add_scalar(f'Exploration_Reg_Grad', torch.cat([g.view(-1) for g in extra_grad]).norm(2), train_step_idx)
                else:
                    extra_grad = None
                grad = self.update_model_with_grads(self._exploration_policy, exploration_grads, self._exploration_policy_optimizer, self._grad_clip, extra_grad=extra_grad)
                if grad is not None:
                    writer.add_scalar(f'Exploration_Grad', grad, train_step_idx)
            else:
                self.train_awr_exploration(train_step_idx, batches, meta_batches, adaptation_policies, value_functions, writer)
        ############################################################################3

        return rollouts, test_rewards, train_rewards, meta_value_losses, meta_policy_losses, value_functions

    #@profile
    def train(self):
        log_path = f'{self._log_dir}/{self._name}'
        if not os.path.exists(log_path):
            os.makedirs(log_path)

        with open(f'{log_path}/args.txt', 'w') as args_file:
            json.dump(self._args.__dict__, args_file)

        tensorboard_log_path = f'{log_path}/tb'
        if not os.path.exists(tensorboard_log_path):
            os.makedirs(tensorboard_log_path)
        summary_writer = SummaryWriter(tensorboard_log_path)

        # Gather initial trajectory rollouts
        if not self._args.load_inner_buffer or not self._args.load_outer_buffer:
            print_('Gathering initial trajectories...', self._silent)
            behavior_policy = self._exploration_policy if self._args.sample_exploration_inner else self._adaptation_policy
            exploration_rewards = np.zeros((self._args.initial_rollouts, len(self._env.tasks)))
            for j in range(self._args.initial_rollouts):
                for i, (inner_buffer, outer_buffer) in enumerate(zip(self._inner_buffers, self._outer_buffers)):
                    print_(f'{j+1,i+1}/{self._args.initial_rollouts,len(self._inner_buffers)}\r', self._silent, end='')
                    self._env.set_task_idx(i)
                    if self._args.render_exploration:
                        print_(f'Task {i}, trajectory {j}', self._silent)
                    trajectory, reward = self._rollout_policy(behavior_policy, self._env, random=self._args.random, render=self._args.render_exploration, test=self._args.render_exploration)
                    exploration_rewards[j,i] = reward
                    #if self._args.render_exploration:
                    print_(f'Reward: {reward}', self._silent)
                    inner_buffer.add_trajectory(trajectory, force=True)
                    if not self._args.load_outer_buffer:
                        outer_buffer.add_trajectory(trajectory, force=True)

            if self._args.debug:
                print_(f'Mean exploration rewards: {exploration_rewards.mean(0)}', self._silent)
                print_(f'Positive exploration rewards: {(exploration_rewards>0).mean(0)}', self._silent)

        rewards = []
        reward_count = 0
        for t in range(self._training_iterations):
            rollouts, test_rewards, train_rewards, value, policy, vfs = self.train_step(t, summary_writer)

            if not self._silent:
                if len(test_rewards):
                    print_(f'{t}: {test_rewards}, {np.mean(value)}, {np.mean(policy)}, {time.time() - self._start_time}', self._silent)
                    if self._args.eval:
                        if reward_count == 0:
                            rewards = test_rewards
                        else:
                            factor = 1 / (reward_count + 1)
                            rewards = [r + (r_ - r) * factor for r, r_ in zip(rewards, test_rewards)]
                        reward_count += 1
                        print_(f'Rewards: {rewards}, {np.mean(rewards)}', self._silent)
                        if self._args.debug:
                            for idx, vf in enumerate(vfs):
                                print_(idx, argmax(vf, torch.zeros(self._observation_dim, device=self._device)), self._silent)

            if len(test_rewards):
                summary_writer.add_scalar(f'Reward_Test/Mean', np.mean(test_rewards), t)
            if len(train_rewards):
                summary_writer.add_scalar(f'Reward_Train/Mean', np.mean(train_rewards), t)

                if self._args.target_reward is not None:
                    if np.mean(train_rewards) > self._args.target_reward:
                        print_('Target reward reached; breaking', self._silent)
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
                if self._args.train_exploration:
                    torch.save(self._exploration_policy.state_dict(), f'{log_path}/ep_LATEST.pt')
                    torch.save(self._exploration_policy.state_dict(), f'{log_path}/ep_LATEST_.pt')
                if self._args.save_buffers:
                    for i, (inner_buffer, outer_buffer) in enumerate(zip(self._inner_buffers, self._outer_buffers)):
                        inner_buffer.save(f'{log_path}/inner_buffer_{i}')
                        outer_buffer.save(f'{log_path}/outer_buffer_{i}')
                    
