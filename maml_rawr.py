from copy import deepcopy
from typing import List

import higher
import numpy as np
import torch
import torch.autograd as A
import torch.nn as nn
import torch.optim as O
from torch.distributions import Normal

from envs import Env
from nn import MLP
from utils import ReplayBuffer, Experience


def copy_model_with_grads(from_model: nn.Module, to_model: nn.Module = None) -> nn.Module:
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
    def __init__(self, envs: List[Env], policy_hidden_layers: List[int], value_function_hidden_layers: List[int],
                 training_iterations: int = 100000, rollouts_per_iteration: int = 4, batch_size: int = 512,
                 alpha1: float = 1e-5, alpha2: float = 1e-5, eta1: float = 1e-5, eta2: float = 1e-5, mu: float = 1e-5,
                 adaptation_temperature: float = 1.0, exploration_temperature: float = 1.0,
                 initial_trajectories: int = 100, device: str = 'cuda:0',
                 maml_steps: int = 1, test_samples: int = 10, advantage_clamp: float = 10.0,
                 action_sigma: float = 0.2):
        self._envs = envs

        example_env = self._envs[0]

        self._observation_dim = example_env.observation_space.shape[0]
        self._action_dim = env_action_dim(example_env)

        self._adaptation_policy = MLP([self._observation_dim] +
                                      policy_hidden_layers +
                                      [self._action_dim]).to(device)
        self._exploration_policy = MLP([self._observation_dim] +
                                       policy_hidden_layers +
                                       [self._action_dim]).to(device)
        self._value_function = MLP([self._observation_dim] + value_function_hidden_layers + [1]).to(device)
        self._adaptation_policy_optimizer = O.Adam(self._adaptation_policy.parameters(), lr=alpha2)
        self._value_function_optimizer = O.Adam(self._value_function.parameters(), lr=eta2)
        self._exploration_policy_optimizer = O.Adam(self._adaptation_policy.parameters(), lr=mu)

        self._buffers = [ReplayBuffer(env._max_episode_steps, env.observation_space.shape[0], env_action_dim(env))
                         for env in self._envs]

        self._training_iterations, self._rollouts_per_iteration = training_iterations, rollouts_per_iteration
        self._batch_size = batch_size
        self._alpha1, self._alpha2, self._eta1, self._eta2, self._mu = alpha1, alpha2, eta1, eta2, mu
        self._adaptation_temperature, self._exploration_temperature = adaptation_temperature, exploration_temperature
        self._device = torch.device(device)
        self._initial_trajectories = initial_trajectories
        self._maml_steps = maml_steps
        self._cpu = torch.device('cpu')
        self._test_samples = test_samples
        self._advantage_clamp = advantage_clamp
        self._action_sigma = action_sigma
        
    def _rollout_policy(self, policy: MLP, env: Env) -> List[Experience]:
        trajectory = []

        state = env.reset()
        done = False
        total_reward = 0
        while not done:
            with torch.no_grad():
                mu = policy(torch.tensor(state).to(self._device).unsqueeze(0).float()).squeeze().to(self._cpu).numpy()
                sigma = self._action_sigma
                action = np.random.normal(0, 1, mu.shape) * sigma + mu

            next_state, reward, done, info_dict = env.step(action)
            trajectory.append(Experience(state, action, next_state, reward))
            state = next_state
            total_reward += reward

        return trajectory, total_reward

    # This function is the body of the main training loop [L4]
    # At every iteration, it adds rollouts from the exploration policy and one of the adapted policies
    #  to the replay buffer. It also updates the adaptation value function, adaptation policy, and
    #  exploration policy
    def train_step(self):
        batches = []
        value_functions = []
        adaptation_policies = []
        meta_value_grads = []
        meta_policy_grads = []
        rewards = []
        for i, (env, buffer) in enumerate(zip(self._envs, self._buffers)):
            env.reset()

            # Sample an exploration trajectory and add to buffer i [L6]
            buffer.add_trajectory(self._rollout_policy(self._exploration_policy, env)[0])

            # Sample J training batches for independent adaptations [L7]
            batches_i = buffer.sample(self._batch_size * self._maml_steps).reshape(
                (self._rollouts_per_iteration, self._maml_steps, self._batch_size // self._rollouts_per_iteration, -1))
            pyt_batch = torch.tensor(batches_i).to(self._device)
            batches.append(pyt_batch)
            meta_batch = torch.tensor(buffer.sample(self._batch_size)).to(self._device)

            value_functions_i = []
            adaptation_policies_i = []
            meta_value_grads_i = []
            meta_policy_grads_i = []
            for j, batch in enumerate(batches_i):
                ##################################################################################################
                # Adapt value function and collect meta-gradients
                ##################################################################################################
                value_function_j = deepcopy(self._value_function)
                opt = O.SGD(value_function_j.parameters(), lr=self._eta1)
                with higher.innerloop_ctx(value_function_j, opt) as (f_value_function_j, diff_value_opt):
                    for batch in pyt_batch[j]:
                        value_estimates = f_value_function_j(batch[:,:self._observation_dim])
                        mc_value_estimates = batch[:,-1:]

                        # Compute loss and adapt value function [L9]
                        loss = (value_estimates - mc_value_estimates).pow(2).mean()
                        diff_value_opt.step(loss)

                    meta_value_estimates = f_value_function_j(meta_batch[:,:self._observation_dim])
                    meta_mc_value_estimates = meta_batch[:,-1:]
                    meta_value_function_loss = (meta_value_estimates - meta_mc_value_estimates).pow(2).mean()
                    meta_value_grad_j = A.grad(meta_value_function_loss, f_value_function_j.parameters(time=0))

                    # Collect grads for the value function update in the outer loop [L14],
                    #  which is not actually performed here
                    meta_value_grads_i.append(meta_value_grad_j)
                    copy_model_with_grads(f_value_function_j, value_function_j)
                    value_functions_i.append(value_function_j)
                ##################################################################################################

                ##################################################################################################
                # Adapt policy and collect meta-gradients [
                ##################################################################################################
                adaptation_policy_j = deepcopy(self._adaptation_policy)
                opt = O.SGD(adaptation_policy_j.parameters(), lr=self._alpha1)
                with higher.innerloop_ctx(adaptation_policy_j, opt) as (f_adaptation_policy_j, diff_policy_opt):
                    for batch in pyt_batch[j]:
                        value_estimates = f_value_function_j(batch[:,:self._observation_dim])
                        mc_value_estimates = batch[:,-1:]
                        advantages = ((1 / self._adaptation_temperature) * (mc_value_estimates - value_estimates))
                        weights = advantages.clamp(max=self._advantage_clamp).exp()
                        action_mu = f_adaptation_policy_j(batch[:,:self._observation_dim])
                        action_sigma = torch.empty_like(action_mu).fill_(self._action_sigma)
                        action_distribution = Normal(action_mu, action_sigma)
                        action_log_probs = action_distribution.log_prob(batch[:,self._observation_dim:self._observation_dim + self._action_dim])

                        # Compute loss and adapt policy [L10]
                        loss = -(action_log_probs * weights.detach()).mean()
                        diff_policy_opt.step(loss)

                    meta_value_estimates = value_function_j(meta_batch[:,:self._observation_dim])
                    meta_mc_value_estimates = meta_batch[:,-1:]
                    meta_advantages = ((1 / self._adaptation_temperature) * (
                                meta_mc_value_estimates - meta_value_estimates))
                    meta_weights = meta_advantages.clamp(max=self._advantage_clamp).exp()
                    meta_action_mu = f_adaptation_policy_j(meta_batch[:,:self._observation_dim])
                    meta_action_sigma = torch.empty_like(meta_action_mu).fill_(self._action_sigma)
                    meta_action_distribution = Normal(meta_action_mu, meta_action_sigma)
                    meta_action_log_probs = meta_action_distribution.log_prob(meta_batch[:, self._observation_dim:self._observation_dim + self._action_dim])
                    meta_policy_loss = -(meta_action_log_probs * meta_weights.detach()).mean()
                    meta_policy_grad_j = A.grad(meta_policy_loss, f_adaptation_policy_j.parameters(time=0))

                    # Collect grads for the adaptation policy update in the outer loop [L15],
                    #  which is not actually performed here
                    meta_policy_grads_i.append(meta_policy_grad_j)
                    copy_model_with_grads(f_adaptation_policy_j, adaptation_policy_j)
                    adaptation_policies_i.append(adaptation_policy_j)
                ##################################################################################################

            value_functions.append(value_functions_i)
            meta_value_grads.append(meta_value_grads_i)
            adaptation_policies.append(adaptation_policies_i)
            meta_policy_grads.append(meta_policy_grads_i)

            # Sample adapted policy trajectory, add to replay buffer i [L12]
            trajectory, reward = self._rollout_policy(adaptation_policies_i[0], env)
            buffer.add_trajectory(trajectory)
            rewards.append(reward)

        ############################################################################3
        # Meta-update value function [L14]
        ############################################################################3
        for idx, parameter in enumerate(self._value_function.parameters()):
            grads = []
            for i in range(len(meta_value_grads)):
                for j in range(len(meta_value_grads[i])):
                    grads.append(meta_value_grads[i][j][idx])
            parameter.grad = sum(grads) / len(grads)
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
        self._adaptation_policy_optimizer.step()
        self._adaptation_policy_optimizer.zero_grad()
        ############################################################################3
        
        ############################################################################3
        # Update exploration policy [L16]
        ############################################################################3
        for i, batches_i in enumerate(batches):
            for j, batch_ij in enumerate(batches_i):
                batch = batch_ij.view(self._maml_steps * self._batch_size // self._rollouts_per_iteration, -1)
                value_estimates = value_functions[i][j](batch[:,:self._observation_dim])
                mc_value_estimates = batch[:,-1:]
                advantages = ((1 / self._exploration_temperature) * (mc_value_estimates - value_estimates))
                weights = advantages.clamp(max=self._advantage_clamp).exp().detach()
                action_mu = self._exploration_policy(batch[:,:self._observation_dim])
                action_sigma = torch.empty_like(action_mu).fill_(self._action_sigma)
                action_distribution = Normal(action_mu, action_sigma)
                action_log_probs = action_distribution.log_prob(batch[:,self._observation_dim:self._observation_dim + self._action_dim])
                exploration_policy_loss = -(action_log_probs * weights.detach()).mean()
                exploration_policy_loss.backward()
        self._exploration_policy_optimizer.step()
        self._exploration_policy_optimizer.zero_grad()
        ############################################################################3

        return rewards

    def train(self):
        # Gather initial trajectory rollouts
        for i, (env, buffer) in enumerate(zip(self._envs, self._buffers)):
            buffer.add_trajectories(
                [self._rollout_policy(self._exploration_policy, env)[0] for _ in range(self._initial_trajectories)])

        for t in range(self._training_iterations):
            # with A.detect_anomaly():
            rewards = self.train_step()
            print(rewards)
            # rewards = np.empty((len(self._envs), self._test_samples))
            # for env_idx, env in enumerate(self._envs):
            #     for idx in range(self._test_samples):
            #         rewards[env_idx, idx] = self._rollout_policy()
