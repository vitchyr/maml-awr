import torch
import torch.nn as nn
import torch.optim.SGD as SGD
import torch.autograd as A
from torch.distributions import Normal
from envs import Env
from typing import List
from nn import MLP
from utils import ReplayBuffer, Experience
from copy import deepcopy
import higher


def copy_model_with_grads(from_model: nn.Module, to_model: nn.Module = None) -> nn.Module:
    if to_model is None:
        to_model = deepcopy(from_model)

    for from_param, to_param in zip(from_model.parameters(), to_model.parameters()):
        to_param.grad = from_param.grad

    return to_model


class MAMLRAWR(object):
    def __init__(self, envs: List[Env], policy_hidden_layers: List[int], value_function_hidden_layers: List[int],
                 training_iterations: int = 100000, rollouts_per_iteration: int = 4, batch_size: int = 128,
                 alpha1: float = 1e-4, alpha2: float = 1e-4, eta1: float = 1e-4, eta2: float = 1e-4, mu: float = 1e-4,
                 adaptation_temperature: float = 1.0, exploration_temperature: float = 1.0,
                 initial_trajectories: int = 100, device: torch.device = torch.device('cuda:0'),
                 maml_steps: int = 1):
        self._envs = envs

        example_env = self._envs[0]
        self._observation_dim = example_env.observation_space.shape[0]
        self._action_dim = example_env.action_space.shape[0]

        self._adaptation_policy = MLP([self._observation_dim] +
                                      policy_hidden_layers +
                                      [self._action_dim * 2]).to(device)
        self._exploration_policy = MLP([self._observation_dim] +
                                       policy_hidden_layers +
                                       [self._action_dim * 2]).to(device)
        self._value_function = MLP([self._observation_dim] + value_function_hidden_layers + [1]).to(device)

        self._buffers = [ReplayBuffer(env._max_episode_steps, env.observation_space.shape[0], env.action_space.shape[0])
                         for env in self._envs]

        self._training_iterations, self._rollouts_per_iteration = training_iterations, rollouts_per_iteration
        self._batch_size = batch_size
        self._alpha1, self._alpha2, self._eta1, self._eta2, self.mu = alpha1, alpha2, eta1, eta2, mu
        self._adaptation_temperature, self._exploration_temperature = adaptation_temperature, exploration_temperature
        self._device = device
        self._initial_trajectories = initial_trajectories
        self._maml_steps = maml_steps

    def _rollout_policy(self, policy: MLP, env: Env) -> List[Experience]:
        trajectory = []

        return trajectory

    def train_step(self,):
        b_ij = []
        psi_ij = []
        theta_ij = []
        inner_samples = []
        meta_grads_ij = []
        for i, (env, buffer) in enumerate(zip(self._envs, self._buffer)):
            buffer.add_trajectory(self._rollout_policy(self._exploration_policy, env))
            b_j = buffer.sample(self._batch_size * self._maml_steps).reshape(
                (self._rollouts_per_iteration, self._maml_steps, self._batch_size // self._rollouts_per_iteration, -1))
            b_ij.append(b_j)
            pyt_batch = torch.tensor(b_j).to(self._device)
            meta_batch = torch.tensor(buffer.sample(self._batch_size)).to(self._device)

            value_functions_j = []
            meta_value_grads_j = []
            for j, batch in enumerate(b_j):
                ##################################################################################################
                # Adapt value function
                ##################################################################################################
                value_function_j = deepcopy(self._value_function)
                opt = SGD(value_function_j, lr=self._eta1)
                with higher.innerloop_ctx(value_function_j, opt) as (f_value_function_j, diff_value_opt):
                    for batch in pyt_batch[j]:
                        value_estimates = f_value_function_j(batch[:self._observation_dim])
                        mc_value_estimates = batch[:,-1:]

                        loss = (value_estimates - mc_value_estimates).pow(2).mean()
                        diff_value_opt.step(loss)

                    meta_value_estimates = f_value_function_j(meta_batch[:,:self._observation_dim])
                    meta_mc_value_estimates = meta_batch[:,-1:]
                    meta_value_function_loss = (meta_value_estimates - meta_mc_value_estimates).pow(2).mean()
                    meta_grad = A.grad(meta_value_function_loss, f_value_function_j.parameters(time=0))
                    meta_value_grads_j.append(meta_grad)
                    copy_model_with_grads(f_value_function_j, value_function_j)
                    value_functions_j.append(value_function_j)
                ##################################################################################################

                ##################################################################################################
                # Adapt policy
                ##################################################################################################
                adaptation_policy_j = deepcopy(self._adaptation_policy)
                opt = SGD(adaptation_policy_j.parameters(), lr=self._alpha1)
                with higher.innerloop_ctx(adaptation_policy_j, opt) as (f_adaptation_policy_j, diff_policy_opt):
                    for batch in pyt_batch[j]:
                        value_estimates = f_value_function_j(batch[:self._observation_dim])
                        mc_value_estimates = batch[:,-1:]
                        advantage_weightings = (
                                    (1 / self._adaptation_temperature) * (mc_value_estimates - value_estimates)).exp()
                        action_parameters = f_adaptation_policy_j(batch[:self._observation_dim])
                        action_mu = action_parameters[:, :self._observation_dim]
                        action_sigma = action_parameters[:,self._observation_dim:].exp()
                        action_distribution = Normal(action_mu, action_sigma)
                        action_log_probs = action_distribution.log_prob(batch[:self._observation_dim])

                        loss = -(action_log_probs * advantage_weightings).mean()
                        diff_policy_opt.step(loss)

                    meta_value_estimates = value_function_j(meta_batch[:,:self._observation_dim])
                    meta_mc_value_estimates = meta_batch[:,-1:]
                    meta_advantages = 
                ##################################################################################################

            psi_ij.append(psi_j)




    def train(self):
        # Gather initial trajectory rollouts
        for i, (env, buffer) in enumerate(zip(self._envs, self._buffer)):
            buffer.add_trajectories(
                [self._rollout_policy(self._exploration_policy, env) for _ in range(self._initial_trajectories)])

        for t in range(self._training_iterations):
            self.train_step()
