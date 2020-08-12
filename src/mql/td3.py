import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from typing import List
from src.utils import NewReplayBuffer
import os
import json
import pickle
from torch.utils.tensorboard import SummaryWriter
import random
import cvxpy as cp


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action, context_dim):
        super(Actor, self).__init__()

        self.l1 = nn.Linear(state_dim + context_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, action_dim)

        self.max_action = max_action

    def forward(self, state, context):
        a = F.relu(self.l1(torch.cat([state, context], -1)))
        a = F.relu(self.l2(a))
        return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, context_dim):
        super(Critic, self).__init__()

        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim + context_dim, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim + context_dim, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

    def forward(self, state, action, context):
        sa = torch.cat([state, action, context], -1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state, action, context):
        sa = torch.cat([state, action, context], -1)

        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1


class TD3Context(object):
    def __init__(self,
                 args,
                 task_config: dict,
                 env,
                 log_dir: str,
                 name,
                 context_hidden,
                 training_iterations=1000000,
                 context_layers=2,
                 discount=0.99,
                 tau=0.005,
                 policy_noise=0.,
                 noise_clip=0.5,
                 policy_freq=1,
                 silent=False,
                 mql_steps1=5,
                 mql_steps2=100
    ):
        self._args = args
        self.task_config = task_config
        self._env = env
        self._log_dir = log_dir
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0]

        has_train_buffers = hasattr(task_config, 'train_buffer_paths') and not args.eval
        has_test_buffers = hasattr(task_config, 'test_buffer_paths')

        load_inner_buffers = has_train_buffers and args.load_inner_buffer
        load_outer_buffers = has_train_buffers and args.load_outer_buffer
        load_test_buffers = has_test_buffers and args.load_inner_buffer # we want the test adaptation data the same as train

        inner_buffers = [task_config.train_buffer_paths.format(idx) if load_inner_buffers else None for idx in task_config.train_tasks]
        outer_buffers = [task_config.train_buffer_paths.format(idx) if load_outer_buffers else None for idx in task_config.train_tasks]
        test_buffers = [task_config.test_buffer_paths.format(idx) if load_test_buffers else None for idx in task_config.test_tasks]

        self._test_buffers = [NewReplayBuffer(args.inner_buffer_size, self.state_dim, self.action_dim,
                                              discount_factor=discount,
                                              immutable=test_buffers[i] is not None, load_from=test_buffers[i], silent=silent, skip=args.inner_buffer_skip,
                                              stream_to_disk=args.from_disk, mode=args.buffer_mode)
                               for i, task in enumerate(task_config.test_tasks)]

        self._inner_buffers = [NewReplayBuffer(args.inner_buffer_size, self.state_dim, self.action_dim,
                                               discount_factor=discount,
                                               immutable=args.offline or args.offline_inner, load_from=inner_buffers[i], silent=silent, skip=args.inner_buffer_skip,
                                               stream_to_disk=args.from_disk, mode=args.buffer_mode)
                               for i, task in enumerate(task_config.train_tasks)]
        
        if args.offline and args.load_inner_buffer and args.load_outer_buffer and (args.replay_buffer_size == args.inner_buffer_size) and (args.buffer_skip == args.inner_buffer_skip) and args.buffer_mode == 'end':
            self._outer_buffers = self._inner_buffers
        else:
            self._outer_buffers = [NewReplayBuffer(args.replay_buffer_size, self.state_dim, self.action_dim,
                                                   discount_factor=discount, immutable=args.offline or args.offline_outer,
                                                   load_from=outer_buffers[i], silent=silent, skip=args.buffer_skip,
                                                   stream_to_disk=args.from_disk)
                                   for i, task in enumerate(task_config.train_tasks)]
        
        self.actor = Actor(self.state_dim, self.action_dim, env.action_space.high[0], context_hidden).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=3e-4)

        self.critic = Critic(self.state_dim, self.action_dim, context_hidden).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=3e-4)

        # GRU with input (s, a, s', r)
        self.context_encoder = nn.GRU(self.state_dim * 2 + self.action_dim + 1,
                                      context_hidden, num_layers=context_layers).to(device)
        self.c0 = nn.Parameter(torch.randn(context_layers, 1, context_hidden)).to(device).detach()
        self.context_optimizer = torch.optim.Adam(list(self.context_encoder.parameters()) + [self.c0],
                                                   lr=3e-4)

        self.max_action = env.action_space.high[0]
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0
        self.max_it = training_iterations
        self._env_seeds = np.random.randint(1e10, size=(int(1e7),))
        self._rollout_counter = 0
        self._name = name
        self._mql_steps1 = mql_steps1
        self._mql_steps2 = mql_steps2

    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def _rollout_policy(self, env, context, policy=None):
        if policy is None:
            policy = self.actor
            
        env.seed(self._env_seeds[self._rollout_counter].item())
        self._rollout_counter += 1
        trajectory = []
        state = env.reset()
        done = False
        total_reward = 0
        episode_t = 0

        success = False
        policy.eval()
        while not done:
            with torch.no_grad():
                pyt_state = torch.tensor(state, device=self._args.device).unsqueeze(0).float()
                action = policy(pyt_state, context).squeeze().detach().cpu().numpy()

            next_state, reward, done, info_dict = env.step(action)
            
            if 'success' in info_dict and info_dict['success']:
                success = True

            trajectory.append((state, action, next_state, reward, done))
            state = next_state
            total_reward += reward
            episode_t += 1
            if episode_t >= env._max_episode_steps or done:
                break
        policy.train()
        return trajectory, total_reward, success

    def get_context_from_batches(self, batches):
        all_other_context = []
        states = []
        actions = []
        next_states = []
        rewards = []
        for batch_ in batches:
            state_, action_, next_state_, reward_ = (batch_['obs'],
                                                     batch_['actions'],
                                                     batch_['next_obs'],
                                                     batch_['rewards'])
            state_ = torch.tensor(state_).to(self._args.device)
            action_ = torch.tensor(action_).to(self._args.device)
            next_state_ = torch.tensor(next_state_).to(self._args.device)
            reward_ = torch.tensor(reward_).to(self._args.device)
            states.append(state_)
            actions.append(action_)
            next_states.append(next_state_)
            rewards.append(reward_)
            
            context_input_ = torch.cat((state_, action_, next_state_, reward_), -1)
            other_context, _ = self.context_encoder(context_input_.unsqueeze(1), self.c0)
            other_context = other_context.squeeze(1)
            all_other_context.append(other_context)
            
        return torch.cat(all_other_context), (torch.cat(states), torch.cat(actions), torch.cat(next_states), torch.cat(rewards))
    
    def eval(self, writer):
        td3_results, mql_results = [], []
        for i, (test_task_idx, test_buffer) in enumerate(zip(self.task_config.test_tasks, self._test_buffers)):
            batch = test_buffer.sample(self._args.batch_size, return_dict=True)
            other_batches = [buf.sample(self._args.batch_size, return_dict=True) for buf in self._inner_buffers]
            
            self._env.set_task_idx(test_task_idx)

            state, action, next_state, reward = (torch.tensor(batch['obs']).to(self._args.device),
                                                 torch.tensor(batch['actions']).to(self._args.device),
                                                 torch.tensor(batch['next_obs']).to(self._args.device),
                                                 torch.tensor(batch['rewards']).to(self._args.device))
            
            with torch.no_grad():
                context_input = torch.cat((state, action, next_state, reward), -1)
                context_seq, h_n = self.context_encoder(context_input.unsqueeze(1), self.c0)
                context = h_n[-1]
                all_test_context = context_seq.squeeze(1)
                all_other_context, (state_, action_, next_state_, reward_) = self.get_context_from_batches(other_batches)
                all_other_context = all_other_context[torch.randperm(all_other_context.shape[0], device=self._args.device)[:self._args.batch_size]]

                labels = np.concatenate((-np.ones((self._args.batch_size)),
                                         np.ones((self._args.batch_size))))
                X = torch.cat((all_test_context, all_other_context)).cpu().numpy()
                w = cp.Variable((all_test_context.shape[-1]))

                obj = cp.Minimize(cp.sum(cp.logistic(cp.neg(cp.multiply(labels, X @ w)))))
                prob = cp.Problem(obj)
                sol = prob.solve()

                w_ = torch.tensor(w.value, device=self._args.device).float()
                test_betas = (-all_test_context @ w_).exp()
                test_ess = (1/test_betas.shape[0]) * test_betas.sum().pow(2) / test_betas.pow(2).sum()
                context_betas = (-all_other_context @ w_).exp()

            traj, reward, success = self._rollout_policy(self._env, context)
            td3_results.append({'reward': reward, 'success': success, 'task': test_task_idx})
            writer.add_scalar(f'TD3_Reward/Task_{test_task_idx}', reward, self.total_it)
            writer.add_scalar(f'TD3_Success/Task_{test_task_idx}', success, self.total_it)

            actor = copy.deepcopy(self.actor)
            critic_target = copy.deepcopy(self.critic_target)
            critic = copy.deepcopy(self.critic)
            actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
            critic_optimizer = torch.optim.Adam(critic.parameters(), lr=3e-4)

            start_actor_params = [p.clone() for p in actor.parameters()]
            start_critic_params = [p.clone() for p in critic.parameters()]

            lambda_ = 1 - test_ess
            context_ = context
            
            for step in range(self._mql_steps1 + self._mql_steps2):
                if step == self._mql_steps1:
                    traj, reward, success = self._rollout_policy(self._env, context, policy=actor)
                    writer.add_scalar(f'MQL1_Reward/Task_{test_task_idx}', reward, self.total_it)
                    writer.add_scalar(f'MQL1_Success/Task_{test_task_idx}', success, self.total_it)

                critic_param_loss = sum([(p - p_.detach()).pow(2).sum() for p, p_ in zip(critic.parameters(), start_critic_params)])
                actor_param_loss = sum([(p - p_.detach()).pow(2).sum() for p, p_ in zip(actor.parameters(), start_actor_params)])
                if step >= self._mql_steps1:
                    # re-assign state, action, next_state, reward to use data from train buffers
                    # use w_ to assign weights
                    # also add regularization to theta
                    other_batches = [buf.sample(self._args.batch_size, return_dict=True) for buf in self._inner_buffers]
                    all_other_context, (state, action, next_state, reward) = self.get_context_from_batches(other_batches)
                    context_betas = (-all_other_context @ w_).exp()
                    context_ess = (1/context_betas.shape[0]) * context_betas.sum().pow(2) / context_betas.pow(2).sum()
                    lambda_ = 1 - context_ess
                    if step == self._mql_steps1 + self._mql_steps2 - 1:
                        writer.add_scalar(f'Test_Beta/Task_{test_task_idx}', test_betas.mean(), self.total_it)
                        writer.add_scalar(f'Context_Beta/Task_{test_task_idx}', context_betas.mean(), self.total_it)
                        writer.add_scalar(f'Test_ESS/Task_{test_task_idx}', test_ess, self.total_it)
                        writer.add_scalar(f'Context_ESS/Task_{test_task_idx}', context_ess, self.total_it)
                else:
                    lambda_ = 1 - test_ess

                not_done = torch.ones((state.shape[0],1)).to(self._args.device)
                
                if context_.shape[0] != state.shape[0]:
                    context_ = context.repeat(state.shape[0], 1)
                    
                with torch.no_grad():
                    next_action = actor(next_state, context_)
                    # Compute the target Q value
                    target_Q1, target_Q2 = critic_target(next_state, next_action, context_)
                    target_Q = torch.min(target_Q1, target_Q2)
                    target_Q = reward + not_done * self.discount * target_Q

                # Get current Q estimates
                current_Q1, current_Q2 = critic(state, action, context_)

                # Compute critic loss
                critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                    current_Q2, target_Q) + (lambda_ / 2) * critic_param_loss

                # Optimize the critic
                critic_optimizer.zero_grad()
                critic_loss.backward(retain_graph=True)
                critic_optimizer.step()

                # Compute actor losse
                actor_loss = -critic.Q1(state, actor(state, context_), context_).mean() + (lambda_ / 2) * actor_param_loss

                # Optimize the actor
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

                for param, target_param in zip(critic.parameters(),
                                               critic_target.parameters()):
                    target_param.data.copy_(self.tau * param.data +
                                            (1 - self.tau) * target_param.data)

            traj, reward, success = self._rollout_policy(self._env, context, policy=actor)
            mql_results.append({'reward': reward, 'success': success, 'task': test_task_idx})
            writer.add_scalar(f'MQL_Reward/Task_{test_task_idx}', reward, self.total_it)
            writer.add_scalar(f'MQL_Success/Task_{test_task_idx}', success, self.total_it)

        return td3_results, mql_results
    
    def train(self):
        batch_size = self._args.batch_size
        log_path = f'{self._log_dir}/{self._name}'
        print('*******************************************************')
        print('*******************************************************')
        if os.path.exists(log_path):
            sep = '.'
            existing = os.listdir(f'{self._log_dir}')
            idx = 0
            for directory in existing:
                if directory.startswith(self._name):
                    idx += 1
            print(f'Experiment output {log_path} already exists.')
            log_path = f'{self._log_dir}/{self._name}{sep}{idx}'
            self._name = f'{self._name}{sep}{idx}'

        print(f'Saving outputs to {log_path}')
        print('*******************************************************')
        print('*******************************************************')
        os.makedirs(log_path)

        with open(f'{log_path}/args.txt', 'w') as args_file:
            json.dump(self._args.__dict__, args_file, indent=4, sort_keys=True)
        with open(f'{log_path}/tasks.pkl', 'wb') as tasks_file:
            pickle.dump(self._env.tasks, tasks_file)
        tensorboard_log_path = f'{log_path}/tb'
        if not os.path.exists(tensorboard_log_path):
            os.makedirs(tensorboard_log_path)
        summary_writer = SummaryWriter(tensorboard_log_path)

        while self.total_it < self.max_it:
            if self.total_it % self._args.gradient_steps_per_iteration == 0:
                print(f'Train step {self.total_it}')
            if self._args.task_batch_size is not None and len(self.task_config.train_tasks) > self._args.task_batch_size:
                tasks = random.sample(self.task_config.train_tasks, self._args.task_batch_size)
            else:
                tasks = self.task_config.train_tasks
            
            self.total_it += 1
            train_results = []
            for i, (train_task_idx, inner_buffer, outer_buffer) in enumerate(zip(self.task_config.train_tasks, self._inner_buffers, self._outer_buffers)):
                # Only train on the randomly selected tasks for this iteration
                if train_task_idx not in tasks:
                    continue
                
                self._env.set_task_idx(train_task_idx)

                # Sample replay buffer
                batch = inner_buffer.sample(batch_size, return_dict=True)
                state, action, next_state, reward = (batch['obs'],
                                                     batch['actions'],
                                                     batch['next_obs'],
                                                     batch['rewards'])
                not_done = torch.ones((batch_size,1), device=self.c0.device)

                context_input = torch.tensor(np.concatenate((state, action, next_state, reward), -1)).to(self._args.device)
                context_seq, h_n = self.context_encoder(context_input.unsqueeze(1), self.c0)
                context = h_n[-1]
                train_context = target_context = context.repeat(batch_size, 1)
                
                batch = outer_buffer.sample(batch_size, return_dict=True)
                state, action, next_state, reward = (batch['obs'],
                                                     batch['actions'],
                                                     batch['next_obs'],
                                                     batch['rewards'])
                state = torch.tensor(state).to(self._args.device)
                action = torch.tensor(action).to(self._args.device)
                next_state = torch.tensor(next_state).to(self._args.device)
                reward = torch.tensor(reward).to(self._args.device)

                if self.total_it % 2 == 0:
                    with torch.no_grad():
                        next_action = self.actor(next_state, target_context)
                        # Compute the target Q value
                        target_Q1, target_Q2 = self.critic_target(next_state, next_action, target_context)
                        target_Q = torch.min(target_Q1, target_Q2)
                        target_Q = reward + not_done * self.discount * target_Q

                    # Get current Q estimates
                    current_Q1, current_Q2 = self.critic(state, action, train_context)
                    summary_writer.add_scalar(f'Q1/Task_{train_task_idx}', current_Q1.mean(), self.total_it)
                    summary_writer.add_scalar(f'Q2/Task_{train_task_idx}', current_Q2.mean(), self.total_it)

                    # Compute critic loss
                    critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(
                        current_Q2, target_Q)
                    summary_writer.add_scalar(f'Critic_Loss/Task_{train_task_idx}', critic_loss.item(), self.total_it)

                    # Optimize the critic
                    critic_loss.backward(retain_graph=True)
                else:
                    # Compute actor losse
                    actor_loss = -self.critic.Q1(state, self.actor(state, train_context), train_context).mean()
                    summary_writer.add_scalar(f'Actor_Loss/Task_{train_task_idx}', actor_loss.item(), self.total_it)
                    
                    # Optimize the actor
                    actor_loss.backward()

                if self.total_it % self._args.gradient_steps_per_iteration == 0:
                    trajectory, reward, success = self._rollout_policy(self._env, context)
                    train_results.append({'reward': reward, 'success': success, 'task': train_task_idx})
                    summary_writer.add_scalar(f'Train_Reward/Task_{train_task_idx}', reward, self.total_it)
                    print(f'Task {train_task_idx} Reward: {reward}')

            if len(train_results):
                train_reward = sum([r['reward'] for r in train_results]) / len(train_results)
                summary_writer.add_scalar(f'Train_Reward/Average', train_reward, self.total_it)
                print(f'Avg Train Reward: {train_reward}')

            if self.total_it % self._args.vis_interval == 0:
                td3_results, mql_results = self.eval(summary_writer)
                td3_reward = sum([r['reward'] for r in td3_results]) / len(td3_results)
                mql_reward = sum([r['reward'] for r in mql_results]) / len(mql_results)

                summary_writer.add_scalar(f'TD3_Reward/Average', td3_reward, self.total_it)
                summary_writer.add_scalar(f'MQL_Reward/Average', mql_reward, self.total_it)

                for td3, mql in zip(td3_results, mql_results):
                    print(f"Test Task {td3['task']} TD3: {td3['reward']} MQL: {mql['reward']}")
                print(f'Avg TD3 Reward: {td3_reward}')
                print(f'Avg MQL Reward: {mql_reward}')

            self.context_optimizer.step()
            self.context_optimizer.zero_grad()

            if self.total_it % 2 == 0:
                self.critic_optimizer.step()
            else:
                self.actor_optimizer.step()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            # Update the frozen target models
            for param, target_param in zip(self.critic.parameters(),
                                           self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data +
                                        (1 - self.tau) * target_param.data)
                
            for param, target_param in zip(self.actor.parameters(),
                                           self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data +
                                        (1 - self.tau) * target_param.data)
                

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(),
                   filename + "_critic_optimizer")

        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(),
                   filename + "_actor_optimizer")
        torch.save(self.context_encoder.state_dictt(), filename + "_encoder")
        torch.save(self.c0.state_dictt(), filename + "_c0")
        torch.save(self.context_optimizer.state_dict(), filename + "_encoder_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(
            torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(
            torch.load(filename + "_actor_optimizer"))
        self.actor_target = copy.deepcopy(self.actor)
