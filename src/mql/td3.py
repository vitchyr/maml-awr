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
                 discount=0.9,
                 tau=0.005,
                 policy_noise=0.,
                 noise_clip=0.5,
                 policy_freq=1,
                 silent=False
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


    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        return self.actor(state).cpu().data.numpy().flatten()

    def _rollout_policy(self, env, context):
        env.seed(self._env_seeds[self._rollout_counter].item())
        self._rollout_counter += 1
        trajectory = []
        state = env.reset()
        done = False
        total_reward = 0
        episode_t = 0

        success = False
        self.actor.eval()
        while not done:
            with torch.no_grad():
                pyt_state = torch.tensor(state, device=self._args.device).unsqueeze(0).float()
                action = self.actor(pyt_state, context).squeeze().detach().cpu().numpy()

            next_state, reward, done, info_dict = env.step(action)
            
            if 'success' in info_dict and info_dict['success']:
                success = True

            trajectory.append((state, action, next_state, reward, done))
            state = next_state
            total_reward += reward
            episode_t += 1
            if episode_t >= env._max_episode_steps or done:
                break
        self.actor.train()
        return trajectory, total_reward, success

    def eval(self, writer):
        results = []
        for i, (test_task_idx, test_buffer) in enumerate(zip(self.task_config.test_tasks, self._test_buffers)):
            batch = test_buffer.sample(self._args.batch_size, return_dict=True)
            self._env.set_task_idx(test_task_idx)

            state, action, next_state, reward = (batch['obs'],
                                                 batch['actions'],
                                                 batch['next_obs'],
                                                 batch['rewards'])
            
            context_input = torch.tensor(np.concatenate((state, action, next_state, reward), -1)).to(self._args.device)
            context_seq, h_n = self.context_encoder(context_input.unsqueeze(1), self.c0)
            context = h_n[-1]

            traj, reward, success = self._rollout_policy(self._env, context)
            results.append({'reward': reward, 'success': success, 'task': test_task_idx})
            writer.add_scalar(f'Eval_Reward/Task_{test_task_idx}', reward, self.total_it)
            writer.add_scalar(f'Eval_Success/Task_{test_task_idx}', success, self.total_it)

        return results
    
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
                        # Select action according to policy and add clipped noise
                        noise = (torch.randn_like(action) * self.policy_noise).clamp(
                            -self.noise_clip, self.noise_clip)

                        next_action = (self.actor(next_state, target_context) + noise).clamp(
                            -self.max_action, self.max_action)

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
                print('Avg Train Reward: {train_reward}')

            if self.total_it % self._args.vis_interval == 0:
                results = self.eval(summary_writer)
                test_reward = sum([r['reward'] for r in test_results]) / len(test_results)
                for r in results:
                    print(f"Test Task {r['task']} Reward: {r['reward']}")
                print('Avg Test Reward: {test_reward}')

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
