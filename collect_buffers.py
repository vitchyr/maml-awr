import gym
from gym.wrappers import TimeLimit

import numpy as np
import os
import pickle
import random
import tensorflow as tf

from src.data_args import get_args
from src.envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv, AntGoalEnv, HumanoidDirEnv, WalkerRandParamsWrappedEnv

def get_metaworld_tasks(env_id: str = 'ml10'):
    def _extract_tasks(env_, skip_task_idxs = []):
        task_idxs = set()
        tasks = [None for _ in range(env.num_tasks - len(skip_task_idxs))]
        while len(task_idxs) < env.num_tasks - len(skip_task_idxs):
            task_dict = env.sample_tasks(1)[0]
            task_idx = task_dict['task']
            if task_idx not in task_idxs and task_idx not in skip_task_idxs:
                task_idxs.add(task_idx)
                tasks[task_idx - len(skip_task_idxs)] = task_dict
        return tasks

    if env_id == 'ml10':
        from metaworld.benchmarks import ML10
        if args.mltest:
            env = ML10.get_test_tasks()
            tasks = _extract_tasks(env)
        else:
            env = ML10.get_train_tasks()
            tasks = _extract_tasks(env, skip_task_idxs=[])

        if args.task_idx is not None:
            tasks = [tasks[args.task_idx]]

        env.tasks = tasks
        print(tasks)
        
        def set_task_idx(idx):
            env.set_task(tasks[idx])
        def task_description(batch: None, one_hot: bool = True):
            one_hot = env.active_task_one_hot.astype(np.float32)
            if batch:
                one_hot = one_hot[None,:].repeat(batch, 0)
            return one_hot

        env.set_task_idx = set_task_idx
        env.task_description = task_description
        env.task_description_dim = lambda: len(env.tasks)
        env._max_episode_steps = 150

        return env
    else:
        raise NotImplementedError()
    
def main(args):
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    if args.env == 'ant_dir':
        ant_dir_tasks = pickle.load(open(f"{args.task_path}/ant_dir_tasks", "rb"))
        env = AntDirEnv(tasks = ant_dir_tasks, include_goal = args.include_goal)
    elif args.env == 'ant_goal':
        env = AntGoalEnv(include_goal = args.include_goal)
    elif args.env == 'cheetah_dir':
        cheetah_dir_tasks = pickle.load(open(f"{args.task_path}/cheetah_dir_tasks", "rb"))
        env = HalfCheetahDirEnv(tasks = cheetah_dir_tasks, include_goal = args.include_goal)
    elif args.env == 'cheetah_vel':
        cheetah_vel_tasks = pickle.load(open(f"{args.task_path}/cheetah_vel_tasks", "rb"))
        env = HalfCheetahVelEnv(tasks = cheetah_vel_tasks, include_goal = args.include_goal)
    elif args.env == 'humanoid_dir':
        env = HumanoidDirEnv(include_goal = args.include_goal)
    elif args.env == 'walker_param':
        walker_tasks = pickle.load(open(f"{args.task_path}/walker_params_tasks", "rb"))
        env = WalkerRandParamsWrappedEnv(tasks = walker_tasks, include_goal = args.include_goal)
    elif args.env == 'ml45':
        from metaworld.benchmarks.base import Benchmark
        from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
        from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT

        args.type = 'train'
        if args.task is None:
            args.task = list(HARD_MODE_ARGS_KWARGS[args.type].keys())[args.task_idx]       
        args_kwargs =  HARD_MODE_ARGS_KWARGS[args.type][args.task]
        args_kwargs['kwargs']['obs_type'] = 'with_goal'
        args_kwargs['task'] = args.task
        env = HARD_MODE_CLS_DICT[args.type][args.task](*args_kwargs['args'], **args_kwargs['kwargs'])
        
    if args.env == 'ml45':
        env = TimeLimit(env, max_episode_steps = 150)
        pickle.dump(args_kwargs, open(args.log_dir + '/env_{}_{}_task{}.pkl'.format(args.env, args.type, args.task_idx), "wb" ))   
    else:
        env.observation_space = gym.spaces.box.Box(env.observation_space.low, env.observation_space.high)
        env.action_space = gym.spaces.box.Box(env.action_space.low, env.action_space.high)
        env = TimeLimit(env, max_episode_steps = 200)
        pickle.dump(env.unwrapped.tasks, open(args.log_dir + '/env_{}_task{}.pkl'.format(args.env, args.task_idx), "wb" ))

    if args.alg == 'td3':
        from stable_baselines.td3.policies import MlpPolicy
        from stable_baselines.ddpg.noise import NormalActionNoise
        from src.td3 import TD3
        
        n_actions = env.action_space.shape[-1]
        action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

        model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1,
                    tensorboard_log = args.log_dir + '/tensorboard/log_{}_task_{}'.format(args.env, args.task_idx),
                    buffer_log = args.log_dir + '/buffers_{}_{}_'.format(args.env, args.task_idx),
                    full_size = args.full_buffer_size,
                    buffer_size = args.replay_buffer_size, 
                    batch_size = args.batch_size, 
                    policy_kwargs={'layers': [400, 300]},
                    learning_rate=args.outer_policy_lr
        )
        print('###################################')
        print('###################################')
        print('## Running *TD3* data collection ##')
        print('###################################')
        print('###################################')
        model.learn(total_timesteps=args.full_buffer_size, log_interval=10)
    else:
        from stable_baselines.sac.policies import MlpPolicy
        from stable_baselines.sac.policies import FeedForwardPolicy
        from src.sac2 import SAC
        
        model = SAC(MlpPolicy,
                    env, 
                    verbose=1, 
                    tensorboard_log = args.log_dir + '/tensorboard/log_{}_task_{}'.format(args.env, args.task_idx),
                    buffer_log = args.log_dir + '/buffers_{}_{}_'.format(args.env, args.task_idx),
                    buffer_size = args.replay_buffer_size, 
                    full_size = args.full_buffer_size,
                    batch_size = args.batch_size, 
                    policy_kwargs={'layers': [300,300,300]},
                    learning_rate = 3e-4,
                    gamma = 0.99)
        print('###################################')
        print('###################################')
        print('## Running *SAC* data collection ##')
        print('###################################')
        print('###################################')
        model.learn(total_timesteps = args.full_buffer_size, log_interval = 1)

    model.save(args.log_dir + '/model_{}_{}'.format(args.env, args.task_idx))

if __name__ == '__main__':
    random.seed(17)
    np.random.seed(17)
    tf.set_random_seed(17)
    
    args = get_args()
    args.task_idx = int(args.task_idx)
    args.task = None
    
    main(args)
