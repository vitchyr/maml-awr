from typing import Optional, List
import argparse
import gym
import pickle
import numpy as np
from multiprocessing import Process
import random
import torch
import metaworld
from collections import namedtuple
import json

from src.envs import HalfCheetahDirEnv, HalfCheetahVelEnv, AntDirEnv, AntGoalEnv, HumanoidDirEnv, WalkerRandParamsWrappedEnv
from src.maml_rawr import MAMLRAWR
from src.args import get_args


def get_metaworld_tasks(env_id: str = 'ml10'):
    def _extract_tasks(env_, skip_task_idxs: Optional[List[int]] = []):
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
            tasks = _extract_tasks(env, skip_task_idxs=[0])

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


def get_gym_env(env: str):
    if env == 'ant':
        env = gym.make('Ant-v2')
    elif env == 'walker':
        env = gym.make('Walker2d-v2')
    elif env == 'humanoid':
        env = gym.make('Humanoid-v2')
    else:
        raise NotImplementedError(f'Unknown env: {env}')
        
    env.tasks = [{}]

    env.task_description_dim = lambda: 1
    def set_task_idx(idx):
        pass
    env.set_task_idx = set_task_idx

    def task_description(batch: None, one_hot: bool = True):
        one_hot = np.zeros((1,))
        if batch:
            one_hot = one_hot[None,:].repeat(batch, 0)
        return one_hot
    env.task_description = task_description

    return env


def run(args: argparse.Namespace, instance_idx: int = 0):
    with open(args.task_config, 'r') as f:
        task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))
    with open(task_config.tasks, 'rb') as f:
        tasks = pickle.load(f)

    if args.task_idx is not None:
        tasks = [tasks[args.task_idx]]

    seed = args.seed if args.seed is not None else instance_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    if task_config.env == 'ant_dir':
        env = AntDirEnv(tasks, args.n_tasks, include_goal = args.include_goal or args.multitask)
    elif task_config.env == 'cheetah_dir':
        env = HalfCheetahDirEnv(tasks, include_goal = args.include_goal or args.multitask)
    elif task_config.env == 'cheetah_vel':
        env = HalfCheetahVelEnv(tasks, args.n_tasks, include_goal = args.include_goal or args.multitask, one_hot_goal=args.one_hot_goal or args.multitask)
    elif task_config.env == 'humanoid_dir':
        env = HumanoidDirEnv(tasks, args.n_tasks, include_goal = args.include_goal)
    elif task_config.env == 'walker_params':
        env = WalkerRandParamsWrappedEnv(tasks, args.n_tasks, include_goal = args.include_goal or args.multitask)
    elif task_config.env == 'ml10':
        env = get_metaworld_tasks(task_config.env)
    else:
        raise RuntimeError(f'Invalid env name {task_config.env}')

    #with open(f'tasks/{task_config.env}_tasks', 'wb') as tasks_list:
    #    pickle.dump(env.tasks, tasks_list)
    
    if args.episode_length is not None:
        env._max_episode_steps = args.episode_length

    if args.name is None:
        args.name = 'throwaway_test_run'
    if instance_idx == 0:
        name = args.name
    else:
        name = f'{args.name}_{instance_idx}'

    network_shape = [256, 256]

    seed = args.seed if args.seed is not None else instance_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    maml_rawr = MAMLRAWR(args, task_config, env, args.log_dir, name, network_shape, network_shape, training_iterations=args.train_steps,
                         visualization_interval=args.vis_interval, silent=instance_idx > 0,
                         gradient_steps_per_iteration=args.gradient_steps_per_iteration,
                         replay_buffer_length=args.replay_buffer_size, discount_factor=args.discount_factor,
                         grad_clip=args.grad_clip)

    maml_rawr.train()


if __name__ == '__main__':
    args = get_args()
    
    if args.instances == 1:
        if args.profile:
            import cProfile
            cProfile.runctx('run(args)', sort='cumtime', locals=locals(), globals=globals())
        else:
            run(args)
    else:
        for instance_idx in range(args.instances):
            subprocess = Process(target=run, args=(args, instance_idx))
            subprocess.start()
