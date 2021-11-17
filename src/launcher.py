import argparse
import joblib
import json
import os.path as osp
import pickle
import random
import uuid
from enum import Enum
from typing import Optional, List

import gym
import numpy as np
import torch
from torch.multiprocessing import Process

from rlkit.data_management.offline_dataset.util import rlkit_buffer_to_macaw_format
from doodad.wrappers.easy_launch import save_doodad_config, DoodadConfig
from src import pythonplusplus as ppp
from src.args import get_default_args
from src.envs import (
    AntDirEnv,
    HalfCheetahVelEnv,
    WalkerRandParamsWrappedEnv,
    HopperRandParamsWrappedEnv,
    HumanoidDirEnv,
)
from src.logging import setup_logger, logger

args = None


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


def get_ml45():
    from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT

    args.type = 'train'
    if args.task is None:
        args.task = list(HARD_MODE_ARGS_KWARGS[args.type].keys())[args.task_idx]
    args_kwargs =  HARD_MODE_ARGS_KWARGS[args.type][args.task]
    args_kwargs['kwargs']['obs_type'] = 'with_goal'
    args_kwargs['task'] = args.task
    env = HARD_MODE_CLS_DICT[args.type][args.task](*args_kwargs['args'], **args_kwargs['kwargs'])

    return


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

    # env = TimeLimit(env, max_episode_steps=200)

    return env


def run(
        log_dir: str,
        env: str,
        pretrain_buffer_path: str,
        instance_idx: int = 0,
        use_rlkit: bool = True,
        # buffer_path_template: str = '',
        saved_tasks_path: str = '',
        # train_task_idxs: List[int] = (),
        # test_task_idxs: List[int] = (),
        seed=0,
        path_length=200,
        load_buffer_kwargs=None,
        **kwargs
):
    global args
    extra_args = []
    for k, v in kwargs.items():
        extra_args.append('--{}'.format(k))
        extra_args.append(str(v))
    args = get_default_args(extra_args)
    # with open(args.task_config, 'r') as f:
    #     task_config = json.load(f, object_hook=lambda d: namedtuple('X', d.keys())(*d.values()))

    if load_buffer_kwargs is None:
        load_buffer_kwargs = {}
    if args.advantage_head_coef == 0:
        args.advantage_head_coef = None

    # if task_config.env != 'ml45':
    #     tasks = []
    #     for task_idx in (range(task_config.total_tasks if args.task_idx is None else [args.task_idx])):
    #         with open(task_config.task_paths.format(task_idx), 'rb') as f:
    #             task_info = pickle.load(f)
    #             assert len(task_info) == 1, f'Unexpected task info: {task_info}'
    #             tasks.append(task_info[0])

    #if args.task_idx is not None:
    #    tasks = [tasks[args.task_idx]]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # if task_config.env == 'ant_dir':
    #     env = AntDirEnv(tasks, args.n_tasks, include_goal = args.include_goal or args.multitask)
    # elif task_config.env == 'cheetah_dir':
    #     env = HalfCheetahDirEnv(tasks, include_goal = args.include_goal or args.multitask)
    # elif task_config.env == 'cheetah_vel':
    #     env = HalfCheetahVelEnv(tasks, include_goal = args.include_goal or args.multitask, one_hot_goal=args.one_hot_goal or args.multitask)
    # elif task_config.env == 'walker_params':
    #     env = WalkerRandParamsWrappedEnv(tasks, args.n_tasks, include_goal = args.include_goal or args.multitask)
    # elif task_config.env == 'ml45':
    #     env = ML45Env(include_goal=args.multitask or args.include_goal)
    # else:
    #     raise RuntimeError(f'Invalid env name {task_config.env}')

    if args.name is None:
        args.name = 'throwaway_test_run'
    if instance_idx == 0:
        name = args.name
    else:
        name = f'{args.name}_{instance_idx}'

    seed = args.seed if args.seed is not None else instance_idx
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # hardcoded
    # tasks = pickle.load(open(saved_tasks_path, 'rb'))
    task_data = joblib.load(saved_tasks_path)
    tasks = task_data['tasks']
    train_task_idxs = task_data['train_task_indices']
    test_task_idxs = task_data['eval_task_indices']
    # inner_buffers = [buffer_path_template.format(idx) for idx in train_task_idxs]
    # outer_buffers = [buffer_path_template.format(idx) for idx in train_task_idxs]
    # # test_buffers = [buffer_path_template.format(idx) for idx in test_task_idxs]
    # # TODO: return to test_task_idx
    # test_buffers = [buffer_path_template.format(idx) for idx in train_task_idxs]

    args.load_inner_buffer = True
    args.load_outer_buffer = True

    idx_to_buffer = load_buffers(pretrain_buffer_path, path_length=path_length, **load_buffer_kwargs)

    if env == 'ant_dir':
        env = AntDirEnv(tasks, args.n_tasks, include_goal = args.include_goal or args.multitask)
    elif env == 'cheetah_vel':
        env = HalfCheetahVelEnv(tasks, include_goal = args.include_goal or args.multitask, one_hot_goal=args.one_hot_goal or args.multitask)
    elif env == 'walker_params':
        env = WalkerRandParamsWrappedEnv(tasks, include_goal = args.include_goal or args.multitask)
    elif env == 'hopper_params':
        env = HopperRandParamsWrappedEnv(tasks, include_goal=args.include_goal or args.multitask)
    elif env == 'humanoid_dir':
        env = HumanoidDirEnv(tasks, include_goal=args.include_goal or args.multitask)
    else:
        import ipdb; ipdb.set_trace()

    if args.episode_length is not None:
        env._max_episode_steps = args.episode_length

    from src.maml_rawr import MAMLRAWR

    inner_buffers = outer_buffers = test_buffers = idx_to_buffer
    if not args.td3ctx:
        model = MAMLRAWR(args,
                         # task_config,
                         env, log_dir,
                         inner_buffers=inner_buffers,
                         outer_buffers=outer_buffers,
                         test_buffers=test_buffers,
                         test_tasks=test_task_idxs,
                         train_tasks=train_task_idxs,
                         name=name, training_iterations=args.train_steps,
                         visualization_interval=args.vis_interval, silent=instance_idx > 0, instance_idx=instance_idx,
                         gradient_steps_per_iteration=args.gradient_steps_per_iteration,
                         replay_buffer_length=args.replay_buffer_size, discount_factor=args.discount_factor, seed=seed,
                         is_rlkit_data=use_rlkit,
                         )
    else:
        # model = TD3Context(args, task_config, env, args.log_dir, name, 30, training_iterations=args.train_steps, silent=instance_idx > 0)
        pass

    model.train()


def load_buffers(pretrain_buffer_path, discount_factor=0.99, **kwargs):
    # pretrain_buffer_path = "/home/vitchyr/mnt2/log2/21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-sac-to-get-buffer-longer/21-02-22-ant-awac--exp7-ant-dir-4-eval-4-train-sac-to-get-buffer-longer_2021_02_23_06_09_23_id000--s270987/extra_snapshot_itr400.cpkl"
    # save_dir = Path(pretrain_buffer_path).parent
    # save_dir = Path(save_dir) / '{}_buffer'.format(output_format)
    # save_dir.mkdir(parents=True, exist_ok=True)
    # saved_replay_buffer = data['replay_buffer']
    # save_dir = Path(
    #     local_path_from_s3_or_local_path(pretrain_buffer_path)
    # ).parent / 'macaw_buffer'
    # save_dir.mkdir(exist_ok=True)
    # for k in saved_replay_buffer.task_buffers:
    #     buffer = saved_replay_buffer.task_buffers[k]
    #     data = rlkit_buffer_to_macaw_format(buffer, discount_factor, path_length)
    #     save_path = save_dir / 'macaw_buffer_task_{}.npy'.format(k)
    #     print('saving to', save_path)
    #     np.save(save_path, data)

    snapshot = joblib.load(pretrain_buffer_path)
    task_idx_to_buffer = {}
    key = 'replay_buffer'
    if key not in snapshot:
        snapshot[key] = snapshot['algorithm'].replay_buffer
    saved_replay_buffer = snapshot[key]
    for task_idx in saved_replay_buffer.task_buffers:
        rlkit_buffer = saved_replay_buffer.task_buffers[task_idx]
        buffer = rlkit_buffer_to_macaw_format(
            rlkit_buffer, discount_factor, **kwargs
        )
        task_idx_to_buffer[task_idx] = buffer
    return task_idx_to_buffer


class MyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, type):
            return {'$class': o.__module__ + "." + o.__name__}
        elif isinstance(o, Enum):
            return {
                '$enum': o.__module__ + "." + o.__class__.__name__ + '.' + o.name
            }
        elif callable(o):
            return {
                '$function': o.__module__ + "." + o.__name__
            }
        return json.JSONEncoder.default(self, o)


def save_variant(log_dir, variant):
    exp_name = log_dir.split('/')[-2]
    unique_id = str(uuid.uuid4())

    variant_to_save = variant.copy()
    variant_to_save['unique_id'] = unique_id
    variant_to_save['exp_name'] = exp_name
    variant_to_save['trial_name'] = log_dir.split('/')[-1]
    print(
        json.dumps(ppp.dict_to_safe_json(variant_to_save, sort=True), indent=2)
    )
    variant_log_path = osp.join(log_dir, 'variant.json')
    with open(variant_log_path, "w") as f:
        json.dump(variant_to_save, f, indent=2, sort_keys=True, cls=MyEncoder)


def run_doodad_experiment(doodad_config: DoodadConfig, params):
    # print(params)
    # import ipdb; ipdb.set_trace()
    save_doodad_config(doodad_config)
    log_dir = doodad_config.output_directory
    save_variant(log_dir, params)

    log_dir = doodad_config.output_directory
    exp_name = log_dir.split('/')[-2]
    setup_logger(logger, variant=params, base_log_dir=None, exp_name=exp_name, log_dir=log_dir)
    run(log_dir, **params)
