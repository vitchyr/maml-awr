import argparse
import gym
import numpy as np
import torch
from multiprocessing import Process

from src.envs import PointMass1DEnv, HalfCheetahDirEnv
from src.maml_rawr import MAMLRAWR


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_steps', type=int, default=100000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--vis_interval', type=int, default=200)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--task_idx', type=int, default=None)
    parser.add_argument('--instances', type=int, default=1)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--inline_render', action='store_true')
    parser.add_argument('--env', type=str, default='point_mass')
    parser.add_argument('--gym_env', type=str, default=None)
    parser.add_argument('--gradient_steps_per_iteration', type=int, default=10)
    parser.add_argument('--replay_buffer_size', type=int, default=1000)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--eta2', type=float, default=1e-4)
    return parser.parse_args()


def run(args: argparse.Namespace):
    for instance_idx in range(args.instances):
        if args.gym_env is None:
            if args.env == 'point_mass':
                envs = [PointMass1DEnv(0), PointMass1DEnv(-1)]
            elif args.env == 'cheetah_dir':
                envs = [HalfCheetahDirEnv(0), HalfCheetahDirEnv(1)]
            # envs = [PointMass1DEnv(task_idx) for task_idx in range(PointMass1DEnv.n_tasks())]
            # envs = [PointMass1DEnv(args.task_idx, fix_random_task=True)]
        else:
            envs = [gym.make(args.gym_env)]
        
        if args.name is None:
            args.name = 'throwaway_test_run'
        if instance_idx == 0:
            name = args.name
        else:
            name = f'{args.name}_{instance_idx}'

        if args.gym_env is None and args.env == 'point_mass':
            network_shape = [32, 32]
            batch_size = 64
        else:
            network_shape = [128, 64]
            batch_size = 256
            
        maml_rawr = MAMLRAWR(envs, args.log_dir, name, network_shape, network_shape, batch_size=batch_size, training_iterations=args.train_steps,
                             device=args.device, visualization_interval=args.vis_interval, silent=args.instances > 1,
                             inline_render=args.inline_render, gradient_steps_per_iteration=args.gradient_steps_per_iteration,
                             replay_buffer_length=args.replay_buffer_size, discount_factor=args.discount_factor, eta2=args.eta2)

        if args.instances > 1:
            subprocess = Process(target=maml_rawr.train)
            subprocess.start()
        else:
            if args.profile:
                import cProfile
                cProfile.runctx('maml_rawr.train()', sort='cumtime', locals=locals(), globals=globals())
            else:
                maml_rawr.train()


if __name__ == '__main__':
    run(get_args())
