import argparse
import gym
import numpy as np
from multiprocessing import Process

from src.envs import PointMass1DEnv
from src.maml_rawr import MAMLRAWR


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_steps', type=int, default=20000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--vis_interval', type=int, default=200)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--task_idx', type=int, default=None)
    parser.add_argument('--instances', type=int, default=1)
    parser.add_argument('--name', type=str, default=None)
    return parser.parse_args()


def run(args: argparse.Namespace):
    for instance_idx in range(args.instances):
        # envs = [PointMass1DEnv(0), PointMass1DEnv(1)]
        envs = [PointMass1DEnv(args.task_idx, fix_random_task=True)]
        # envs = [gym.make('HalfCheetah-v2')]
        
        if args.name is None:
            args.name = 'throwaway_test_run'
        if instance_idx == 0:
            name = args.name
        else:
            name = f'{args.name}_{instance_idx}'
        
        maml_rawr = MAMLRAWR(envs, args.log_dir, name, training_iterations=args.train_steps, device=args.device,
                             visualization_interval=args.vis_interval, silent=args.instances > 1)        

        subprocess = Process(target=maml_rawr.train)
        subprocess.start()


if __name__ == '__main__':
    run(get_args())
