from envs import PointMass1DEnv
from maml_rawr import MAMLRAWR
import argparse
import gym


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_steps', type=int, default=100000)
    parser.add_argument('--device', type=str, default='cuda:0')
    return parser.parse_args()


def run(args: argparse.Namespace):
    envs = [PointMass1DEnv(0), PointMass1DEnv(1)]
    maml_rawr = MAMLRAWR(envs, [32], [32], training_iterations=args.train_steps, device=args.device)

    maml_rawr.train()


if __name__ == '__main__':
    run(get_args())
