import argparse
import gym


from src.envs import PointMass1DEnv
from src.maml_rawr import MAMLRAWR


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_steps', type=int, default=100000)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--vis_interval', type=int, default=10)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('name', type=str)
    return parser.parse_args()


def run(args: argparse.Namespace):
    # envs = [PointMass1DEnv(0), PointMass1DEnv(1)]
    envs = [PointMass1DEnv(0)]
    maml_rawr = MAMLRAWR(envs, training_iterations=args.train_steps, device=args.device)

    maml_rawr.train(args)


if __name__ == '__main__':
    run(get_args())
