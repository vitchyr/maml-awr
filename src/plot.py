from tensorflow.python.summary.summary_iterator import summary_iterator
import argparse
import glob
import os
import re
import numpy as np


def run(args: argparse.Namespace):
    runs = {}
    for path in glob.glob(args.path + '/*'):
        basename = os.path.basename(path)
        if re.match('^.*_[0-9]+$', basename) is not None:
            runs[basename] = []
        else:
            basename = basename[:basename.rindex('_')]

        events_file = glob.glob(path + '/tb/*')[0]
        it = summary_iterator(events_file)

        rewards = []
        for x in it:
            try:
                tag = x.summary.value[0].tag
                value = x.summary.value[0].simple_value
                if tag.startswith('Reward_Train/Task_'):
                    idx = int(tag[tag.rindex('_')+1:])
                    while idx >= len(rewards):
                        rewards.append([])
                    rewards[idx].append(value)
            except Exception as e:
                print(e)

        min_length = min([len(r) for r in rewards])
        rewards = np.array([r[:min_length] for r in rewards])
        reduced_rewards = np.mean(rewards, 0)
        import pdb; pdb.set_trace()
            

def get_args():
    args = argparse.ArgumentParser()
    args.add_argument('--path', type=str)
    return args.parse_args()


if __name__ == '__main__':
    run(get_args())
