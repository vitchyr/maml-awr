import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator
import pickle
import numpy as np
import argparse
import re
from scipy.ndimage.filters import gaussian_filter1d


SMALL_SIZE = 14
MEDIUM_SIZE = 18
BIGGER_SIZE = 22

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


def get_vals(path: str):
    vals = np.empty((27,21))
    vals.fill(np.float('nan'))
    for entry in summary_iterator(path):
        try:
            step = entry.step
            v = entry.summary.value[0]
            tag, value = v.tag, v.simple_value
            task = int(re.search(r'\d+', tag).group())
            vals[task-13,step] = value
        except Exception as e:
            pass

    assert not np.isnan(vals).any()
    means = vals.mean(0)
    medians = np.median(vals, 0)
    stds = vals.std(0)
    lowerq = np.percentile(vals, 25, 0)
    upperq = np.percentile(vals, 75, 0)
    mins = np.min(vals, 0)
    maxs = np.max(vals, 0)

    return means, medians, stds, lowerq, upperq, mins, maxs


def extract_macaw(path, terminate: int = None):
    y = []
    x = []
    for entry in summary_iterator(path):
        try:
            if len(entry.summary.value):
                v = entry.summary.value[0]
                step, tag, value = entry.step, v.tag, v.simple_value
                if terminate and step > terminate:
                    break
                if tag != 'Reward_Train/Mean':
                    continue
                print(tag, step, value)
                y.append(value)
                x.append(step)
        except Exception as e:
            print(entry)
            raise e

    y = gaussian_filter1d(y, sigma=2)        
    return np.array(x).astype(np.float32), np.array(y)


def trim(x, y, val):
    v = np.where(np.squeeze(x) > val)[0]
    if len(v) > 0:
        v = v[0]
        return x[:v], y[:v]
    else:
        return x, y

def run(args: argparse.Namespace):
    adv_x, adv_y = extract_macaw('/iris/u/em7/code/maml-rawr/log/adv_retest/rollback/tb/events.out.tfevents.1586660504.iris-ws-3.stanford.edu.18046.0', args.terminate)
    noadv_x, noadv_y = extract_macaw('/iris/u/em7/code/maml-rawr/log/adv_retest/rollback_noadv/tb/events.out.tfevents.1586661027.iris-ws-3.stanford.edu.32631.0', args.terminate)
    adv_x, adv_y = trim(adv_x, adv_y, args.terminate)
    noadv_x, noadv_y = trim(noadv_x, noadv_y, args.terminate)

    fig, axes = plt.subplots(figsize=(8,6))
    
    axes.plot(noadv_x, noadv_y, label='MAML + AWR')
    axes.plot(adv_x, adv_y, label='MACAW (Ours)')
    #color = next(axes._get_lines.prop_cycler)['color']
    #plt.plot([0,20], [-237,-237], '--', color=color, label='Behavior Policy')
    axes.set_title('Cheetah-Velocity Comparison')
    axes.set_xlabel('Number of gradient steps')
    axes.set_ylabel('Average Reward')
    plt.legend()

    fig.savefig(args.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--terminate', type=int, default=None)
    parser.add_argument('--name', type=str)
    run(parser.parse_args())
