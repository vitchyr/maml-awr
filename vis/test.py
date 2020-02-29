
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from tensorflow.python.summary.summary_iterator import summary_iterator
import re
from collections import defaultdict
import numpy as np
from matplotlib.ticker import MaxNLocator

'''
SMALL_SIZE = 14
MEDIUM_SIZE = 16
BIGGER_SIZE = 20

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
'''


def get_vals(path: str):

    vals = []
    for entry in summary_iterator(path):
        try:
            step = entry.step
            v = entry.summary.value[0]
            tag, value = v.tag, v.simple_value
            task = int(re.search(r'\d+', tag).group())
            print(step, tag, value)
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


def run(args: argparse.Namespace):
    path = '/iris/u/rafailov/pearl_logs/tensorboard/cheetah_vel_dist_c8/events.out.tfevents.1581484270.iris-hp-z8.stanford.edu'
    get_vals(path)
    macaw = get_vals(args.macaw_path)
    mt = get_vals(args.mt_path)
    
    sym = '-'

    x = np.arange(0,21)
    ax = plt.figure().gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(x,macaw[1],color=color, label='MACAW (ours)')
    #plt.plot(x,macaw[0], sym, color=color, label='MACAW (mean)')
    plt.fill_between(x, macaw[3], macaw[4], alpha=0.5, interpolate=True)
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot(x,mt[1], '--', color=color, label='MT + finetune')
    #plt.plot(x,mt[0], sym, color=color, label='MT + finetune (mean)')
    plt.fill_between(x, mt[3], mt[4], alpha=0.5, interpolate=True)
    color = next(ax._get_lines.prop_cycler)['color']
    color = next(ax._get_lines.prop_cycler)['color']
    plt.plot([0,20], [-160,-160], '-.', color=color, label='PEARL (|c|=256)')
    ax.set_xticks(np.arange(0, 21, step=2))
    plt.legend(loc=4)
    plt.title('Out of Distribution Performance')
    plt.xlabel('Number of gradient steps')
    plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig('ood')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--macaw_path', type=str)
    parser.add_argument('--mt_path', type=str)
    run(parser.parse_args())
