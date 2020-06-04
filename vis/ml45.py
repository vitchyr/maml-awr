import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
from tensorflow.python.summary.summary_iterator import summary_iterator
import pickle
import numpy as np
import argparse
import re
from scipy.ndimage.filters import gaussian_filter1d
from collections import defaultdict


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


def extract_macaw(path, terminate: int = None, prefix: str = None, suffix: str = None):
    y = []
    x = []
    xscale = 200
    pearl = False
    d = None
    try:
        for entry in summary_iterator(path):
            try:
                if len(entry.summary.value):
                    v = entry.summary.value[0]
                    step, tag, value = entry.step, v.tag, v.simple_value
                    if terminate and step > terminate:
                        break
                    if prefix is None:
                        if tag != 'Eval_Reward/Mean' and tag != 'test_tasks_mean_reward/mean_return':
                            continue
                        if tag == 'test_tasks_mean_reward/mean_return':
                            pearl = True
                        if pearl:
                            step *= xscale
                        y.append(value)
                        x.append(step)
                    else:
                        if not tag.startswith(prefix) or (suffix is not None and not tag.endswith(suffix)):
                            continue
                        if d is None:
                            d = defaultdict(list)
                        d[step].append(value)
            except Exception as e:
                print(entry)
                raise e
    except Exception as e:
        print(e)

    if d is not None:
        x = np.sort(list(d.keys()))
        y = np.array([np.mean(d[x_]) for x_ in x])
    y = gaussian_filter1d(y, sigma=2 if prefix else 10)
    return np.array(x).astype(np.float32) / 1000, np.array(y)


def trim(x, y, val):
    v = np.where(np.squeeze(x) > val)[0]
    if len(v) > 0:
        v = v[0]
        return x[:v], y[:v]
    else:
        return x, y


def load_mt(path):
    x = np.load(path, allow_pickle=True)[()]['x']
    y = np.load(path, allow_pickle=True)[()]['y']
    success = np.load(path, allow_pickle=True)[()]['success']
    return x/1000, y, success

def cumavg(array):
    return array.cumsum() / (1 + np.arange(array.shape[0]))

def run(args: argparse.Namespace):
    macaw_x, macaw_reward = extract_macaw(args.macaw_path, args.terminate)
    _, macaw_success = extract_macaw(args.macaw_path, args.terminate, prefix='Eval_Success')
    macaw_success = cumavg(macaw_success)

    pearl_x, pearl_reward = extract_macaw(args.pearl_path, args.terminate)
    _, pearl_success = extract_macaw(args.pearl_path, args.terminate, prefix='test_tasks_mean_succes/mean_succes')
    pearl_success = cumavg(pearl_success)
    
    mt_x, mt_reward, mt_success = load_mt(args.mt_path)
    mt_success = cumavg(mt_success)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9,6))
    axes2 = axes.twinx()
    axes.tick_params(axis=u'both', which=u'both',length=0)
    axes.grid(linestyle='--', linewidth=0.75)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    axes2.tick_params(axis=u'both', which=u'both',length=0)
    axes2.spines['top'].set_visible(False)
    axes2.spines['right'].set_visible(False)
    axes2.spines['bottom'].set_visible(False)
    axes2.spines['left'].set_visible(False)
    
    color = next(axes._get_lines.prop_cycler)['color']
    l1, = axes.plot(macaw_x, macaw_reward, color=color, label='MACAW')
    l2, = axes2.plot(macaw_x, macaw_success, '--', color=color)
    color = next(axes._get_lines.prop_cycler)['color']
    color = next(axes._get_lines.prop_cycler)['color']
    axes.plot(pearl_x, pearl_reward, color=color, label='PEARL')
    axes2.plot(pearl_x, pearl_success,  '--', color=color)
    color = next(axes._get_lines.prop_cycler)['color']
    #axes.plot(mt_x, mt_reward, color=color, label='MT + Fine tune')
    #axes2.plot(mt_x, mt_success,  '--', color=color)
    axes.set_title('Meta-World ML45 Benchmark')
    axes.set_xlabel('Training Steps (thousands)')
    axes.set_ylabel('Reward')
    axes2.set_ylabel('Cumulative Success Rate')
    leg1 = axes.legend(loc='upper right')
    leg2 = axes2.legend([l1,l2], ['Reward', 'Cum. Success Rate'], loc='upper center')
    leg2.legendHandles[0].set_color('black')
    leg2.legendHandles[1].set_color('black')

    plt.tight_layout()
    fig.savefig(args.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--macaw_path', type=str)
    parser.add_argument('--pearl_path', type=str)
    parser.add_argument('--mt_path', type=str)
    parser.add_argument('--terminate', type=int, default=None)
    parser.add_argument('--name', type=str, default='ML45')
    run(parser.parse_args())
