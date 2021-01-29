import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

import pickle
import numpy as np
import argparse
import re
from scipy.ndimage.filters import gaussian_filter1d


LEGEND_SIZE=19
SMALL_SIZE = 20
MEDIUM_SIZE = 23
BIGGER_SIZE = 30

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=LEGEND_SIZE)    # legend fontsize
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
    pearl = False
    try:
        for entry in summary_iterator(path):
            try:
                if len(entry.summary.value):
                    v = entry.summary.value[0]
                    step, tag, value = entry.step, v.tag, v.simple_value
                    if terminate and step > terminate:
                        break
                    if tag != 'Eval_Reward/Mean' and tag != 'test_tasks_mean_reward/mean_return':
                        continue
                    if tag == 'test_tasks_mean_reward/mean_return':
                        pearl = True
                        step *= 2000

                    #print(tag, step, value)
                    y.append(value)
                    x.append(step)
            except Exception as e:
                print(entry)
                raise e
    except Exception as e:
        print(e)

    sigma = 5
    y = gaussian_filter1d(y, sigma=sigma if not pearl else sigma/8.)        
    return np.array(x).astype(np.float32) / 1000, np.array(y)


def trim(x, y, val):
    v = np.where(np.squeeze(x) > val)[0]
    if len(v) > 0:
        v = v[0]
        return x[:v], y[:v]
    else:
        return x, y

def run(args: argparse.Namespace):
    '''
    macaw_half = -185.4 # @386k steps
    macaw_quarter = -201.5 # @352k steps
    macaw_eighth = -125.2 # @184k steps
    macaw_sixteenth = -126.4 # @268k steps
    '''

    macaw_all = -128
    macaw_half = -169.7 # @104k
    macaw_quarter = -200.6 # @92k
    macaw_eighth = -130 # @94k
    macaw_sixteenth = -103.1 # @132k

    '''
    # 7x data
    pearl_half = -42
    pearl_quarter = -70
    pearl_eighth = -150
    pearl_sixteenth = -320
    '''

    pearl_all = -306
    pearl_half = -137.5
    pearl_quarter = -76.035
    pearl_eighth = -116.5
    pearl_sixteenth = -313.1

    mt_all = -171
    mt_half = -200
    mt_quarter = -249
    mt_eighth = -270
    mt_sixteenth = -267
    
    w = 9
    h = 6
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(w,h))
    axes.tick_params(axis=u'both', which=u'both',length=0)
    axes.grid(linestyle='--', linewidth=1.25)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    
    color1 = next(axes._get_lines.prop_cycler)['color']
    color2 = next(axes._get_lines.prop_cycler)['color']
    color3 = next(axes._get_lines.prop_cycler)['color']
    color3 = next(axes._get_lines.prop_cycler)['color']

    bw = 0.2


    x = np.arange(5)
    #tasks = ['35 tasks', '20 tasks', '10 tasks', '5 tasks', '3 tasks']
    tasks = ['35', '20', '10', '5', '3']
    macaw = [macaw_all, macaw_half, macaw_quarter, macaw_eighth, macaw_sixteenth]
    pearl = [pearl_all, pearl_half, pearl_quarter, pearl_eighth, pearl_sixteenth]
    mt = [mt_all, mt_half, mt_quarter, mt_eighth, mt_sixteenth]
    lp, = axes.plot(x, pearl, marker='s', color=color3, linewidth=5, markersize=10,label='Off. PEARL')
    lmac, = axes.plot(x, macaw, marker='s', color=color1, linewidth=7, markersize=12, label='MACAW')
    lmt, = axes.plot(x, mt, marker='s', color=color2, linewidth=5, markersize=10, label='Off. MT+FT')
    axes.set_xticks(x)
    axes.set_xticklabels(tasks)
    axes.set_title('Test Performance with Sparse Task Sampling')
    axes.set_ylim([-350,-0.01])
    axes.set_xlabel('Number of Training Tasks')
    axes.set_ylabel('Asmyptotic Reward')
    handles, labels = axes.get_legend_handles_labels()
    order = [1,2,0]
    axes.legend([handles[idx] for idx in order],[labels[idx] for idx in order], loc='upper left', prop={'size': 17})#, bbox_to_anchor=(0.475,0.423))
    plt.tight_layout()
    fig.savefig(args.name, bbox_inches = "tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--half_path', type=str)
    parser.add_argument('--quarter_path', type=str)
    parser.add_argument('--eighth_path', type=str)
    parser.add_argument('--sixteenth_path', type=str)
    parser.add_argument('--pearl_half_path', type=str)
    parser.add_argument('--pearl_quarter_path', type=str)
    parser.add_argument('--pearl_eighth_path', type=str)
    parser.add_argument('--pearl_sixteenth_path', type=str)
    '''
    parser.add_argument('--half_path', type=str)
    parser.add_argument('--quarter_path', type=str)
    parser.add_argument('--eighth_path', type=str)
    parser.add_argument('--sixteenth_path', type=str)
    '''
    parser.add_argument('--terminate', type=int, default=None)
    parser.add_argument('--name', type=str, default='task_ablation')
    run(parser.parse_args())
