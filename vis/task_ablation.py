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
    macaw_half_x, macaw_half_y = extract_macaw(args.half_path, args.terminate)
    macaw_quarter_x, macaw_quarter_y = extract_macaw(args.quarter_path, args.terminate)
    macaw_eighth_x, macaw_eighth_y = extract_macaw(args.eighth_path, args.terminate)
    macaw_sixteenth_x, macaw_sixteenth_y = extract_macaw(args.sixteenth_path, args.terminate)

    pearl_half_x, pearl_half_y = extract_macaw(args.pearl_half_path, args.terminate)
    pearl_quarter_x, pearl_quarter_y = extract_macaw(args.pearl_quarter_path, args.terminate)
    pearl_eighth_x, pearl_eighth_y = extract_macaw(args.pearl_eighth_path, args.terminate)
    pearl_sixteenth_x, pearl_sixteenth_y = extract_macaw(args.pearl_sixteenth_path, args.terminate)

    w = 9
    h = 6
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(w,h))
    axes.tick_params(axis=u'both', which=u'both',length=0)
    axes.grid(linestyle='--', linewidth=0.75)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    
    color1 = next(axes._get_lines.prop_cycler)['color']
    color2 = next(axes._get_lines.prop_cycler)['color']
    color2 = next(axes._get_lines.prop_cycler)['color']
    color3 = next(axes._get_lines.prop_cycler)['color']
    color4 = next(axes._get_lines.prop_cycler)['color']
    color4 = next(axes._get_lines.prop_cycler)['color']
    color4 = next(axes._get_lines.prop_cycler)['color']

    l1, = axes.plot(macaw_half_x, macaw_half_y, color=color1, label='20 Tasks')
    l2, = axes.plot(pearl_half_x, pearl_half_y, '--', color=color1)

    axes.plot(macaw_quarter_x, macaw_quarter_y, color=color2, label='10 Tasks')
    axes.plot(pearl_quarter_x, pearl_quarter_y, '--', color=color2)

    axes.plot(macaw_eighth_x, macaw_eighth_y, color=color3, label='5 Tasks')
    axes.plot(pearl_eighth_x, pearl_eighth_y, '--', color=color3)

    axes.plot(macaw_sixteenth_x, macaw_sixteenth_y, color=color4, label='3 Tasks')
    axes.plot(pearl_sixteenth_x, pearl_sixteenth_y, '--', color=color4)

    axes.set_title('Test Performance Under Various Task Samplings')

    #axes.set_xscale('log')
    axes.set_xlabel('Training Steps (thousands)')
    axes.set_ylabel('Reward')
    leg1 = axes.legend(loc='lower right')
    leg2 = axes.legend([l1,l2], ['MACAW', 'PEARL'], loc='lower center')
    leg2.legendHandles[0].set_color('black')
    leg2.legendHandles[1].set_color('black')
    plt.gca().add_artist(leg1)
    plt.tight_layout()
    fig.savefig(args.name)


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
