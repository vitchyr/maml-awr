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
import glob


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


def extract_macaw(path, terminate: int = None, prefix: str = 'Eval_Reward_FTStep/Task_', step_divisor: int = 1):
    path = glob.glob(path + '/*')[0]
    n_steps = 1000 if prefix is not "task_" else 980
    n_steps = 100
    steps = np.arange(n_steps) * 200 / 1000.
    y_data = np.full((n_steps, 13), np.nan)
    try:
        for entry in summary_iterator(path):
            try:
                if len(entry.summary.value):
                    v = entry.summary.value[0]
                    step, tag, value = entry.step, v.tag, v.simple_value
                    if step >= n_steps:
                        continue
                    if (not tag.startswith(prefix)) or tag.endswith('Task_26'):
                        continue
                    task = [int(s) for s in tag.split('_') if s.isdigit()][0] - 27
                    y_data[step//step_divisor,task] = value
                    
            except Exception as e:
                print(entry)
                raise e
    except Exception as e:
        print(e)

    assert not np.any(np.isnan(y_data))
    smoothed = []
    for col in y_data.T:
        #smoothed.append(gaussian_filter1d(col, sigma=4))
        smoothed.append(col)
    smoothed = np.stack(smoothed).T
    mean = smoothed.mean(-1)
    std = smoothed.std(-1) / 13**0.5
    start = smoothed[:1]
    endn = 10
    end = smoothed[-endn:]
    inc = end-start
    n = endn*13
    print('start', start.mean(), start.std() / 13**0.5)
    print('end', end.mean(), end.std() / n**0.5)
    print('inc', inc.mean(), inc.std() / n**0.5)
    return steps, mean, std


def trim(x, y, val):
    v = np.where(np.squeeze(x) > val)[0]
    if len(v) > 0:
        v = v[0]
        return x[:v], y[:v]
    else:
        return x, y

def run(args: argparse.Namespace):
    vel_macaw_x, vel_macaw_y, vel_macaw_std = extract_macaw(args.macaw_vel_path, args.terminate)
    vel_pearl_x, vel_pearl_y, vel_pearl_std = extract_macaw(args.pearl_vel_path, args.terminate, prefix='task_')

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9,6))
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
    alpha= 0.5
    axes.plot(vel_macaw_x, vel_macaw_y, linewidth=2, color=color1)
    axes.fill_between(vel_macaw_x, vel_macaw_y-vel_macaw_std, vel_macaw_y + vel_macaw_std, color=color1, label='MACAW', alpha=alpha, zorder=2)
    axes.plot(vel_pearl_x, vel_pearl_y, linewidth=2, color=color3)
    axes.fill_between(vel_pearl_x, vel_pearl_y-vel_pearl_std, vel_pearl_y + vel_pearl_std, color=color3, label='Offline PEARL', alpha=alpha, zorder=2)

    axes.set_xlabel('Environment Steps (thousands)')
    axes.set_ylabel('Mean Reward')
    #axes.legend(loc='center left')
    plt.suptitle('Online Fine-Tuning on OOD Tasks')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig('rebuttal_figs/' + args.name, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--macaw_vel_path', type=str, default='log/iclr_rebuttal/macaw_vel_extrapolation_online2_1e-6/tb')
    parser.add_argument('--pearl_vel_path', type=str, default='/iris/u/rafailov/ICLR_rebuttal/v2/oyster/log/cheetah_vel_extrapolation/seed_0/V4')
    parser.add_argument('--terminate', type=int, default=None)
    parser.add_argument('--name', type=str, default='arch')
    run(parser.parse_args())
