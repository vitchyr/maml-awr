import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def load_mt(path):
    x = np.load(path, allow_pickle=True)[()]['x']
    y = np.load(path, allow_pickle=True)[()]['y']

    y = gaussian_filter1d(y, sigma=3)
    return trim(x,y,2000)
    #return x, y


def extract_macaw(path, terminate: int = None, prefix: str = None):
    y = []
    x = []
    d = None
    macaw = 'macaw' in path
    if 'cheetah' in path:
        xscale = 2000
    if 'walker' in path or 'ant' in path:
        xscale = 4000
    pearl = False
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
                        if not tag.startswith(prefix):
                            continue

                        if d is None:
                            d = defaultdict(list)
                        if prefix == 'test_task':
                            pearl = True
                        if pearl:
                            step *= xscale
                        d[step].append(value)
            except Exception as e:
                print(entry)
                raise e
    except Exception as e:
        print(e)

    if d is not None:
        x = np.sort(list(d.keys()))
        y = np.array([np.mean(d[x_]) for x_ in x])
    y = gaussian_filter1d(y, sigma=4)        
    x, y = np.array(x).astype(np.float32), np.array(y)
    return trim(x,y,2000)

def trim(x, y, val):
    v = np.where(np.squeeze(x) >= val)[0]
    if len(v) > 0:
        v = v[0]
        return x[v:], y[v:]
    else:
        return x, y

def run(args: argparse.Namespace):
    dir_macaw_x, dir_macaw_y = extract_macaw(args.macaw_dir_path, args.terminate)
    vel_macaw_x, vel_macaw_y = extract_macaw(args.macaw_vel_path, args.terminate)
    walker_macaw_x, walker_macaw_y = extract_macaw(args.macaw_walker_path, args.terminate)
    ant_macaw_x, ant_macaw_y = extract_macaw(args.macaw_ant_path, args.terminate)

    dir_pearl_x, dir_pearl_y = extract_macaw(args.pearl_dir_path, args.terminate, prefix='test_task')
    vel_pearl_x, vel_pearl_y = extract_macaw(args.pearl_vel_path, args.terminate, prefix='test_task')
    walker_pearl_x, walker_pearl_y = extract_macaw(args.pearl_walker_path, args.terminate, prefix='test_task')
    ant_pearl_x, ant_pearl_y = extract_macaw(args.pearl_ant_path, args.terminate, prefix='test_task')

    dir_mt_x, dir_mt_y = load_mt(args.mt_dir_path)
    vel_mt_x, vel_mt_y = load_mt(args.mt_vel_path)
    walker_mt_x, walker_mt_y = load_mt(args.mt_walker_path)
    ant_mt_x, ant_mt_y = load_mt(args.mt_ant_path)

    w = 5.4*3
    h = w/2
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(w,h))

    color1 = next(axes[0,0]._get_lines.prop_cycler)['color']
    color2 = next(axes[0,0]._get_lines.prop_cycler)['color']
    color3 = next(axes[0,0]._get_lines.prop_cycler)['color']
    color3 = next(axes[0,0]._get_lines.prop_cycler)['color']

    axes[0,0].plot(dir_macaw_x, dir_macaw_y, color=color1, linewidth=2)
    axes[0,0].plot([dir_macaw_x[-1], dir_pearl_x[-1]], [dir_macaw_y[-1]] * 2, color=color1, linewidth=2)
    axes[0,0].plot(dir_mt_x, dir_mt_y, linewidth=2, color=color2)
    axes[0,0].plot([dir_mt_x[-1], dir_pearl_x[-1]], [dir_mt_y[-1]] * 2, color=color2, linewidth=2)
    axes[0,0].plot(dir_pearl_x, dir_pearl_y, linewidth=2, color=color3)
    axes[0,0].set_title('Cheetah-Direction')
    axes[0,0].set_xlabel('Training Steps')
    axes[0,0].set_ylabel('Reward')
    
    axes[0,1].plot(vel_macaw_x, vel_macaw_y, linewidth=2, color=color1)
    axes[0,1].plot([vel_macaw_x[-1], vel_pearl_x[-1]], [vel_macaw_y[-1]] * 2, color=color1, linewidth=2)
    axes[0,1].plot(vel_mt_x, vel_mt_y, linewidth=2, color=color2)
    axes[0,1].plot([vel_mt_x[-1], vel_pearl_x[-1]], [vel_mt_y[-1]] * 2, color=color2, linewidth=2)
    axes[0,1].plot(vel_pearl_x, vel_pearl_y, linewidth=2, color=color3)
    axes[0,1].set_title('Cheetah-Velocity')
    axes[0,1].set_xlabel('Training Steps')
    axes[0,1].set_ylabel('Reward')

    axes[1,0].plot(walker_macaw_x, walker_macaw_y, linewidth=2, label='MACAW', color=color1)
    axes[1,0].plot([walker_macaw_x[-1], walker_pearl_x[-1]], [walker_macaw_y[-1]] * 2, linewidth=2, color=color1)
    axes[1,0].plot(walker_mt_x, walker_mt_y, linewidth=2, label='MT + fine tune', color=color2)
    axes[1,0].plot([walker_mt_x[-1], walker_pearl_x[-1]], [walker_mt_y[-1]] * 2, linewidth=2, color=color2)
    axes[1,0].plot(walker_pearl_x, walker_pearl_y, linewidth=2, label='PEARL', color=color3)
    axes[1,0].set_title('Walker-Params')
    axes[1,0].set_xlabel('Training Steps')
    axes[1,0].set_ylabel('Reward')
    #axes[1,0].legend(loc=2)
    
    axes[1,1].plot(ant_macaw_x, ant_macaw_y, linewidth=2, label='MACAW', color=color1)
    axes[1,1].plot([ant_macaw_x[-1], ant_pearl_x[-1]], [ant_macaw_y[-1]] * 2, linewidth=2, color=color1)
    axes[1,1].plot(ant_mt_x, ant_mt_y, linewidth=2, label='MT + fine tune', color=color2)
    axes[1,1].plot([ant_mt_x[-1], ant_pearl_x[-1]], [ant_mt_y[-1]] * 2, linewidth=2, color=color2)
    axes[1,1].plot(ant_pearl_x, ant_pearl_y, linewidth=2, label='PEARL', color=color3)
    axes[1,1].set_title('Ant-Direction')
    axes[1,1].set_xlabel('Training Steps')
    axes[1,1].set_ylabel('Reward')
    axes[1,1].legend(loc='lower left', bbox_to_anchor=(0,0.13))
    for ax in axes:
        for a in ax:
            a.set_xscale('log')
            a.tick_params(axis=u'both', which=u'both',length=0)
            a.grid(linestyle='--', linewidth=0.75)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['bottom'].set_visible(False)
            a.spines['left'].set_visible(False)

    #plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.tight_layout()
    fig.savefig(args.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--macaw_dir_path', type=str)
    parser.add_argument('--macaw_vel_path', type=str)
    parser.add_argument('--macaw_walker_path', type=str)
    parser.add_argument('--macaw_ant_path', type=str)
    parser.add_argument('--pearl_dir_path', type=str)
    parser.add_argument('--pearl_vel_path', type=str)
    parser.add_argument('--pearl_walker_path', type=str)
    parser.add_argument('--pearl_ant_path', type=str)
    parser.add_argument('--mt_dir_path', type=str)
    parser.add_argument('--mt_vel_path', type=str)
    parser.add_argument('--mt_walker_path', type=str)
    parser.add_argument('--mt_ant_path', type=str)
    parser.add_argument('--terminate', type=int, default=None)
    parser.add_argument('--name', type=str, default='mujoco_neurips')
    run(parser.parse_args())
