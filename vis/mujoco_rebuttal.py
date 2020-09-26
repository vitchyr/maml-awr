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
import glob, os


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


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)

def extract(path, tag_: str, terminate: int = None, xscale=1, smooth=1):
    paths = glob.glob(path)

    xs, ys = [], []
    for path in paths:
        if 'tfevents' in path:
            fn = path
        else:
            fn = None
        for root, dirs, files in os.walk(path, topdown=False):
            if fn is not None:
                break
            for name in files:
                if fn is not None:
                    break
                if "tfevents" in name:
                    fn = os.path.join(root, name)
        print(f'Got TB: {fn}')
        x,y = [], []
        try:
            for entry in summary_iterator(fn):
                try:
                    if len(entry.summary.value):
                        v = entry.summary.value[0]
                        step, tag, value = entry.step, v.tag, v.simple_value
                        if terminate and step > terminate:
                            break
                        if tag == tag_:
                            y.append(value)
                            x.append(step * xscale)
                except Exception as e:
                    print(e, entry)
                    pass
        except Exception as e:
            print(e)
            pass
        xs.append(x)
        ys.append(running_mean(y, smooth))
    ylens = [len(z) for z in ys]

    ys = [y_[:min(ylens)] for y_ in ys]
    xs = [x_[:min(ylens)] for x_ in xs]
    
    ys = np.array(ys)
    xs = np.array(xs)
    ymean = np.mean(ys,0)
    ystd = np.std(ys, 0)
    x = xs[0]
    #print(x)
    x = x[:ymean.shape[0]]
    print(x.shape, ymean.shape, ystd.shape)
    return x, ymean, ystd

def trim(x, y, val):
    v = np.where(np.squeeze(x) >= val)[0]
    if len(v) > 0:
        v = v[0]
        return x[v:], y[v:]
    else:
        return x, y

def run(args: argparse.Namespace):

    walker_mt_x, walker_mt_y, walker_mt_std = extract(args.mt_walker_path, 'FT_Eval_Reward/Mean_Step20', args.terminate, smooth=5)
    dir_mt_x, dir_mt_y, dir_mt_std = extract(args.mt_dir_path, 'FT_Eval_Reward/Mean_Step20', args.terminate, smooth=5)
    vel_mt_x, vel_mt_y, vel_mt_std = extract(args.mt_vel_path, 'FT_Eval_Reward/Mean_Step20', args.terminate, smooth=5)
    ant_mt_x, ant_mt_y, ant_mt_std = extract(args.mt_ant_path, 'FT_Eval_Reward/Mean_Step20', args.terminate, smooth=5)

    dir_td3_x, dir_td3_y, dir_td3_std = extract(args.td3_dir_path, 'Eval_Reward/Average', args.terminate, smooth=20)
    vel_td3_x, vel_td3_y, vel_td3_std = extract(args.td3_vel_path, 'Eval_Reward/Average', args.terminate, smooth=20)
    walker_td3_x, walker_td3_y, walker_td3_std = extract(args.td3_walker_path, 'Eval_Reward/Average', args.terminate, smooth=20)
    ant_td3_x, ant_td3_y, ant_td3_std = extract(args.td3_ant_path, 'Eval_Reward/Average', args.terminate, smooth=20)
    
    dir_macaw_x, dir_macaw_y, dir_macaw_std = extract(args.macaw_dir_path, 'Eval_Reward/Mean', args.terminate, smooth=20)
    vel_macaw_x, vel_macaw_y, vel_macaw_std = extract(args.macaw_vel_path, 'Eval_Reward/Mean', args.terminate, smooth=20)
    walker_macaw_x, walker_macaw_y, walker_macaw_std = extract(args.macaw_walker_path, 'Eval_Reward/Mean', args.terminate, smooth=20)
    ant_macaw_x, ant_macaw_y, ant_macaw_std = extract(args.macaw_ant_path, 'Eval_Reward/Mean', args.terminate, smooth=20)

    dir_pearl_x, dir_pearl_y, dir_pearl_std = extract(args.pearl_dir_path, 'test_tasks_mean_reward/mean_return', args.terminate, 2000, 3)
    vel_pearl_x, vel_pearl_y, vel_pearl_std = extract(args.pearl_vel_path, 'test_tasks_mean_reward/mean_return', args.terminate, 2000, 3)
    walker_pearl_x, walker_pearl_y, walker_pearl_std = extract(args.pearl_walker_path, 'test_tasks_mean_reward/mean_return', args.terminate, 4000, 3)
    ant_pearl_x, ant_pearl_y, ant_pearl_std = extract(args.pearl_ant_path, 'test_tasks_mean_reward/mean_return', args.terminate, 4000, 3)

    
    w = 5.4*3
    h = w/2
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(w,h))

    color1 = next(axes[0,0]._get_lines.prop_cycler)['color']
    color2 = next(axes[0,0]._get_lines.prop_cycler)['color']
    color3 = next(axes[0,0]._get_lines.prop_cycler)['color']
    color3 = next(axes[0,0]._get_lines.prop_cycler)['color']
    color4 = next(axes[0,0]._get_lines.prop_cycler)['color']

    axes[0,0].plot(dir_macaw_x, dir_macaw_y, color=color1, linewidth=2)
    axes[0,0].fill_between(dir_macaw_x, dir_macaw_y-dir_macaw_std, dir_macaw_y + dir_macaw_std, color=color1, alpha = 0.5)
    #axes[0,0].plot([dir_macaw_x[-1], dir_pearl_x[-1]], [dir_macaw_y[-1]] * 2, color=color1, linewidth=2)
    axes[0,0].plot(dir_mt_x, dir_mt_y, linewidth=2, color=color2)
    axes[0,0].fill_between(dir_mt_x, dir_mt_y-dir_mt_std, dir_mt_y + dir_mt_std, color=color2, alpha = 0.5)
    #axes[0,0].plot([dir_mt_x[-1], dir_pearl_x[-1]], [dir_mt_y[-1]] * 2, color=color2, linewidth=2)
    axes[0,0].plot(dir_pearl_x, dir_pearl_y, linewidth=2, color=color3)
    axes[0,0].fill_between(dir_pearl_x, dir_pearl_y-dir_pearl_std, dir_pearl_y + dir_pearl_std, color=color3, alpha = 0.5)
    axes[0,0].plot(dir_td3_x, dir_td3_y, linewidth=2, color=color4)
    axes[0,0].fill_between(dir_td3_x, dir_td3_y-dir_td3_std, dir_td3_y + dir_td3_std, color=color4, alpha = 0.5)
    axes[0,0].set_title('Cheetah-Direction')
    axes[0,0].set_xlabel('Training Steps')
    axes[0,0].set_ylabel('Reward')
    
    axes[0,1].plot(vel_macaw_x, vel_macaw_y, linewidth=2, color=color1)
    axes[0,1].fill_between(vel_macaw_x, vel_macaw_y-vel_macaw_std, vel_macaw_y + vel_macaw_std, color=color1, alpha = 0.5)
    #axes[0,1].plot([vel_macaw_x[-1], vel_pearl_x[-1]], [vel_macaw_y[-1]] * 2, color=color1, linewidth=2)
    axes[0,1].plot(vel_mt_x, vel_mt_y, linewidth=2, color=color2)
    axes[0,1].fill_between(vel_mt_x, vel_mt_y-vel_mt_std, vel_mt_y + vel_mt_std, color=color2, alpha = 0.5)
    #axes[0,1].plot([vel_mt_x[-1], vel_pearl_x[-1]], [vel_mt_y[-1]] * 2, color=color2, linewidth=2)
    axes[0,1].plot(vel_pearl_x, vel_pearl_y, linewidth=2, color=color3)
    axes[0,1].fill_between(vel_pearl_x, vel_pearl_y-vel_pearl_std, vel_pearl_y + vel_pearl_std, color=color3, alpha = 0.5)
    axes[0,1].plot(vel_td3_x, vel_td3_y, linewidth=2, color=color4)
    axes[0,1].fill_between(vel_td3_x, vel_td3_y-vel_td3_std, vel_td3_y + vel_td3_std, color=color4, alpha = 0.5)
    axes[0,1].set_title('Cheetah-Velocity')
    axes[0,1].set_xlabel('Training Steps')
    axes[0,1].set_ylabel('Reward')

    axes[1,0].plot(walker_macaw_x, walker_macaw_y, linewidth=2, label='MACAW', color=color1)
    axes[1,0].fill_between(walker_macaw_x, walker_macaw_y-walker_macaw_std, walker_macaw_y + walker_macaw_std, color=color1, alpha = 0.5)
    #axes[1,0].plot([walker_macaw_x[-1], walker_pearl_x[-1]], [walker_macaw_y[-1]] * 2, linewidth=2, color=color1)
    axes[1,0].plot(walker_mt_x, walker_mt_y, linewidth=2, label='MT + fine tune', color=color2)
    axes[1,0].fill_between(walker_mt_x, walker_mt_y-walker_mt_std, walker_mt_y + walker_mt_std, color=color2, alpha = 0.5)
    #axes[1,0].plot([walker_mt_x[-1], walker_pearl_x[-1]], [walker_mt_y[-1]] * 2, linewidth=2, color=color2)
    axes[1,0].plot(walker_pearl_x, walker_pearl_y, linewidth=2, label='PEARL', color=color3)
    axes[1,0].fill_between(walker_pearl_x, walker_pearl_y-walker_pearl_std, walker_pearl_y + walker_pearl_std, color=color3, alpha = 0.5)
    axes[1,0].plot(walker_td3_x, walker_td3_y, linewidth=2, color=color4)
    axes[1,0].fill_between(walker_td3_x, walker_td3_y-walker_td3_std, walker_td3_y + walker_td3_std, color=color4, alpha = 0.5)
    axes[1,0].set_title('Walker-Params')
    axes[1,0].set_xlabel('Training Steps')
    axes[1,0].set_ylabel('Reward')
    #axes[1,0].legend(loc=2)
    
    axes[1,1].plot(ant_macaw_x, ant_macaw_y, linewidth=2, label='MACAW', color=color1)
    axes[1,1].fill_between(ant_macaw_x, ant_macaw_y-ant_macaw_std, ant_macaw_y + ant_macaw_std, color=color1, alpha = 0.5)
    #axes[1,1].plot([ant_macaw_x[-1], ant_pearl_x[-1]], [ant_macaw_y[-1]] * 2, linewidth=2, color=color1)
    axes[1,1].plot(ant_mt_x, ant_mt_y, linewidth=2, label='MT + fine tune', color=color2)
    axes[1,1].fill_between(ant_mt_x, ant_mt_y-ant_mt_std, ant_mt_y + ant_mt_std, color=color2, alpha = 0.5)
    #axes[1,1].plot([ant_mt_x[-1], ant_pearl_x[-1]], [ant_mt_y[-1]] * 2, linewidth=2, color=color2)
    axes[1,1].plot(ant_pearl_x, ant_pearl_y, linewidth=2, label='PEARL', color=color3)
    axes[1,1].fill_between(ant_pearl_x, ant_pearl_y-ant_pearl_std, ant_pearl_y + ant_pearl_std, color=color3, alpha = 0.5)
    axes[1,1].plot(ant_td3_x, ant_td3_y, linewidth=2, label='TD3 + Context', color=color4)
    axes[1,1].fill_between(ant_td3_x, ant_td3_y-ant_td3_std, ant_td3_y + ant_td3_std, color=color4, alpha = 0.5)
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
    fig.savefig('rebuttal_figs/' + args.name)


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
    parser.add_argument('--td3_dir_path', type=str)
    parser.add_argument('--td3_vel_path', type=str)
    parser.add_argument('--td3_walker_path', type=str)
    parser.add_argument('--td3_ant_path', type=str)
    parser.add_argument('--terminate', type=int, default=None)
    parser.add_argument('--name', type=str, default='mujoco_neurips')
    run(parser.parse_args())
