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
    try:
        for entry in summary_iterator(path):
            try:
                if len(entry.summary.value):
                    v = entry.summary.value[0]
                    step, tag, value = entry.step, v.tag, v.simple_value
                    if terminate and step > terminate:
                        break
                    if tag != 'Eval_Reward/Mean':
                        continue
                    #print(tag, step, value)
                    y.append(value)
                    x.append(step)
            except Exception as e:
                print(entry)
                raise e
    except Exception as e:
        print(e)

    y = gaussian_filter1d(y, sigma=5)        
    return np.array(x).astype(np.float32) / 1000, np.array(y)


def trim(x, y, val):
    v = np.where(np.squeeze(x) > val)[0]
    if len(v) > 0:
        v = v[0]
        return x[:v], y[:v]
    else:
        return x, y

def run(args: argparse.Namespace):
    macaw_dir_x, macaw_dir_y = extract_macaw(args.dir_path, args.terminate)
    macaw_vel_x, macaw_vel_y = extract_macaw(args.vel_path, args.terminate)
    macaw_walker_x, macaw_walker_y = extract_macaw(args.walker_path, args.terminate)
    macaw_ant_x, macaw_ant_y = extract_macaw(args.ant_path, args.terminate)

    maml_dir_x, maml_dir_y = extract_macaw(args.maml_dir_path, args.terminate)
    maml_vel_x, maml_vel_y = extract_macaw(args.maml_vel_path, args.terminate)
    maml_walker_x, maml_walker_y = extract_macaw(args.maml_walker_path, args.terminate)
    maml_ant_x, maml_ant_y = extract_macaw(args.maml_ant_path, args.terminate)

    wlinear_dir_x, wlinear_dir_y = extract_macaw(args.wlinear_dir_path, args.terminate)
    wlinear_vel_x, wlinear_vel_y = extract_macaw(args.wlinear_vel_path, args.terminate)
    wlinear_walker_x, wlinear_walker_y = extract_macaw(args.wlinear_walker_path, args.terminate)
    wlinear_ant_x, wlinear_ant_y = extract_macaw(args.wlinear_ant_path, args.terminate)

    w = 5.4*3
    h = w/2
    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(w,h))
    for ax in axes:
        for a in ax:
            a.tick_params(axis=u'both', which=u'both',length=0)
            a.grid(linestyle='--', linewidth=0.75)
            a.spines['top'].set_visible(False)
            a.spines['right'].set_visible(False)
            a.spines['bottom'].set_visible(False)
            a.spines['left'].set_visible(False)

    color1 = next(axes[0,0]._get_lines.prop_cycler)['color']
    color2 = next(axes[0,0]._get_lines.prop_cycler)['color']
    color2 = next(axes[0,0]._get_lines.prop_cycler)['color']
    color3 = next(axes[0,0]._get_lines.prop_cycler)['color']

    axes[0,0].plot(macaw_dir_x, macaw_dir_y, color=color1, label='MACAW')
    axes[0,0].plot(maml_dir_x, maml_dir_y, color=color2, label='No Adv Head')
    axes[0,0].plot(wlinear_dir_x, wlinear_dir_y, color=color3, label='No Weight Transf.')

    axes[0,1].plot(macaw_vel_x, macaw_vel_y, color=color1, label='MACAW')
    axes[0,1].plot(maml_vel_x, maml_vel_y, color=color2, label='No Adv Head')
    axes[0,1].plot(wlinear_vel_x, wlinear_vel_y, color=color3, label='No Weight Transf.')

    axes[1,0].plot(macaw_walker_x, macaw_walker_y, color=color1, label='MACAW')
    axes[1,0].plot(maml_walker_x, maml_walker_y, color=color2, label='No Adv Head')
    axes[1,0].plot(wlinear_walker_x, wlinear_walker_y, color=color3, label='No Weight Transf.')

    axes[1,1].plot(macaw_ant_x, macaw_ant_y, color=color1, label='MACAW')
    axes[1,1].plot(maml_ant_x, maml_ant_y, color=color2, label='No Adv Head')
    axes[1,1].plot(wlinear_ant_x, wlinear_ant_y, color=color3, label='No Weight Transf.')

    axes[0,0].set_title('Cheetah-Direction')
    axes[0,1].set_title('Cheetah-Velocity')
    axes[1,0].set_title('Walker-Params')
    axes[1,1].set_title('Ant-Direction')
    for ax in axes:
        for a in ax:
            a.set_xlabel('Training Steps (thousands)')
            a.set_ylabel('Reward')
    axes[0,0].legend(loc=4)
    plt.suptitle('Ablating Architectural Components of MACAW')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(args.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str)
    parser.add_argument('--vel_path', type=str)
    parser.add_argument('--walker_path', type=str)
    parser.add_argument('--ant_path', type=str)
    parser.add_argument('--maml_dir_path', type=str)
    parser.add_argument('--maml_vel_path', type=str)
    parser.add_argument('--maml_walker_path', type=str)
    parser.add_argument('--maml_ant_path', type=str)
    parser.add_argument('--wlinear_dir_path', type=str)
    parser.add_argument('--wlinear_vel_path', type=str)
    parser.add_argument('--wlinear_walker_path', type=str)
    parser.add_argument('--wlinear_ant_path', type=str)
    parser.add_argument('--terminate', type=int, default=None)
    parser.add_argument('--name', type=str, default='arch')
    run(parser.parse_args())
