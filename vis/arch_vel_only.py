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
import os


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

'''
SMALL_SIZE = 20
MEDIUM_SIZE = 22
BIGGER_SIZE = 28

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
'''


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
    files = [f for f in os.listdir(path) if 'events' in f]
    path = f'{path}/{files[0]}'
    print(path)
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
    #macaw_dir_x, macaw_dir_y = extract_macaw(args.dir_path, args.terminate)
    macaw_vel_x, macaw_vel_y = extract_macaw(args.vel_path, args.terminate)
    #macaw_walker_x, macaw_walker_y = extract_macaw(args.walker_path, args.terminate)
    #macaw_ant_x, macaw_ant_y = extract_macaw(args.ant_path, args.terminate)

    #maml_dir_x, maml_dir_y = extract_macaw(args.maml_dir_path, args.terminate)
    #maml_vel_x, maml_vel_y = extract_macaw(args.maml_vel_path, args.terminate)
    #maml_walker_x, maml_walker_y = extract_macaw(args.maml_walker_path, args.terminate)
    #maml_ant_x, maml_ant_y = extract_macaw(args.maml_ant_path, args.terminate)

    #wlinear_dir_x, wlinear_dir_y = extract_macaw(args.wlinear_dir_path, args.terminate)
    wlinear_vel_x, wlinear_vel_y = extract_macaw(args.wlinear_vel_path, args.terminate)
    #wlinear_walker_x, wlinear_walker_y = extract_macaw(args.wlinear_walker_path, args.terminate)
    #wlinear_ant_x, wlinear_ant_y = extract_macaw(args.wlinear_ant_path, args.terminate)

    #w = 5.4*3
    #h = w/2
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9,6))
    a= axes
    a.tick_params(axis=u'both', which=u'both',length=0)
    a.grid(linestyle='--', linewidth=1.25)
    a.spines['top'].set_visible(False)
    a.spines['right'].set_visible(False)
    a.spines['bottom'].set_visible(False)
    a.spines['left'].set_visible(False)

    color1 = next(axes._get_lines.prop_cycler)['color']
    color2 = next(axes._get_lines.prop_cycler)['color']
    color2 = next(axes._get_lines.prop_cycler)['color']
    color3 = next(axes._get_lines.prop_cycler)['color']

    axes.plot(macaw_vel_x, macaw_vel_y, color=color1, linewidth=5, label='MACAW')
    axes.plot(wlinear_vel_x, wlinear_vel_y, '--', color=color1, linewidth=5, label='No Weight Transf.')

    a.set_xlabel('Training Steps (thousands)')
    a.set_ylabel('Reward')
    axes.legend(loc=4)
    plt.suptitle('Ablating MACAW\'s Weight Transform')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(args.name, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir_path', type=str)
    parser.add_argument('--vel_path', type=str, default='log/NeurIPS_multiseed/macaw_vel/tb')
    parser.add_argument('--walker_path', type=str)
    parser.add_argument('--ant_path', type=str)
    parser.add_argument('--maml_dir_path', type=str)
    parser.add_argument('--maml_vel_path', type=str)
    parser.add_argument('--maml_walker_path', type=str)
    parser.add_argument('--maml_ant_path', type=str)
    parser.add_argument('--wlinear_dir_path', type=str)
    parser.add_argument('--wlinear_vel_path', type=str, default='log/NeurIPS3/macaw_vel_nowlinear/tb')
    parser.add_argument('--wlinear_walker_path', type=str)
    parser.add_argument('--wlinear_ant_path', type=str)
    parser.add_argument('--terminate', type=int, default=None)
    parser.add_argument('--name', type=str, default='arch')
    run(parser.parse_args())
