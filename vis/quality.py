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
    macaw_start_x, macaw_start_y = extract_macaw(args.start_path, args.terminate)
    macaw_middle_x, macaw_middle_y = extract_macaw(args.middle_path, args.terminate)
    macaw_end_x, macaw_end_y = extract_macaw(args.end_path, args.terminate)

    maml_start_x, maml_start_y = extract_macaw(args.maml_start_path, args.terminate)
    maml_middle_x, maml_middle_y = extract_macaw(args.maml_middle_path, args.terminate)
    maml_end_x, maml_end_y = extract_macaw(args.maml_end_path, args.terminate)
    
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9,6))
    axes.tick_params(axis=u'both', which=u'both',length=0)
    axes.grid(linestyle='--', linewidth=1.25)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    
    color = next(axes._get_lines.prop_cycler)['color']
    color = next(axes._get_lines.prop_cycler)['color']
    l1, = axes.plot(macaw_end_x, macaw_end_y, color=color, linewidth=4, label='Good Data')
    l2, = axes.plot(maml_end_x, maml_end_y, '--', color=color, linewidth=4)
    color = next(axes._get_lines.prop_cycler)['color']
    color = next(axes._get_lines.prop_cycler)['color']
    color = next(axes._get_lines.prop_cycler)['color']
    axes.plot(macaw_middle_x, macaw_middle_y, color=color, linewidth=4, label='Medium Data')
    axes.plot(maml_middle_x, maml_middle_y,  '--', color=color, linewidth=4)
    color = next(axes._get_lines.prop_cycler)['color']
    axes.plot(macaw_start_x, macaw_start_y, color=color, linewidth=4, label='Bad Data')
    axes.plot(maml_start_x, maml_start_y,  '--', color=color, linewidth=4)
    axes.set_title('Ablating MACAW\'s Enriched Policy Update')
    axes.set_xlabel('Training Steps (thousands)')
    axes.set_ylabel('Reward')
    leg1 = axes.legend(loc=4, bbox_to_anchor=(0,-0.02,1.05,1.0))
    leg2 = axes.legend([l1, l2], ['MACAW', 'MAML+AWR'], loc='lower center', bbox_to_anchor=(0,-0.02,0.78,1))
    leg2.legendHandles[0].set_color('black')
    leg2.legendHandles[1].set_color('black')
    plt.gca().add_artist(leg1)
    plt.ylim([-270, -24])
    plt.tight_layout()
    fig.savefig(args.name, bbox_inches = "tight")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--start_path', type=str, default="log/NeurIPS3/macaw_vel_start.3/tb")
    parser.add_argument('--middle_path', type=str, default="log/NeurIPS3/macaw_vel_middle.3/tb")
    parser.add_argument('--end_path', type=str, default="log/NeurIPS3/macaw_vel_end.3/tb")
    parser.add_argument('--maml_start_path', type=str, default="log/NeurIPS3/mamlawr_vel_start/tb")
    parser.add_argument('--maml_middle_path', type=str, default="log/NeurIPS3/mamlawr_vel_middle/tb")
    parser.add_argument('--maml_end_path', type=str, default="log/NeurIPS3/mamlawr_vel_end/tb")
    parser.add_argument('--terminate', type=int, default=None)
    parser.add_argument('--name', type=str, default='adv_regression_ablation')
    run(parser.parse_args())
