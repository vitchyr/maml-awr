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
plt.rc('figure', titlesize=MEDIUM_SIZE)  # fontsize of the figure title


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


def extract_macaw(path, terminate: int = None, tag_='eval_acc/outer', sigma=10):
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
                    if tag != tag_:
                        continue
                    #print(tag, step, value)
                    y.append(value)
                    x.append(step)
            except Exception as e:
                print(entry)
                raise e
    except Exception as e:
        print(e)

    y = gaussian_filter1d(y, sigma=sigma)
    return np.array(x).astype(np.float32) / 1000, np.array(y)


def trim(x, y, val):
    v = np.where(np.squeeze(x) > val)[0]
    if len(v) > 0:
        v = v[0]
        return x[:v], y[:v]
    else:
        return x, y

def run(args: argparse.Namespace):
    wlinear_x, wlinear_y = extract_macaw(args.wlinear_path, args.terminate)
    linear_width_x, linear_width_y = extract_macaw(args.linear_width_path, args.terminate)
    linear_params_x, linear_params_y = extract_macaw(args.linear_params_path, args.terminate)
    tr_wlinear_x, tr_wlinear_y = extract_macaw(args.wlinear_path, args.terminate, tag_='train_acc/outer', sigma=3)
    tr_linear_width_x, tr_linear_width_y = extract_macaw(args.linear_width_path, args.terminate, tag_='train_acc/outer', sigma=3)
    tr_linear_params_x, tr_linear_params_y = extract_macaw(args.linear_params_path, args.terminate, tag_='train_acc/outer', sigma=3)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9,6))
    axes.tick_params(axis=u'both', which=u'both',length=0)
    axes.grid(linestyle='--', linewidth=1)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)

    _ = next(axes._get_lines.prop_cycler)['color']
    color1 = next(axes._get_lines.prop_cycler)['color']
    _ = next(axes._get_lines.prop_cycler)['color']
    _ = next(axes._get_lines.prop_cycler)['color']
    color2 = next(axes._get_lines.prop_cycler)['color']
    color3 = next(axes._get_lines.prop_cycler)['color']

    l1, = axes.plot(tr_wlinear_x, tr_wlinear_y, color=color1, linewidth=3, linestyle=':')
    axes.plot(tr_linear_params_x, tr_linear_params_y, color=color3, linewidth=3, linestyle=':')
    axes.plot(tr_linear_width_x, tr_linear_width_y, color=color2, linewidth=3, linestyle=':')
    l2, = axes.plot(wlinear_x, wlinear_y, color=color1, linewidth=3, label='Weight Transform')
    axes.plot(linear_params_x, linear_params_y, color=color3, linewidth=3, label='No WT-Equal Params')
    axes.plot(linear_width_x, linear_width_y, color=color2, linewidth=3, label='No WT-Equal Width')

    axes.set_xlabel('Training Steps (thousands)')
    axes.set_ylabel('Accuracy')
    leg1 = axes.legend(loc='lower right')
    leg2 = axes.legend([l1, l2], ['Train', 'Val'], loc='center right')
    leg2.legendHandles[0].set_color('black')
    leg2.legendHandles[1].set_color('black')
    plt.gca().add_artist(leg1)
    plt.suptitle('Omniglot Weight Transform Ablation')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(args.name, bbox_inches='tight')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--wlinear_path', type=str, default='env/.guild/runs/08d5665d41ef46ab9ddd5bd782df88fe/.guild/')
    parser.add_argument('--linear_params_path', type=str, default='env/.guild/runs/d17be653a7194223910b15e1513308d7/.guild/')
    parser.add_argument('--linear_width_path', type=str, default='env/.guild/runs/eb8c7a878c704e6c8c99f2f53e28112c/.guild/')
    parser.add_argument('--terminate', type=int, default=None)
    parser.add_argument('--name', type=str, default='arch')
    run(parser.parse_args())
