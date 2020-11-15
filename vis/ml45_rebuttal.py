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
import os
import glob


SMALL_SIZE = 16
MEDIUM_SIZE = 17
BIGGER_SIZE = 24

plt.rc('font', size=MEDIUM_SIZE)          # controls default text sizes
#plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
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


def extract_macaw(path, terminate: int = None, prefix: str = None, suffix: str = None, xscale = None, smooth=1, n_vals=250):
    paths = glob.glob(path)

    xs, ys = [], []
    d = None
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
        y = []
        x = []
        try:
            for entry in summary_iterator(fn):
                try:
                    if len(entry.summary.value):
                        v = entry.summary.value[0]
                        step, tag, value = entry.step, v.tag, v.simple_value
                        if terminate and step > terminate:
                            break
                        if prefix is None:
                            if tag != 'Eval_Reward/Mean' and tag != 'test_tasks_mean_reward/mean_return':
                                continue
                            if xscale is not None:
                                step *= xscale
                            y.append(value)
                            x.append(step)
                        else:
                            if not tag.startswith(prefix) or (suffix is not None and not tag.endswith(suffix)):
                                continue
                            if d is None:
                                d = defaultdict(list)
                            if xscale is not None:
                                step *= xscale
                            d[step].append(value)
                except Exception as e:
                    print(entry)
                    raise e
        except Exception as e:
            print('exception', e)
        #xs.append(x)
        #ys.append(running_mean(y, smooth))

    x = np.sort(list(d.keys()))
    y = np.array([np.mean(d[x_]) for x_ in x])
    ystd = np.array([np.std(d[x_]) for x_ in x])
    #ylens = [len(z) for z in ys]

    #ys = [y_[:min(ylens)] for y_ in ys]
    #xs = [x_[:min(ylens)] for x_ in xs]
    ymean = running_mean(y, smooth)
    #ys = np.array(ys)
    #xs = np.array(xs)
    x = x[:ymean.shape[0]]
    ystd = ystd[:ymean.shape[0]]
    logx = np.exp((np.linspace(0,np.log(x.max())*np.log(10), n_vals)[:,None]))
    logx = logx/(logx.max() / x.max())
    idxs = np.argmin(np.abs(x[None,:] - logx), 1)
    idxs = np.unique(idxs)
    newx = x[idxs]
    #print(newx)
    newymean = ymean[idxs]
    newystd = ystd[idxs]
    print(x.shape, ymean.shape, ystd.shape)
    return newx, newymean, newystd * 0.1


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def extract(path, tag_: str, terminate: int = None, xscale=1, smooth=1):
    paths = glob.glob(path)

    xs, ys = [], []
    print(paths)
    for path in paths:
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

    x = x[:ymean.shape[0]]
    print(x.shape, ymean.shape, ystd.shape)
    return x, ymean, ystd


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
    #mt_x, mt_success, mt_std = extract_macaw(args.mt_path, args.terminate, prefix='FT_Eval_Success', suffix='Step4', smooth=10)
    mt2_x, mt2_success, mt2_std = extract_macaw(args.mt_path, args.terminate, prefix='FT_Eval_Success', suffix='Step19', smooth=10)

    macaw_x, macaw_success, macaw_std = extract_macaw(args.macaw_path, args.terminate, smooth=10, prefix='Eval_Success')
    #macaw_success = cumavg(macaw_success)

    pearl_x, pearl_success, pearl_std = extract_macaw(args.pearl_path, args.terminate, prefix='test_tasks_mean_succes/mean_succes', smooth=1, xscale=200)
    #pearl_success = cumavg(pearl_success)

    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(9,6))
    #axes2 = axes.twinx()
    axes.tick_params(axis=u'both', which=u'both',length=0)
    axes.grid(linestyle='--', linewidth=0.75)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.spines['bottom'].set_visible(False)
    axes.spines['left'].set_visible(False)
    #axes2.tick_params(axis=u'both', which=u'both',length=0)
    #axes2.spines['top'].set_visible(False)
    #axes2.spines['right'].set_visible(False)
    #axes2.spines['bottom'].set_visible(False)
    #axes2.spines['left'].set_visible(False)
    print(macaw_success.shape, macaw_std.shape)
    color1 = next(axes._get_lines.prop_cycler)['color']
    color2 = next(axes._get_lines.prop_cycler)['color']
    color3 = next(axes._get_lines.prop_cycler)['color']
    color3 = next(axes._get_lines.prop_cycler)['color']

    l1, = axes.plot(macaw_x, macaw_success, linewidth=3, color=color1, label='MACAW')
    #l1, = axes.plot([macaw_x[-1], pearl_x[-1]], [macaw_success[-1]] * 2, linewidth=3, color=color1)
    axes.fill_between(macaw_x, macaw_success-macaw_std, macaw_success + macaw_std, color=color1, alpha = 0.5)
    #l2, = axes.plot(macaw_train_x, macaw_success_train, '--', linewidth=3, color=color)

    axes.plot(pearl_x, pearl_success, linewidth=3, color=color3, label='PEARL')
    axes.fill_between(pearl_x, pearl_success-pearl_std, pearl_success + pearl_std, color=color3, alpha = 0.5)
    #axes.plot(pearl_train_x, pearl_success_train,  '--', linewidth=3, color=color)

    axes.plot(mt2_x, mt2_success, linewidth=3, color=color2, label='MT + fine tune')
    #axes.plot([mt2_x[-1], pearl_x[-1]], [mt2_success[-1]] * 2, linewidth=3, color=color2)
    axes.fill_between(mt2_x, mt2_success-mt2_std, mt2_success + mt2_std, color=color2, alpha = 0.5)

    #axes.plot(mt_x, mt_success, linewidth=3, color=color, label='MT + fine tune (5)')
    #axes.plot([mt_x[-1], pearl_x[-1]], [mt_success[-1]] * 2, linewidth=3, color=color2)
    #axes.fill_between(mt_x, mt_success-mt_std, mt_success + mt_std, color=color, alpha = 0.5)
    #axes.plot(mt_train_x, mt_success_train,  '--', linewidth=3, color=color)
    axes.set_xscale('log')
    axes.set_title('Meta-World ML45 Benchmark')
    axes.set_xlabel('Training Steps')
    #axes.set_ylabel('Reward')
    axes.set_xlim(900,None)
    axes.set_ylabel('Average Test Success Rate')
    leg = axes.legend(loc='center left', bbox_to_anchor=(0,0.63))

    plt.tight_layout()
    fig.savefig('rebuttal_figs/' + args.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--macaw_path', type=str)
    parser.add_argument('--pearl_path', type=str)
    parser.add_argument('--mt_path', type=str)
    parser.add_argument('--terminate', type=int, default=None)
    parser.add_argument('--name', type=str, default='ML45')
    run(parser.parse_args())
