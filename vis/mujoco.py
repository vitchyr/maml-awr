import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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


def extract_macaw(path, prefix: str = None, terminate: int = None):
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

    y = gaussian_filter1d(y, sigma=2)        
    return np.array(x).astype(np.float32), np.array(y)


def trim(x, y, val):
    v = np.where(np.squeeze(x) > val)[0]
    if len(v) > 0:
        v = v[0]
        return x[:v], y[:v]
    else:
        return x, y

def run(args: argparse.Namespace):
    dir_macaw_x, dir_macaw_y = extract_macaw('/iris/u/em7/code/maml-rawr/log/gm/cheetah_dir_macaw_gpu/tb/events.out.tfevents.1581050697.iris-ws-3.stanford.edu.15061.0', args.terminate)
    vel_macaw_x, vel_macaw_y = extract_macaw('/iris/u/em7/code/maml-rawr/log/gm/cheetah_vel_macaw/tb/events.out.tfevents.1581050839.iris-hp-z8.stanford.edu.7367.0', args.terminate)
    walker_macaw_x, walker_macaw_y = extract_macaw('/iris/u/em7/code/maml-rawr/log/gm/walker_params_macaw_fresh/tb/events.out.tfevents.1581298618.iris-ws-3.stanford.edu.2805.0', args.terminate)

    dir_mt_x, dir_mt_y = extract_macaw('/iris/u/em7/code/maml-rawr/log/ablations2/cheetah_dir_mt5/tb/events.out.tfevents.1581412839.iris-hp-z8.stanford.edu.15806.0', args.terminate)
    vel_mt_x, vel_mt_y = extract_macaw('/iris/u/em7/code/maml-rawr/log/ablations2/cheetah_vel_mt_ood3/tb/events.out.tfevents.1581412103.iris1.stanford.edu.35232.0', args.terminate)
    walker_mt_x, walker_mt_y = extract_macaw('/iris/u/em7/code/maml-rawr/log/ablations2/walker_param_mt2/tb/events.out.tfevents.1581412377.iris2.stanford.edu.15556.0', args.terminate)
    
    with open('/iris/u/rafailov/pearl_logs/tensorboard/cheetah_vel.pkl', 'rb') as f:
        vel_pearl = pickle.load(f)
    with open('/iris/u/rafailov/pearl_logs/tensorboard/cheetah_dir.pkl', 'rb') as f:
        dir_pearl = pickle.load(f)
    with open('/iris/u/rafailov/pearl_logs/tensorboard/walker_param.pkl', 'rb') as f:
        walker_pearl = pickle.load(f)

    dir_pearl_y = np.array(dir_pearl['test_task_2'] + dir_pearl['test_task_3']) / 2
    dir_pearl_y = gaussian_filter1d(dir_pearl_y, sigma=2)        

    vel_pearl_y = np.array([vel_pearl[k] for k in vel_pearl.keys() if 'return' in k]).mean(0)
    vel_pearl_y = gaussian_filter1d(vel_pearl_y, sigma=2)        

    walker_pearl_y = np.array([walker_pearl[k] for k in walker_pearl.keys() if 'return' in k]).mean(0)
    walker_pearl_y = gaussian_filter1d(walker_pearl_y, sigma=2)        
    
    dir_pearl_x = (dir_pearl['test_task_2_step'] * 100).astype(np.float32)
    vel_pearl_x = np.arange(len(vel_pearl_y)).astype(np.float32) * 100
    walker_pearl_x = np.arange(len(walker_pearl_y)).astype(np.float32) * 500

    dir_pearl_x, dir_pearl_y = trim(dir_pearl_x, dir_pearl_y, args.terminate)
    vel_pearl_x, vel_pearl_y = trim(vel_pearl_x, vel_pearl_y, args.terminate)
    walker_pearl_x, walker_pearl_y = trim(walker_pearl_x, walker_pearl_y, args.terminate)
    dir_macaw_x, dir_macaw_y = trim(dir_macaw_x, dir_macaw_y, args.terminate)
    vel_macaw_x, vel_macaw_y = trim(vel_macaw_x, vel_macaw_y, args.terminate)
    walker_macaw_x, walker_macaw_y = trim(walker_macaw_x, walker_macaw_y, args.terminate)
    dir_mt_x, dir_mt_y = trim(dir_mt_x, dir_mt_y, args.terminate)
    vel_mt_x, vel_mt_y = trim(vel_mt_x, vel_mt_y, args.terminate)
    walker_mt_x, walker_mt_y = trim(walker_mt_x, walker_mt_y, args.terminate)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(21,4.75))

    xunit = 1
    dir_macaw_x /= xunit
    dir_mt_x /= xunit
    dir_pearl_x /= xunit
    vel_macaw_x /= xunit
    vel_mt_x /= xunit
    vel_pearl_x /= xunit
    walker_macaw_x /= xunit
    walker_mt_x /= xunit
    walker_pearl_x /= xunit

    axes[0].plot(dir_macaw_x, dir_macaw_y, label='MACAW')
    axes[0].plot(dir_mt_x, dir_mt_y, '--', label='MT + finetune')
    color = next(axes[0]._get_lines.prop_cycler)['color']
    axes[0].plot(dir_pearl_x, dir_pearl_y, '-.', label='PEARL')
    axes[0].set_xscale('log')
    axes[0].set_title('Cheetah-Direction')
    axes[0].set_xlabel('Number of gradient steps')
    axes[0].set_ylabel('Reward')

    axes[1].plot(vel_macaw_x, vel_macaw_y)
    axes[1].plot(vel_mt_x, vel_mt_y, '--')
    color = next(axes[1]._get_lines.prop_cycler)['color']
    axes[1].plot(vel_pearl_x, vel_pearl_y, '-.')
    axes[1].set_xscale('log')
    axes[1].set_title('Cheetah-Velocity')
    axes[1].set_xlabel('Number of gradient steps')
    axes[1].set_ylabel('Reward')

    axes[2].plot(walker_macaw_x, walker_macaw_y, label='MACAW (ours)')
    axes[2].plot(walker_mt_x, walker_mt_y, '--', label='MT + finetune')
    color = next(axes[2]._get_lines.prop_cycler)['color']
    axes[2].plot(walker_pearl_x, walker_pearl_y, '-.', label='PEARL')
    axes[2].set_xscale('log')
    axes[2].set_title('Walker Params')
    axes[2].legend(loc=2)
    axes[2].set_xlabel('Number of gradient steps')
    axes[2].set_ylabel('Reward')
    plt.tight_layout()
    fig.savefig(args.name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--terminate', type=int, default=None)
    parser.add_argument('--name', type=str)
    run(parser.parse_args())
