import click
from pathlib import Path

from doodad.wrappers.easy_launch import sweep_function
from src.launcher import run_doodad_experiment


@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--suffix', default=None)
@click.option('--nseeds', default=1)
@click.option('--mode', default='local')
def main(debug, suffix, nseeds, mode):
    base_dir = Path(__file__).parent.parent

    path_parts = __file__.split('/')
    suffix = '' if suffix is None else '--{}'.format(suffix)
    exp_name = 'macaw-{}{}'.format(
        path_parts[-1].split('.')[0],
        suffix,
    )

    if debug:
        exp_name = 'dev--' + exp_name
        mode = 'local'
        nseeds = 1

    params = {
        'env': ['ant_dir'],
        'use_rlkit': [
            True,
        ],
        'seed': list(range(nseeds)),
    }
    default_params = {
        'buffer_path_template':'/preloaded_buffer/ant_dir_32/macaw_buffer_iter50/macaw_buffer_task_{}.npy',
        'saved_tasks_path': '/preloaded_buffer/ant_dir_32/macaw_buffer_iter50/tasks.pkl',
        'train_task_idxs': [0, 1, 2, 3],
        'test_task_idxs': [4, 5, 6, 7],
    }

    if mode == 'local':
        remote_mount_configs = [
            dict(
                local_dir='/home/vitchyr/mnt2/log2/demos/',
                mount_point='/preloaded_buffer',
            ),
        ]
    elif mode == 'azure':
        remote_mount_configs = [
            dict(
                local_dir='/doodad_tmp/demos/',
                mount_point='/preloaded_buffer',
            ),
        ]
    else:
        remote_mount_configs = []
    sweep_function(
        run_doodad_experiment,
        params,
        default_params=default_params,
        log_path=exp_name,
        mode=mode,
        docker_image='vitchyr/macaw-v1',
        code_dirs_to_mount=[
            '/home/vitchyr/git/macaw/',
            '/home/vitchyr/git/metaworld/',
            '/home/vitchyr/git/rand_param_envs/',
            '/home/vitchyr/git/doodad/',
        ],
        non_code_dirs_to_mount=[
            dict(
                local_dir='/home/vitchyr/.mujoco/',
                mount_point='/root/.mujoco',
            ),
        ],
        use_gpu=True,
        remote_mount_configs=remote_mount_configs,
    )


if __name__ == '__main__':
    main()
