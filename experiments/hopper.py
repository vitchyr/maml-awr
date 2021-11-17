import click

from doodad.wrappers.easy_launch import sweep_function
from src.launcher import run_doodad_experiment


@click.command()
@click.option('--debug', is_flag=True, default=False)
@click.option('--suffix', default=None)
@click.option('--nseeds', default=1)
@click.option('--mode', default='azure')
def main(debug, suffix, nseeds, mode):
    path_parts = __file__.split('/')
    suffix = '' if suffix is None else '--{}'.format(suffix)
    exp_name = 'macaw-{}{}'.format(
        path_parts[-1].split('.')[0],
        suffix,
    )
    use_gpu = True

    if debug:
        exp_name = 'dev--' + exp_name
        mode = 'here_no_doodad'
        nseeds = 1
        use_gpu = False

    params = {
        'env': ['hopper_params'],
        'use_rlkit': [
            True,
        ],
        'seed': list(range(nseeds)),
    }

    if mode == 'azure':
        remote_mount_configs = [
            dict(
                local_dir='/doodad_tmp/',
                mount_point='/preloaded_data',
            ),
        ]
        exp_base_path = '/preloaded_data/'
        device = 'cuda:0'
    elif mode == 'here_no_doodad':
        remote_mount_configs = []
        exp_base_path = '/Users/vitchyr/data/doodad/'
        device = 'cpu'
    else:
        raise ValueError(mode)

    exp_dir_path = exp_base_path + '21-11-14_smac-iclr22-hopper--hopper-data-gen--v4/23h-02m-27s_run0/'
    default_params = {
        'pretrain_buffer_path': exp_dir_path + 'extra_snapshot_itr40.cpkl',
        'saved_tasks_path': exp_dir_path + 'tasks_description.joblib',
        'load_buffer_kwargs': {
            'start_idx': -1200,
            'end_idx': None,
        },
        'device': device,
    }

    print(exp_name)
    sweep_function(
        run_doodad_experiment,
        params,
        default_params=default_params,
        log_path=exp_name,
        mode=mode,
        docker_image='vitchyr/macaw-v2',
        code_dirs_to_mount=[
            '/Users/vitchyr/code/maml-awr/',
            '/Users/vitchyr/code/metaworld/',
            '/Users/vitchyr/code/multiworld/',
            '/Users/vitchyr/code/rand_param_envs/',
            '/Users/vitchyr/code/doodad/',
            '/Users/vitchyr/code/railrl-private/',
        ],
        use_gpu=use_gpu,
        remote_mount_configs=remote_mount_configs,
    )
    print(exp_name)


if __name__ == '__main__':
    main()
