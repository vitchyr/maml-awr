import doodad
from doodad.wrappers.easy_launch import config
from doodad.wrappers.easy_launch import sweep_function
from src.launcher import run_doodad_experiment

if __name__ == '__main__':
    params = {
        'env': ['ant_dir'],
        'use_rlkit': [
            True,
        ],
    }
    default_params = {
        'buffer_path_template':'/preloaded_buffer/ant_dir_32/macaw_buffer_iter50/macaw_buffer_task_{}.npy',
        'saved_tasks_path': '/preloaded_buffer/ant_dir_32/macaw_buffer_iter50/tasks.pkl',
        'train_task_idxs': [0, 1, 2, 3],
        'test_task_idxs': [4, 5, 6, 7],
    }
    sweep_function(
        run_doodad_experiment,
        params,
        default_params=default_params,
        log_path='macaw_ant_4_dir',
        mode='local',
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
        remote_mount_configs=[
            dict(
                local_dir='/home/vitchyr/mnt2/log2/demos/',
                mount_point='/preloaded_buffer',
            ),
        ]
    )
