import doodad
from doodad.wrappers.easy_launch import config
from doodad.wrappers.easy_launch import sweep_function
from src.run_experiment import run_doodad_experiment

if __name__ == '__main__':
    params = {
        'env': ['ant_dir'],
    }
    sweep_function(
        run_doodad_experiment,
        params,
        log_path='test_macaw',
        # mode='here_no_doodad',
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
            dict(
                local_dir='/home/vitchyr/mnt2/log2/demos/ant_four_dir/buffer_550k_each/macaw/',
                mount_point='/data',
            )
        ],
    )
    # set_start_method('spawn')
    # args = get_args()
    #
    # if args.instances == 1:
    #     if args.profile:
    #         import cProfile
    #         cProfile.runctx('run(args)', sort='cumtime', locals=locals(), globals=globals())
    #     else:
    #         run(args)
    # else:
    #     for instance_idx in range(args.instances):
    #         subprocess = Process(target=run, args=(args, instance_idx))
    #         subprocess.start()
