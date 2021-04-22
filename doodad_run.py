import doodad
from doodad.wrappers.easy_launch import config
from doodad.wrappers.easy_launch import sweep_function

if __name__ == '__main__':
    target = '/home/vitchyr/git/macaw/run.py'
    az_mount = doodad.MountAzure(
        '',
        mount_point='/output',
    )
    mode = 'azure'
    log_path = 'test_macaw'
    docker_img = 'vitchyr/macaw-v0'
    sweeper, output_mount = sweep_function._create_sweeper_and_output_mount(mode, log_path, docker_img)
    # import ipdb; ipdb.set_trace()
    params = {
        'env': ['ant_dir'],
    }
    sweeper.run_sweep_local(
        target,
        params,
        return_output=True,
        verbose=True,
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
