import doodad
from doodad.wrappers.easy_launch import config
from doodad.wrappers.easy_launch.sweep_function import sweep_function
from src.run_experiment import run_doodad_experiment

if __name__ == '__main__':
    params = {
        'env': ['ant_dir'],
    }
    import ipdb; ipdb.set_trace()
    sweep_function(
        run_doodad_experiment,
        params,
        log_path='test_macaw',
        # mode='here_no_doodad',
        mode='local',
        docker_img='vitchyr/macaw-v0',
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
