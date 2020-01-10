## Prerequisites

This project requires Python 3.6; Ubuntu 16.04 users can install it from Felix Kurll's deadsnakes PPA by following the instructions [here](https://askubuntu.com/questions/865554/how-do-i-install-python-3-6-using-apt-get).

You'll also need to have MuJoCo installed (version `mujoco200`, but `mjpro131` might work as well).

Required Python packages can be installed with `pip install -r requirements.txt`. Sometimes the `mujoco_py` package gives some errors when installing. This can be due to a variety of things, including missing graphics drivers or misplaced/incorrectly installed mujoco binaries. The mujoco_py repo has more installation information: `https://github.com/openai/mujoco-py`.

## Running training

Training can be invoked by running `python -m run --name NAME --env [cheetah_dir|cheetah_vel] --device [cpu|cuda:0|cuda:X] --log_dir SOME_DIR --advantage_head_coef 0.1`, which will run on the CPU by default.

For example:

`python -m run --name test --env cheetah_dir --log_dir log/test --advantage_head_coef 0.1`

To run on GPU 0:

`python -m run --name test --env cheetah_dir --log_dir log/test --advantage_head_coef 0.1 --device cuda:0`

To run the cheetah velocity task (target velocities [0, 0.5, 1, 1.5, 2]):

`python -m run --name test --env cheetah_vel --log_dir log/test --advantage_head_coef 0.1 --device cuda:0`

To run training with the exploration policy (by default, inner loop and outer loop buffers are both populated with on-policy data), pass `--train_exploration` `--cvae` and `--sample_exploration_inner`.

`python -m run --name test --env cheetah_vel --log_dir log/test --advantage_head_coef 0.1 --device cuda:0 --train_exploration --cvae --sample_exploration_inner`

`--cvae` does conditional VAE exploration (as opposed to importance weighting), `--sample_exploration_inner` uses the exploration policy for inner loop data instead of on-policy post-adaptation data.

To run a pre-trained network, you can use the `--ap_archive`, `--vf_archive`, and `--ep_archive` parameters for adaptation policy, value function, and exploration policy, respectively. These archives will be saved in the directory you specify for logging.