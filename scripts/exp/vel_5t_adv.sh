#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --job-name="vel"

source env4/bin/activate
which python

python -m run --name macaw_vel_5t_adv --log_dir log/newruns2 --device cuda:0 --task_config config/cheetah_vel/5tasks_offline.json --macaw_params config/alg/adv_regression.json
