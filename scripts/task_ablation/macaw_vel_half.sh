#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --job-name="vel_half_tasks"

source env4/bin/activate
which python

python -m run --name macaw_vel_lrs_half --log_dir log/newruns --device cuda:0 --task_config config/cheetah_vel/half_tasks_offline.json --macaw_params config/alg/standard.json
