#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="vel_sixtyfourth_data"

source env4/bin/activate
which python

python -m run --name macaw_vel_sixtyfourth_data --log_dir log/newruns --device cuda:0 --task_config config/cheetah_vel/half_tasks_offline.json --macaw_params config/alg/standard.json --buffer_skip 64
