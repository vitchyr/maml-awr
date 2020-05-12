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

python -m run --name macaw_vel --log_dir log/adv_retest2 --device cuda:0 --task_config config/cheetah_vel/half_tasks_offline.json --macaw_params config/alg/standard.json
