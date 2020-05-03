#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --job-name="vel"

source env4/bin/activate
which python

python -m run --name macaw_vel_lrs_quarter --log_dir log/newruns --advantage_head_coef 0.1 --device cuda:0 --task_config config/cheetah_vel/quarter_tasks_offline.json --offline --load_inner_buffer --load_outer_buffer --replay_buffer_size 2500000 --outer_value_lr 1e-3 --outer_policy_lr 1e-3 --batch_size 256 --inner_batch_size 32 --maml_steps 1 --lrlr 0
