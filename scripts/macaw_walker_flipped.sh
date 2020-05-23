#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="walker"

source env4/bin/activate
which python

python -m run --name macaw_walker_flipped2 --log_dir log/newruns --advantage_head_coef 0.1 --device cuda:0 --task_config config/walker_params/flipped.json --offline --load_inner_buffer --load_outer_buffer --replay_buffer_size 1000000 --outer_value_lr 1e-3 --outer_policy_lr 1e-3 --batch_size 256 --inner_batch_size 32 --maml_steps 1 --lrlr 1e-4 --target_vf_alpha 0. --grad_clip 1e4
