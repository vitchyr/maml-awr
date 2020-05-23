#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --job-name="ml10_mltrain"
#SBATCH --array=0-9

source activate maml_awr
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64:/sailhome/rafailov/.mujoco/mujoco200/bin
cd /iris/u/rafailov/maml_awr/maml_awr/maml-awr
which python
python -m run_experiment --env ml10 --task_idx $SLURM_ARRAY_TASK_ID --replay_buffer_size 1000 --full_buffer_size 75000 --save_dir /iris/u/rafailov/data
