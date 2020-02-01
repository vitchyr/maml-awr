#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --job-name="ml10_mltest"
#SBATCH --array=0-4

export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64:/sailhome/rafailov/.mujoco/mujoco200/bin
source /iris/u/rafailov/cluster_env/cluster_env/bin/activate
cd /iris/u/rafailov/maml_awr/maml_awr/maml-awr
python -m run_experiment --env ml10 --mltest --task_idx $SLURM_ARRAY_TASK_ID --full_buffer_size 75000 --save_dir /iris/u/rafailov/data
