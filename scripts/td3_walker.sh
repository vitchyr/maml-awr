#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --job-name="walker"

eval "$(conda shell.bash hook)"
conda activate macaw
python -u -m run --device cuda:0 --name mql_walker --log_dir log/td3ctx --task_config config/walker_params/50tasks_offline.json --macaw_param config/alg/td3ctx.json --td3ctx
