#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name="mt_ant"

eval "$(conda shell.bash hook)"
conda activate macaw
which python

NAME="multitask_ant"
LOG_DIR="log/NeurIPS_multiseed_multitask"
TASK_CONFIG="config/ant_dir/50tasks_offline.json"
MACAW_PARAMS="config/alg/multitask_standard.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
