#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --job-name="mt_vel_extrapolation"

eval "$(conda shell.bash hook)"
conda activate macaw
which python

NAME="multitask_vel_extrapolation"
LOG_DIR="log/mt_test"
TASK_CONFIG="config/cheetah_vel/extrapolation.json"
MACAW_PARAMS="config/alg/multitask.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
