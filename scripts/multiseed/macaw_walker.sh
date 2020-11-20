#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --job-name="walker"

eval "$(conda shell.bash hook)"
conda activate macaw
which python

NAME="macaw_walker"
LOG_DIR="log/iclr_rebuttal/multiseed"
TASK_CONFIG="config/walker_params/50tasks_offline.json"
MACAW_PARAMS="config/alg/standard.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS

