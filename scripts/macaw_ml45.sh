#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="macaw_ml45"

eval "$(conda shell.bash hook)"
conda activate macaw
which python

NAME="macaw_ml45"
LOG_DIR="log/NeurIPS2"
TASK_CONFIG="config/ml45/default.json"
MACAW_PARAMS="config/alg/standard_ml45.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
