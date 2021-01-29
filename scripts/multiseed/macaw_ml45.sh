#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name="macaw_ml45"

eval "$(conda shell.bash hook)"
conda activate macaw
which python

NAME="macaw_ml45_logtargets"
LOG_DIR="log/iclr_rebuttal/multiseed"
TASK_CONFIG="config/ml45/default.json"
MACAW_PARAMS="config/alg/standard_ml45.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
