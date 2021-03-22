#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=128:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="vel"
#SBATCH --exclude=iris4

eval "$(conda shell.bash hook)"
conda activate macaw
which python

NAME="macaw_vel_randinner"
LOG_DIR="log/icml_rebuttal/multiseed"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"
MACAW_PARAMS="config/alg/standard_rand_inner.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
