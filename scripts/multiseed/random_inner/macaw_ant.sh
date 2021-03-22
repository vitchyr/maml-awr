#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name="ant"
#SBATCH --exclude=iris4

eval "$(conda shell.bash hook)"
conda activate macaw
which python

NAME="macaw_ant_randinner"
LOG_DIR="log/icml_rebuttal/multiseed"
TASK_CONFIG="config/ant_dir/50tasks_offline.json"
MACAW_PARAMS="config/alg/standard_rand_inner.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
