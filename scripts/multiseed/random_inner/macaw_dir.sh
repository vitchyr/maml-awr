#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --job-name="dir"
#SBATCH --exclude=iris4

eval "$(conda shell.bash hook)"
conda activate macaw
which python

NAME="macaw_dir_randinner"
LOG_DIR="log/icml_rebuttal/multiseed"
TASK_CONFIG="config/cheetah_dir/2tasks_offline.json"
MACAW_PARAMS="config/alg/standard_rand_inner.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
