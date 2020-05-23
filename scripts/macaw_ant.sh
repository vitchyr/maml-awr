#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --job-name="ant"

eval "$(conda shell.bash hook)"
conda activate macaw
which python

NAME="macaw_ant"
LOG_DIR="log/newruns4"
TASK_CONFIG="config/ant_dir/50tasks_offline_iris.json"
MACAW_PARAMS="config/alg/standard.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
