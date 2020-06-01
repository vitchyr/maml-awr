#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=96:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --job-name="vel"

eval "$(conda shell.bash hook)"
conda activate macaw
which python

NAME="macaw_vel_validation"
LOG_DIR="log/newruns4"
TASK_CONFIG="config/cheetah_vel/validation2.json"
MACAW_PARAMS="config/alg/validation.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
