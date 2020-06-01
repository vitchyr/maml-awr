#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="mamlawr_quality_ablation"

eval "$(conda shell.bash hook)"
conda activate macaw
which python

LOG_DIR="log/NeurIPS3"
MACAW_PARAMS="config/alg/mamlawr.json"

########################################################################

NAME="mamlawr_vel_end"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"
OVERRIDE="config/alg/overrides/end.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="mamlawr_vel_middle"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"
OVERRIDE="config/alg/overrides/middle.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="mamlawr_vel_start"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"
OVERRIDE="config/alg/overrides/start.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE

########################################################################
