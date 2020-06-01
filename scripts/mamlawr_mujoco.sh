#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="mamlawr_mujoco"

eval "$(conda shell.bash hook)"
conda activate macaw
which python

LOG_DIR="log/NeurIPS2"
MACAW_PARAMS="config/alg/standard.json"
OVERRIDE="config/alg/overrides/noadv.json"
########################################################################

NAME="mamlawr_dir"
TASK_CONFIG="config/cheetah_dir/2tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="mamlawr_vel"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="mamlawr_ant"
TASK_CONFIG="config/ant_dir/50tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="mamlawr_walker"
TASK_CONFIG="config/walker_params/50tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE

########################################################################
