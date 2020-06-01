#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="macaw_mujoco_nowlinear"

eval "$(conda shell.bash hook)"
conda activate macaw
which python

LOG_DIR="log/NeurIPS3"
MACAW_PARAMS="config/alg/standard.json"
OVERRIDE="config/alg/overrides/no_wlinear.json"

########################################################################

NAME="macaw_dir_nowlinear"
TASK_CONFIG="config/cheetah_dir/2tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="macaw_vel_nowlinear"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="macaw_ant_nowlinear"
TASK_CONFIG="config/ant_dir/50tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="macaw_walker_nowlinear"
TASK_CONFIG="config/walker_params/50tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE

########################################################################
