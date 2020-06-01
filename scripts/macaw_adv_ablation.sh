#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=6
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --job-name="macaw_adv_ablation"

eval "$(conda shell.bash hook)"
conda activate macaw
which python

LOG_DIR="log/NeurIPS3"
MACAW_PARAMS="config/alg/standard.json"

########################################################################

NAME="macaw_vel_adv"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"
OVERRIDE="config/alg/overrides/adv.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE &

########################################################################

NAME="macaw_vel_noadv"
TASK_CONFIG="config/cheetah_vel/40tasks_offline.json"
OVERRIDE="config/alg/overrides/noadv.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS $OVERRIDE

########################################################################
