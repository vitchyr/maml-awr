#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=48:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name="macaw_task_ablation"

eval "$(conda shell.bash hook)"
conda activate macaw
which python

LOG_DIR="log/NeurIPS3"
MACAW_PARAMS="config/alg/standard.json"

########################################################################

NAME="macaw_vel_sixteenth"
TASK_CONFIG="config/cheetah_vel/sixteenth_tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS &

########################################################################

NAME="macaw_vel_half"
TASK_CONFIG="config/cheetah_vel/half_tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS &

########################################################################

NAME="macaw_vel_quarter"
TASK_CONFIG="config/cheetah_vel/quarter_tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS &

########################################################################

NAME="macaw_vel_eighth"
TASK_CONFIG="config/cheetah_vel/eighth_tasks_offline.json"
./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS

########################################################################
