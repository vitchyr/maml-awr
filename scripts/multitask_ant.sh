#!/bin/bash

NAME="multitask_ant_dir"
LOG_DIR="log/mt_test"
TASK_CONFIG="config/ant_dir/50tasks_offline.json"
MACAW_PARAMS="config/alg/multitask.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
