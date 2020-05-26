#!/bin/bash

NAME="multitask_cheetah_dir"
LOG_DIR="log/mt_test"
TASK_CONFIG="config/cheetah_dir/2tasks_offline.json"
MACAW_PARAMS="config/alg/multitask.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS