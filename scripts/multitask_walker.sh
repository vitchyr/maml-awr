#!/bin/bash

NAME="multitask_walker"
LOG_DIR="log/mt_test"
TASK_CONFIG="config/walker_params/50tasks_offline_iris.json"
MACAW_PARAMS="config/alg/multitask_nobootstrap.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
