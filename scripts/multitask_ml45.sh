#!/bin/bash

NAME="multitask_ml45"
LOG_DIR="log/NeurIPS"
TASK_CONFIG="config/ml45/default.json"
MACAW_PARAMS="config/alg/multitask_ml45.json"

./scripts/runner.sh $NAME $LOG_DIR $TASK_CONFIG $MACAW_PARAMS
