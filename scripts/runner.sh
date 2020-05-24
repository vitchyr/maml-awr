#!/bin/bash
# Run MACAW in a given configuration

NAME=$1
LOG_DIR=$2
TASK_CONFIG=$3
MACAW_PARAMS=$4

CMD="python3 -u -m run --device cpu --name $NAME --log_dir $LOG_DIR --task_config $TASK_CONFIG --macaw_params $MACAW_PARAMS"

echo "***************************************************"
echo "***************************************************"
echo "RUNNING COMMAND:"
echo $CMD
echo "***************************************************"

echo "***************************************************"
echo "SAVING TO $LOG_DIR/$NAME"
echo "***************************************************"

echo "***************************************************"
echo "TASK CONFIGURATION"
cat $TASK_CONFIG
echo "***************************************************"

echo "***************************************************"
echo "MACAW CONFIGURATION"
cat $MACAW_PARAMS
echo "***************************************************"
echo "***************************************************"

$CMD
