#!/bin/bash

# these values are inclusive
START=16
END=25

for i in $(seq $START $END)
do
    CMD="python -m collect_buffers --env ml45 --task_idx $i --alg sac --replay_buffer_size 1000000 --full_buffer_size 5000000 --task_path /home/ubuntu/maml-awr/tasks --log_dir /home/ubuntu/NIPS2020_data/ML45"
    echo $CMD
    $CMD &
done
