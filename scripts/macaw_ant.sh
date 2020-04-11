#!/bin/bash
#SBATCH --partition=iris
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --job-name="macaw_ant"

#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64:/sailhome/rafailov/.mujoco/mujoco200/bin
export MUJOCO_PY_MJKEY_PATH=/iris/u/em7/.mujoco/mjkey.txt
export MUJOCO_PY_MUJOCO_PATH=/iris/u/em7/.mujoco/mujoco200
export MUJOCO_PY_MJPRO_PATH=/iris/u/em7/.mujoco/mjpro131
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/iris/u/em7/.mujoco/mujoco200/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/iris/u/em7/.mujoco/mjpro131/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda-10.0/lib64
cd /iris/u/em7/code/maml-rawr
which python
python -m run --name $1 --log_dir $2 --advantage_head_coef 0.1 --device cuda:0 --task_config config/ant_dir/50tasks_offline_iris.json --offline --load_inner_buffer --load_outer_buffer --replay_buffer_size 2000000 --outer_value_lr 1e-3 --outer_policy_lr 1e-3 --batch_size 256 --inner_batch_size 16 --maml_steps $3
