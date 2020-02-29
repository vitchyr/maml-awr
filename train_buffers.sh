python -m run --name cheetah_dir --env cheetah_dir --log_dir log/cheetah_dir --device cuda:0 --inner_policy_lr 0 --inner_value_lr 0 --include_goal --train_steps 100000 --gradient_steps_per_iteration 5 --replay_buffer_size 5 --full_buffer_size 20000 --save_buffers &
python -m run --name cheetah_vel --env cheetah_vel --log_dir log/cheetah_vel --device cuda:0 --inner_policy_lr 0 --inner_value_lr 0 --include_goal --train_steps 10000 --gradient_steps_per_iteration 5 --replay_buffer_size 5 --full_buffer_size 2000 --save_buffers &
python -m run --name ant_dir --env ant_dir --log_dir log/ant_dir --device cuda:0 --inner_policy_lr 0 --inner_value_lr 0 --include_goal --train_steps 250000 --gradient_steps_per_iteration 5 --replay_buffer_size 5 --full_buffer_size 50000 --save_buffers &
python -m run --name ant_goal --env ant_goal --log_dir log/ant_goal --device cuda:0 --inner_policy_lr 0 --inner_value_lr 0 --include_goal --train_steps 25000 --gradient_steps_per_iteration 5 --replay_buffer_size 5 --full_buffer_size 5000 --save_buffers &
python -m run --name humanoid_dir --env humanoid_dir --log_dir log/humanoid_dir --device cuda:0 --inner_policy_lr 0 --inner_value_lr 0 --include_goal --train_steps 25000 --gradient_steps_per_iteration 5 --replay_buffer_size 5 --full_buffer_size 5000 --save_buffers &
python -m run --name walker_param --env walker_param --log_dir log/walker_param --device cuda:0 --inner_policy_lr 0 --inner_value_lr 0 --include_goal --train_steps 100000 --gradient_steps_per_iteration 5 --replay_buffer_size 5 --full_buffer_size 20000 --save_buffers &
