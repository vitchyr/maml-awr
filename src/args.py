import argparse


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--goal_dim', type=int, default=0)
    parser.add_argument('--info_dim', type=int, default=0)
    parser.add_argument('--multitask', action='store_true')
    parser.add_argument('--mt_value_lr', type=float, default=1e-2)
    parser.add_argument('--mt_policy_lr', type=float, default=1e-3)
    parser.add_argument('--pad_buffers', action='store_true')
    parser.add_argument('--task_batch_size', type=int, default=None)
    parser.add_argument('--action_sigma', type=float, default=0.2)
    parser.add_argument('--traj_hold_out_test', dest='traj_hold_out_train', action='store_false')
    parser.add_argument('--traj_hold_out_train', action='store_true', default=None)
    parser.add_argument('--trim_obs', type=int, default=None)
    parser.add_argument('--n_tasks', type=int, default=None)
    parser.add_argument('--multitask_eval', action='store_true')
    parser.add_argument('--multitask_bias_only', action='store_true')
    parser.add_argument('--mltest', action='store_true')
    parser.add_argument('--vae_steps', type=int, default=None)
    parser.add_argument('--noclamp', action='store_true')
    parser.add_argument('--lrlr', type=float, default=1e-4)
    parser.add_argument('--huber', action='store_true')
    parser.add_argument('--kld_coef', type=float, default=1.0)
    parser.add_argument('--cvae_skip', type=int, default=10)
    parser.add_argument('--cvae_prior_conditional', action='store_true')
    parser.add_argument('--cvae_preprocess', action='store_true')
    parser.add_argument('--exploration_batch_size', type=int, default=64)
    parser.add_argument('--exploration_reg', type=float, default=None)
    parser.add_argument('--trim_episodes', type=int, default=0)
    parser.add_argument('--episode_length', type=int, default=None)
    parser.add_argument('--normalize_values_outer', action='store_true')
    parser.add_argument('--normalize_values', action='store_true')
    parser.add_argument('--fixed_exploration_task', type=int, default=None)
    parser.add_argument('--random_task_percent', type=float, default=None)
    parser.add_argument('--no_norm', action='store_true')
    parser.add_argument('--no_bootstrap', action='store_true')
    parser.add_argument('--q', action='store_true')
    parser.add_argument('--reward_offset', type=float, default=0.)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--render_exploration', action='store_true')
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--random', action='store_true')
    parser.add_argument('--train_exploration', action='store_true')
    parser.add_argument('--sample_exploration_inner', action='store_true')
    parser.add_argument('--cvae', action='store_true')
    parser.add_argument('--iw_exploration', action='store_true')
    parser.add_argument('--traj_iw_exploration', action='store_true')
    parser.add_argument('--unconditional', action='store_true')
    parser.add_argument('--latent_dim', type=int, default=64)
    parser.add_argument('--n_adaptations', type=int, default=1)
    parser.add_argument('--pre_adapted', action='store_true')
    parser.add_argument('--train_steps', type=int, default=1000000)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--inner_batch_size', type=int, default=256)
    parser.add_argument('--inner_policy_lr', type=float, default=0.01)
    parser.add_argument('--inner_value_lr', type=float, default=0.01)
    parser.add_argument('--outer_policy_lr', type=float, default=1e-4)
    parser.add_argument('--outer_value_lr', type=float, default=1e-4)
    parser.add_argument('--exploration_lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--vis_interval', type=int, default=1000)
    parser.add_argument('--log_dir', type=str, default='log')
    parser.add_argument('--include_goal', action='store_true')
    parser.add_argument('--single_task', action='store_true')  
    parser.add_argument('--one_hot_goal', action='store_true')
    parser.add_argument('--task_idx', type=int, default=None)
    parser.add_argument('--instances', type=int, default=1)
    parser.add_argument('--name', type=str, default=None)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--gradient_steps_per_iteration', type=int, default=50)
    parser.add_argument('--replay_buffer_size', type=int, default=20000)
    parser.add_argument('--full_buffer_size', type=int, default=20000)
    parser.add_argument('--discount_factor', type=float, default=0.99)
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--vf_archive', type=str, default=None)
    parser.add_argument('--ap_archive', type=str, default=None)
    parser.add_argument('--ep_archive', type=str, default=None)
    parser.add_argument('--initial_rollouts', type=int, default=30)
    parser.add_argument('--offline', action='store_true')
    parser.add_argument('--offline_outer', action='store_true')
    parser.add_argument('--offline_inner', action='store_true')
    parser.add_argument('--grad_clip', type=float, default=1e9) # Essentially no clip, but use this to measure the size of gradients
    parser.add_argument('--exp_advantage_clip', type=float, default=10.0)
    parser.add_argument('--eval_maml_steps', type=int, default=1)
    parser.add_argument('--maml_steps', type=int, default=1)
    parser.add_argument('--adaptation_temp', type=float, default=1)
    parser.add_argument('--exploration_temp', type=float, default=1)
    parser.add_argument('--no_bias_linear', action='store_true')
    parser.add_argument('--advantage_head_coef', type=float, default=None)
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--target_reward', type=float, default=None)
    parser.add_argument('--save_buffers', action='store_true')
    parser.add_argument('--ratio_clip', type=float, default=0.5)
    parser.add_argument('--task_config', type=str, default=None)
    parser.add_argument('--buffer_paths', type=str, nargs='+', default=None)
    parser.add_argument('--test_buffer_paths', type=str, nargs='+', default=None)
    parser.add_argument('--load_inner_buffer', action='store_true')
    parser.add_argument('--load_outer_buffer', action='store_true')
    return parser.parse_args()
