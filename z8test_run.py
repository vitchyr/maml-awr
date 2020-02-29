import gym

from gym.wrappers import TimeLimit
from stable_baselines import PPO1
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.policies import FeedForwardPolicy, register_policy
from metaworld.envs.mujoco.sawyer_xyz import SawyerNutDisassembleEnv
    
env = SawyerNutDisassembleEnv()
env = TimeLimit(env, max_episode_steps = 150)

class CustomPolicy(FeedForwardPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=[dict(pi=[128, 128, 128],
                                                          vf=[128, 128, 128])],
                                           feature_extraction="mlp")    
    
model = PPO1(CustomPolicy, env, verbose=1, tensorboard_log = 'ppo_logs')
model.learn(total_timesteps=10000000)
model.save("ppo1_nut_dissassemble")
    