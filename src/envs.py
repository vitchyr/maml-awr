import numpy as np
from typing import Optional, Tuple, List
from src.tp_envs.half_cheetah_vel import HalfCheetahVelEnv as HalfCheetahVelEnv_
from src.tp_envs.half_cheetah_dir import HalfCheetahDirEnv as HalfCheetahDirEnv_
from src.tp_envs.ant_dir import AntDirEnv as AntDirEnv_
from src.tp_envs.ant_goal import AntGoalEnv as AntGoalEnv_
from src.tp_envs.humanoid_dir import HumanoidDirEnv as HumanoidDirEnv_
from src.tp_envs.walker_rand_params_wrapper import WalkerRandParamsWrappedEnv as WalkerRandParamsWrappedEnv_
from gym.spaces import Box
from metaworld.benchmarks.base import Benchmark
from metaworld.envs.mujoco.multitask_env import MultiClassMultiTaskEnv
from metaworld.envs.mujoco.env_dict import HARD_MODE_ARGS_KWARGS, HARD_MODE_CLS_DICT
from gym.wrappers import TimeLimit
from copy import deepcopy


class ML45Env(object):
    def __init__(self):
        self.n_tasks = 50
        self.tasks = list(HARD_MODE_ARGS_KWARGS['train'].keys()) + list(HARD_MODE_ARGS_KWARGS['test'].keys())

        self._max_episode_steps = 150

        self._env = None
        self._envs = []

        _cls_dict = {**HARD_MODE_CLS_DICT['train'], **HARD_MODE_CLS_DICT['test']}
        _args_kwargs = {**HARD_MODE_ARGS_KWARGS['train'], **HARD_MODE_ARGS_KWARGS['test']}
        for idx in range(self.n_tasks):
            task = self.tasks[idx]
            args_kwargs = _args_kwargs[task]
            args_kwargs['kwargs']['obs_type'] = 'with_goal'
            args_kwargs['task'] = task
            env = _cls_dict[task](*args_kwargs['args'], **args_kwargs['kwargs'])
            self._envs.append(TimeLimit(env, max_episode_steps=self._max_episode_steps))
        
        self.set_task_idx(0)

    def set_task_idx(self, idx):
        self._env = self._envs[idx]

    def __getattribute__(self, name):
        '''
        If we try to access attributes that only exist in the env, return the
        env implementation.
        '''
        try:
            return object.__getattribute__(self, name)
        except AttributeError as e:
            e_ = e
            try:
                return object.__getattribute__(self._env, name)
            except AttributeError as env_exception:
                pass
            except Exception as env_exception:
                e_ = env_exception
        raise e_


class HalfCheetahDirEnv(HalfCheetahDirEnv_):
    def __init__(self, tasks: List[dict], include_goal: bool = False):
        self.include_goal = include_goal
        super(HalfCheetahDirEnv, self).__init__()
        if tasks is None:
            tasks = [{'direction': 1}, {'direction': -1}]
        self.tasks = tasks
        self.set_task_idx(0)
        self._max_episode_steps = 200

    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs
    
    def set_task(self, task):
        self._task = task
        self._goal_dir = self._task['direction']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])
        

class HalfCheetahVelEnv(HalfCheetahVelEnv_):
    def __init__(self, tasks: List[dict] = None, include_goal: bool = False, one_hot_goal: bool = False, n_tasks: int = None):
        self.include_goal = include_goal
        self.one_hot_goal = one_hot_goal
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.n_tasks = len(tasks)
        super().__init__(tasks)
        self.set_task_idx(0)
        self._max_episode_steps = 200

    def _get_obs(self):
        if self.include_goal:
            obs = super()._get_obs()
            if self.one_hot_goal:
                goal = np.zeros((self.n_tasks,))
                goal[self.tasks.index(self._task)] = 1
            else:
                goal = np.array([self._goal_vel])
            obs = np.concatenate([obs, goal])
        else:
            obs = super()._get_obs()

        return obs
        
    def set_task(self, task):
        self._task = task
        self._goal_vel = self._task['velocity']
        self.reset()

<<<<<<< HEAD
<<<<<<< HEAD
    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal_vel = self._task['velocity']
        self._goal = self._goal_vel
        self.reset()

class AntDirEnv(AntDirEnv_):
    def __init__(self, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False, include_goal: bool = False):
        self.include_goal = include_goal
        super(AntDirEnv, self).__init__(forward_backward=True)
=======
class HalfCheetahVelEnv(HalfCheetahEnv):
    """Half-cheetah environment with target velocity, as described in [1]. The 
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py
    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each 
    time step a reward composed of a control cost and a penalty equal to the 
    difference between its current velocity and the target velocity. The tasks 
    are generated by sampling the target velocities from the uniform 
    distribution on [0, 2].
    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
        model-based control", 2012 
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """
    def __init__(self, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False):
>>>>>>> origin/master
=======
    def set_task_idx(self, idx):
        self.task_idx = idx
        self.set_task(self.tasks[idx])

class AntDirEnv(AntDirEnv_):
    def __init__(self, tasks: List[dict], n_tasks: int = None, include_goal: bool = False):
        self.include_goal = include_goal
        super(AntDirEnv, self).__init__(forward_backward=n_tasks == 2)
>>>>>>> origin/master
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.tasks = tasks
<<<<<<< HEAD
        self._task = tasks[task_idx]
        if single_task:
            self.tasks = self.tasks[task_idx:task_idx+1]
<<<<<<< HEAD
        self._goal = self._task['goal']
        self._max_episode_steps = 200
        self.info_dim = 1
    
    def step(self, action):
        obs, rew, done, info = super().step(action)
        if done == True:
            rew = rew - 1.0
            done = False
        return (obs, rew, done, info)
    
    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs
    
    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()
=======
        self._velocity = self._task['velocity']
        super(HalfCheetahVelEnv, self).__init__()
        self._max_episode_steps = 200
        self.info_dim = 1

    def compute_reward(self, observation: np.ndarray, action: np.ndarray, next_observation: np.ndarray, info: np.ndarray, next_info: np.ndarray):
        batch_shape = observation.shape[:-1]

        observation = observation.reshape((-1, observation.shape[-1]))
        action = action.reshape((-1, action.shape[-1]))
        next_observation = next_observation.reshape((-1, next_observation.shape[-1]))
        info = info.reshape((-1, info.shape[-1]))
        next_info = next_info.reshape((-1, next_info.shape[-1]))

        xpos_idx = 0
        xposbefore = info[:,0]
        xposafter = next_info[:,0]
        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._velocity)
        ctrl_cost = 0.5 * 1e-1 * np.square(action).sum(-1)

        rewards = forward_reward - ctrl_cost

        return rewards.reshape(batch_shape)

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._velocity)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost, task=self._velocity,
                     info=xposbefore, next_info=xposafter)
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks):
        velocities = self.np_random.uniform(0.0, 2.0, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks
>>>>>>> origin/master
=======
        self.n_tasks = len(self.tasks)
        self.set_task_idx(0)
        self._max_episode_steps = 200
    
    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self.tasks.index(self._task)
            except:
                pass
            one_hot = np.zeros(50, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs
    
    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()
>>>>>>> origin/master

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])
        

######################################################
######################################################
# <BEGIN DEPRECATED> #################################
######################################################
######################################################
class AntGoalEnv(AntGoalEnv_):
    def __init__(self, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False, include_goal: bool = False,
                 reward_offset: float = 0.0, can_die: bool = False):
        self.include_goal = include_goal
        self.reward_offset = reward_offset
        self.can_die = can_die
        super().__init__()
        if tasks is None:
            tasks = self.sample_tasks(130) #Only backward-forward tasks
        self.tasks = tasks
        self._task = tasks[task_idx]
        self.task = tasks[task_idx]
        if single_task:
            self.tasks = self.tasks[task_idx:task_idx+1]
        self._goal = self._task['goal']
        self._max_episode_steps = 200
        self.info_dim = 2
    
    def _get_obs(self):
        if self.include_goal:
            obs = super()._get_obs()
            obs = np.concatenate([obs, self._goal])
        else:
            obs = super()._get_obs()
        return obs
    
    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']        
        self.reset()
        
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/master
class HumanoidDirEnv(HumanoidDirEnv_):
    def __init__(self, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False, include_goal: bool = False):
        self.include_goal = include_goal
        super(HumanoidDirEnv, self).__init__()
<<<<<<< HEAD
=======

class HalfCheetahDirEnv(HalfCheetahEnv):
    """Half-cheetah environment with target direction, as described in [1]. The 
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand_direc.py
    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each 
    time step a reward composed of a control cost and a reward equal to its 
    velocity in the target direction. The tasks are generated by sampling the 
    target directions from a Bernoulli distribution on {-1, 1} with parameter 
    0.5 (-1: backward, +1: forward).
    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
        model-based control", 2012 
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """
    def __init__(self, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False):
>>>>>>> origin/master
=======
>>>>>>> origin/master
        if tasks is None:
            tasks = self.sample_tasks(130) #Only backward-forward tasks
        self.tasks = tasks
        self._task = tasks[task_idx]
        if single_task:
            self.tasks = self.tasks[task_idx:task_idx+1]
<<<<<<< HEAD
<<<<<<< HEAD
=======
>>>>>>> origin/master
        self._goal = self._task['goal']
        self._max_episode_steps = 200
        self.info_dim = 1
        
    def _get_obs(self):
        if self.include_goal:
            obs = super()._get_obs()
            obs = np.concatenate([obs, np.array([np.cos(self._goal), np.sin(self._goal)])])
        else:
            obs = super()._get_obs()
        return obs
    
    def step(self, action):
        obs, rew, done, info = super().step(action)
        if done == True:
            rew = rew - 5.0
            done = False
        return (obs, rew, done, info)
    
<<<<<<< HEAD
=======
        self._direction = self._task['direction']

        super(HalfCheetahDirEnv, self).__init__()
        self._max_episode_steps = 200
        self.info_dim = 1

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._direction * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
                     reward_ctrl=-ctrl_cost, task=self._task,
                     info=xposbefore, next_info=xposafter)
        return (observation, reward, done, infos)

>>>>>>> origin/master
=======
>>>>>>> origin/master
    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']
        self.reset()
######################################################
######################################################
# </END DEPRECATED> ##################################
######################################################
######################################################

class WalkerRandParamsWrappedEnv(WalkerRandParamsWrappedEnv_):
    def __init__(self, tasks: List[dict] = None, n_tasks: int = None, include_goal: bool = False):
        self.include_goal = include_goal
        super(WalkerRandParamsWrappedEnv, self).__init__(n_tasks=n_tasks if n_tasks is not None else 2)
        if tasks is not None:
            self.tasks = tasks
        self.n_tasks = len(self.tasks)
        self.set_task_idx(0)
        self._max_episode_steps = 200
        
    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self._goal
            except:
                pass
            one_hot = np.zeros(self.n_tasks, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs
        
    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
   
     
