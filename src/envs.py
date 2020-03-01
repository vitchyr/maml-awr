import numpy as np
from typing import Optional, Tuple, List
from src.tp_envs.half_cheetah_vel import HalfCheetahVelEnv as HalfCheetahVelEnv_
from src.tp_envs.half_cheetah_dir import HalfCheetahDirEnv as HalfCheetahDirEnv_
from src.tp_envs.ant_dir import AntDirEnv as AntDirEnv_
from src.tp_envs.ant_goal import AntGoalEnv as AntGoalEnv_
from src.tp_envs.humanoid_dir import HumanoidDirEnv as HumanoidDirEnv_
from src.tp_envs.walker_rand_params_wrapper import WalkerRandParamsWrappedEnv as WalkerRandParamsWrappedEnv_
from gym.spaces import Box


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
                if self.train:
                    goal = np.zeros((self.n_tasks,))
                    goal[self.tasks.index(self._task)] = 1
                else:
                    goal = np.zeros((self.n_tasks,))
            else:
                if self.train:
                    goal = np.array([self._goal_vel])
                else:
                    goal = np.array([-1.])
            obs = np.concatenate([obs, goal])
        else:
            obs = super()._get_obs()

        return obs
        
    def set_task(self, task):
        self._task = task
        self._goal_vel = self._task['velocity']
        self.reset()

    def set_task_idx(self, idx):
        self.set_task(self.tasks[idx])

class AntDirEnv(AntDirEnv_):
    def __init__(self, tasks: List[dict], n_tasks: int = None, include_goal: bool = False):
        self.include_goal = include_goal
        super(AntDirEnv, self).__init__(forward_backward=n_tasks == 2)
        if tasks is None:
            assert n_tasks is not None, "Either tasks or n_tasks must be non-None"
            tasks = self.sample_tasks(n_tasks)
        self.tasks = tasks
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
        
class HumanoidDirEnv(HumanoidDirEnv_):
    def __init__(self, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False, include_goal: bool = False):
        self.include_goal = include_goal
        super(HumanoidDirEnv, self).__init__()
        if tasks is None:
            tasks = self.sample_tasks(130) #Only backward-forward tasks
        self.tasks = tasks
        self._task = tasks[task_idx]
        if single_task:
            self.tasks = self.tasks[task_idx:task_idx+1]
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
   
     
