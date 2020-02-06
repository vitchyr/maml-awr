import numpy as np
from typing import Optional, Tuple, List
from src.tp_envs.half_cheetah_vel import HalfCheetahVelEnv as HalfCheetahVelEnv_
from src.tp_envs.half_cheetah_dir import HalfCheetahDirEnv as HalfCheetahDirEnv_
from src.tp_envs.ant_dir import AntDirEnv as AntDirEnv_
from src.tp_envs.ant_goal import AntGoalEnv as AntGoalEnv_
from src.tp_envs.humanoid_dir import HumanoidDirEnv as HumanoidDirEnv_
from src.tp_envs.walker_rand_params_wrapper import WalkerRandParamsWrappedEnv as WalkerRandParamsWrappedEnv_
from gym.spaces import Box


class Env(object):
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError()

    def reset(self) -> np.ndarray:
        raise NotImplementedError()

    def n_tasks() -> int:
        raise NotImplementedError()

class HalfCheetahDirEnv(HalfCheetahDirEnv_):
    def __init__(self, n_tasks: int, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False, include_goal: bool = False):
        if n_tasks != 2:
            raise ValueError('Can only have 2 tasks for direction task')

        self.include_goal = include_goal
        super(HalfCheetahDirEnv, self).__init__()
        if tasks is None:
            tasks = [{'direction': 1}, {'direction': -1}]
        self.tasks = tasks
        self._task = tasks[task_idx]
        if single_task:
            self.tasks = self.tasks[task_idx:task_idx+1]
        self._goal_dir = self._task['direction']
        self._goal = self._goal_dir
        self._max_episode_steps = 200
        self.info_dim = 1
    
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
    
#    def step(self, action):
#        obs, rew, done, info = super().step(action)
#        info['info'] = self._goal_dir
#        return (obs, rew, done, info)

    def set_task(self, task):
        self._task = task
        self._goal_dir = self._task['direction']
        self._goal = self._goal_dir
        self.reset()

    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal_dir = self._task['direction']
        self._goal = self._goal_dir
        self.reset()
        
class HalfCheetahVelEnv(HalfCheetahVelEnv_):
    def __init__(self,  n_tasks: int, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False, include_goal: bool = False, train: bool = True, one_hot_goal: bool = False):
        self.include_goal = include_goal
        self.one_hot_goal = one_hot_goal
        self.train = train
        tasks = self.sample_tasks(n_tasks, seed=1337 if train else 1338)
        self.n_tasks = n_tasks
        super().__init__(tasks)
        if single_task:
            tasks = tasks[task_idx:task_idx+1]
            self.tasks = tasks
        self._max_episode_steps = 200
        self.info_dim = 1

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
        
#    def step(self, action):
#        obs, rew, done, info = super().step(action)
#        info['info'] = self._goal_vel
#        return (obs, rew, done, info)
    
    def set_task(self, task):
        self._task = task
        self._goal_vel = self._task['velocity']
        self._goal = self._goal_vel
        self.reset()

    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal_vel = self._task['velocity']
        self._goal = self._goal_vel
        self.reset()

class AntDirEnv(AntDirEnv_):
    def __init__(self, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False, include_goal: bool = False, n_tasks: int = None):
        if n_tasks is None:
            n_tasks = 2
        self.include_goal = include_goal
        super(AntDirEnv, self).__init__(forward_backward=False)
        if tasks is None:
            tasks = self.sample_tasks(n_tasks)
        self.tasks = tasks
        self._task = tasks[task_idx]
        if single_task:
            self.tasks = self.tasks[task_idx:task_idx+1]
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
        
#    def step(self, action):
#        obs, rew, done, info = super().step(action)
#        info['info'] = self._goal
#        return (obs, rew, done, info)
    
    def set_task(self, task):
        self._task = task
        self._goal = task['goal']
        self.reset()

    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = self._task['goal']
        self.reset()
        
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

#    def step(self, action):
#        obs, rew, done, info = super().step(action)
#        info['info'] = self._goal
#        return (obs, rew, done, info)
    
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
    
class WalkerRandParamsWrappedEnv(WalkerRandParamsWrappedEnv_):
    def __init__(self, tasks: List[dict] = None, task_idx: int = 0, single_task: bool = False, include_goal: bool = False):
        self.include_goal = include_goal
        super(WalkerRandParamsWrappedEnv, self).__init__(n_tasks=50)
        if tasks is not None:
            self.tasks = tasks
            self.reset_task(task_idx)
#        if tasks is None:
#            tasks = self.sample_tasks(50) 
#        self.tasks = tasks
        self._task = self.tasks[task_idx]
        if single_task:
            self.tasks = self.tasks[task_idx:task_idx+1]
        self._max_episode_steps = 200
        
    def _get_obs(self):
        if self.include_goal:
            idx = 0
            try:
                idx = self._goal
            except:
                pass
#            one_hot = np.zeros(len(self.tasks), dtype=np.float32)
            one_hot = np.zeros(50, dtype=np.float32)
            one_hot[idx] = 1.0
            obs = super()._get_obs()
            obs = np.concatenate([obs, one_hot])
        else:
            obs = super()._get_obs()
        return obs
        
    def step(self, action):
        obs, rew, done, info = super().step(action)
        if done == True:
            rew = rew - 1.0
            done = False
        return (obs, rew, done, info)
        
    def set_task_idx(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
   
     
