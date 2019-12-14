import numpy as np
from typing import List, Optional

from src.tp_envs.ant_multitask_base import MultitaskAntEnv
from . import register_env


# Copy task structure from https://github.com/jonasrothfuss/ProMP/blob/master/meta_policy_search/envs/mujoco_envs/ant_rand_goal.py
@register_env('ant-goal')
class AntGoalEnv(MultitaskAntEnv):
    def __init__(self, tasks: List[dict] = None, task_idx: int = None, **kwargs):
        if tasks is None:
            tasks = [AntGoalEnv.sample_tasks(1, a=a, r=r)[0]
                     for a, r in zip(list(np.linspace(0,0.25,5) * 2 * np.pi), [4] * 5)]
        if task_idx is None:
            task_idx = 0
        else:
            tasks = [tasks[task_idx]]
            task_idx = 0

        self.tasks = tasks
        self._task_idx = task_idx
        self._task = tasks[task_idx]
        self._goal = self._task['goal']
        super(AntGoalEnv, self).__init__(tasks, task_idx, **kwargs)
        self._max_episode_steps = 200

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        xposafter = np.array(self.get_body_com("torso"))
        goal_reward = -np.sum(np.abs(xposafter[:2] - self._goal)) # make it happy, not suicidal

        ctrl_cost = .1 * np.square(action).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 0.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            goal_forward=goal_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward,
        )

    def task_description_dim(self, one_hot: bool = True):
        return len(self.tasks) if one_hot else 2

    def task_description(self, task: dict = None, batch: Optional[int] = None, one_hot: bool = True):
        idx = self._task_idx
        one_hot = np.zeros((self.task_description_dim(),), dtype=np.float32)
        one_hot[idx] = 1
        if batch:
            one_hot = one_hot[None,:].repeat(batch, 0)
        return one_hot

    @staticmethod
    def sample_tasks(num_tasks, a: float = None, r: float = None):
        if a is None:
            a = np.random.random(num_tasks) * 2 * np.pi
        if r is None:
            r = 3 * np.random.random(num_tasks) ** 0.5
        goals = np.stack((r * np.cos(a), r * np.sin(a)), axis=-1)
        if num_tasks == 1:
            goals = goals[None,:]
        tasks = [{'goal': goal} for goal in goals]
        return tasks

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat,
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])
