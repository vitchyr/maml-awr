#
# Toy environments for testing maml-rawr
#
import gym
from gym.spaces import Box
import imageio
import numpy as np
from typing import Optional, Tuple, List

from src.utils import Experience


class Env(object):
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError()

    def reset(self) -> np.ndarray:
        raise NotImplementedError()

    def n_tasks() -> int:
        raise NotImplementedError()


class PointMass1DEnv(Env):
    targets = np.linspace(-1,1,4)
    def __init__(self, task_idx: Optional[int] = 0, fix_random_task: bool = False):
        self.action_space = Box(
            np.array([-1.]),
            np.array([1.])
        )
        self.observation_space = Box(
            np.array([-1., -1.]),
            np.array([1., 1.])
        )

        self._mass = 10
        self._dt = 0.1
        self._t = 0
        self._x = 0
        self._v = 0

        if task_idx is None and fix_random_task:
            task_idx = np.random.choice(PointMass1DEnv.n_tasks())

        self._task_idx = task_idx
        if task_idx is not None:
            self._task_target = PointMass1DEnv.targets[self._task_idx]
            
        self._this_task_idx = self._task_idx            
        self._max_episode_steps = 50

    @staticmethod
    def n_tasks() -> int:
        return len(PointMass1DEnv.targets)
        
    def render_rollout(self, rollout: List[Experience], path: Optional[str] = None) -> np.ndarray:
        RED, GREEN, BLUE = np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])
        resolution = 300
        padding = self._max_episode_steps
        image = np.zeros((self._max_episode_steps, resolution * 2, 3))
        for idx, experience in enumerate(rollout):
            path_column = resolution + int((experience.state[0]) * (resolution - 1))
            image[idx, path_column] = GREEN
            if idx % 2 == 0:
                column = resolution + int(resolution * np.tanh(self._task_target))
                image[idx, column] /= 2
                image[idx, column] += BLUE / 2

        padding_image = np.zeros((padding, resolution * 2, 3))
        image = np.concatenate((padding_image, image, padding_image), 0)

        if path is not None:
            imageio.imwrite(path, (image * 255).astype(np.uint8))
        
        return image

    def _compute_state(self) -> np.ndarray:
        # return np.array([np.tanh(self._x), self._v, self._t/float(self._max_episode_steps)], dtype=np.float32)
        # return np.array([(self._x), self._v, self._t/float(self._max_episode_steps)], dtype=np.float32)
        return np.array([np.tanh(self._x), self._v], dtype=np.float32)

    def reset(self) -> np.ndarray:
        if self._task_idx is None:
            self._this_task_idx = np.random.choice(PointMass1DEnv.n_tasks())
            self._task_target = PointMass1DEnv.targets[self._this_task_idx]

        self._x = np.random.normal() * 0.
        self._v = np.random.normal() * 0.
        self._t = 0

        return self._compute_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        action = action.clip(self.action_space.low, self.action_space.high)
        assert action.shape == (1,)
        # Move time forward
        self._t += 1

        # Update velocity with action and compute new position
        self._v += action[0] / self._mass
        self._x += self._dt * self._v
        
        done = self._t == self._max_episode_steps

        effort_penalty = -0.01 * np.abs(action[0])
        proximity_reward = -0.1 * np.abs(self._x - self._task_target) ** 2
        reward = proximity_reward + effort_penalty

        return self._compute_state(), reward, done, {'task_idx': self._this_task_idx}


class HalfCheetahDirEnv(Env):
    def __init__(self, task_idx: Optional[int] = None, fix_random_task: bool = False):
        self.env = gym.make('HalfCheetah-v2')
        if task_idx is None:
            task_idx = np.random.randint(self.n_tasks())

        self._task_idx = task_idx
        
    @staticmethod
    def n_tasks():
        return 2

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def _max_episode_steps(self):
        return self.env._max_episode_steps
    
    def reset(self) -> np.ndarray:
        return self.env.reset()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        xposbefore = self.env.sim.data.qpos[0]
        self.env.do_simulation(action, self.env.frame_skip)
        xposafter = self.env.sim.data.qpos[0]
        ob = self.env.env._get_obs()
        reward_ctrl = -0.1 * np.square(action).sum()

        if self._task_idx == 0:
            reward_run = (xposafter - xposbefore)/self.env.dt
        else:
            reward_run = -(xposafter - xposbefore)/self.env.dt

        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(reward_run=reward_run, reward_ctrl=reward_ctrl)
