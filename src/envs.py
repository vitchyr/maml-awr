#
# Toy environments for testing maml-rawr
#
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

    # def show(self, rollout: List[Experience]) -> np.ndarray:
    #     raise NotImplementedError()


class PointMass1DEnv(Env):
    def __init__(self, task_idx: Optional[int] = 0):
        self._targets = [-1, 1]

        self.action_space = Box(
            np.array([-1.]),
            np.array([1.])
        )
        self.observation_space = Box(
            np.array([-10., -1., -1.]),
            np.array([10., 1., 1.])
        )

        self._dt = 0.1
        self._t = 0
        self._x = 0
        self._v = 0

        self._task_idx = task_idx
        self._this_task_idx = self._task_idx
        if task_idx is not None:
            self._task_target = self._targets[task_idx]

        self._max_episode_steps = 50

    def render(self, rollout: List[Experience], path: Optional[str] = None) -> np.ndarray:
        RED, GREEN, BLUE = np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])
        resolution = 300
        padding = self._max_episode_steps
        image = np.zeros((self._max_episode_steps, resolution * 2, 3))
        for idx, experience in enumerate(rollout):
            path_column = resolution + int(np.tanh(experience.state[0]) * (resolution - 1))
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
        return np.array([(self._x), self._v, self._t/float(self._max_episode_steps)], dtype=np.float32)
        # return np.array([(self._x), self._v], dtype=np.float32)

    def reset(self) -> np.ndarray:
        if self._task_idx is None:
            self._this_task_idx = np.random.choice(range(len(self._targets)))
            self._task_target = self._targets[self._this_task_idx]

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
        self._v += action[0]
        self._x += self._dt * self._v
        
        done = self._t == self._max_episode_steps

        # effort_penalty = -0.1 * action[0]
        effort_penalty = 0
        proximity_reward = -0.01 * np.abs(self._x - self._task_target)
        reward = proximity_reward + effort_penalty

        return self._compute_state(), reward, done, {'task_idx': self._this_task_idx}

