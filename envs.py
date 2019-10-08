#
# Toy environments for testing maml-rawr
#
from typing import Optional, Tuple
import numpy as np
from gym.spaces import Box


class Env(object):
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError()


class PointMass1DEnv(Env):
    def __init__(self, task_idx: Optional[int] = 0):
        self._targets = [-1, 1]

        self.action_space = Box(
            np.array([-1., -1.]),
            np.array([1., 1.])
        )
        self.observation_space = Box(
            np.array([-1., -1.]),
            np.array([1., 1.])
        )

        self._dt = 0.1
        self._t = 0
        self._x = 0
        self._v = 0

        self._task_idx = task_idx
        if task_idx is not None:
            self._task_target = self._targets[task_idx]

        self._max_episode_steps = 150

    def _compute_state(self) -> np.ndarray:
        return np.array([np.exp(-self._x) / (1 + np.exp(-self._x)), self._v])

    def reset(self) -> np.ndarray:
        if self._task_idx is None:
            self._task_target = self._targets[np.random.choice(range(len(self._targets)))]

        self._x = np.random.normal()
        self._v = np.random.normal()
        self._t = 0

        return self._compute_state()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        self._t += 1

        self._v += action[0]
        self._x += self._dt * self._v

        r = -(self._x - self._task_target)

        done = self._t == self._max_episode_steps

        return self._compute_state(), r, done, {}

