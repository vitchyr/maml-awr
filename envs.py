#
# Toy environments for testing maml-rawr
#
from typing import Optional, Tuple
import numpy as np
from gym.spaces import Box


class Env(object):
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        raise NotImplementedError()

    def reset(self) -> np.ndarray:
        raise NotImplementedError()


class PointMass1DEnv(Env):
    def __init__(self, task_idx: Optional[int] = 0):
        self._targets = [-1, 1]

        self.action_space = Box(
            np.array([-1.]),
            np.array([1.])
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
        self._this_task_idx = self._task_idx
        if task_idx is not None:
            self._task_target = self._targets[task_idx]

        self._max_episode_steps = 50

    def _compute_state(self) -> np.ndarray:
        return np.array([np.tanh(self._x), self._v], dtype=np.float32)

    def reset(self) -> np.ndarray:
        if self._task_idx is None:
            self._this_task_idx = np.random.choice(range(len(self._targets)))
            self._task_target = self._targets[self._this_task_idx]

        self._x = np.random.normal() * 0.1
        self._v = np.random.normal() * 0.1
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

        effort_penalty = 0.01 * action[0]
        proximity_reward = -0.001 * (self._x - self._task_target) ** 2
        # print(self._task_target, self._x - self._task_target, effort_penalty, proximity_reward)
        
        reward = proximity_reward - effort_penalty
        reward = max(-10, reward)
        
        done = self._t == self._max_episode_steps

        return self._compute_state(), reward, done, {'task_idx': self._this_task_idx}

