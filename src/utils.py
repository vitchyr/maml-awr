from typing import NamedTuple, List

import numpy as np


class Experience(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float
    done: bool
    

class ReplayBuffer(object):
    def __init__(self, trajectory_length: int, state_dim: int, action_dim: int, max_trajectories: int = 10000,
                 discount_factor: float = 0.99):
        self._trajectories = np.empty((max_trajectories, trajectory_length, state_dim + action_dim + state_dim + state_dim + 1 + 1 + 1 + 1), dtype=np.float32)
        self._stored_trajectories = 0
        self._new_trajectory_idx = 0
        self._max_trajectories = max_trajectories
        self._trajectory_length = trajectory_length
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._discount_factor = discount_factor
        
    def __len__(self):
        return self._stored_trajectories

    def add_trajectory(self, trajectory: List[Experience]):
        if len(trajectory) != self._trajectory_length:
            raise ValueError(f'Invalid trajectory length: {len(trajectory)}')

        mc_reward = 0
        terminal_state = None
        terminal_factor = 1
        for idx, experience in enumerate(trajectory[::-1]):
            if terminal_state is None:
                terminal_state = experience.next_state

            slice_idx = 0
            self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + self._state_dim] = experience.state
            slice_idx += self._state_dim

            self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + self._action_dim] = experience.action
            slice_idx += self._action_dim

            self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + self._state_dim] = experience.next_state
            slice_idx += self._state_dim
            
            self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + self._state_dim] = terminal_state
            slice_idx += self._state_dim

            terminal_factor *= self._discount_factor
            self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + 1] = terminal_factor
            slice_idx += 1
            
            self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + 1] = experience.done
            slice_idx += 1

            self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + 1] = experience.reward
            slice_idx += 1

            mc_reward = experience.reward + self._discount_factor * mc_reward

            self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + 1] = mc_reward

        self._new_trajectory_idx += 1
        self._new_trajectory_idx %= self._max_trajectories
            
        if self._stored_trajectories < self._max_trajectories:
            self._stored_trajectories += 1

    def add_trajectories(self, trajectories: List[List[Experience]]):
        for trajectory in trajectories:
            self.add_trajectory(trajectory)

    def sample(self, batch_size):
        idxs = np.random.choice(np.arange(self._stored_trajectories * self._trajectory_length), batch_size)
        trajectory_idxs = idxs // self._trajectory_length
        time_steps = idxs % self._trajectory_length

        batch = self._trajectories[trajectory_idxs, time_steps]

        return batch


def generate_test_trajectory(state_dim: int, action_dim: int):
    trajectory = []
    next_state = np.random.uniform(0,1,(state_dim,))
    for idx in range(trajectory_length):
        state = next_state
        action = np.random.uniform(-1,0,(action_dim,))
        next_state = np.random.uniform(0,1,(state_dim,))
        reward = np.random.uniform()
        trajectory.append(Experience(state, action, next_state, reward))

    return trajectory

if __name__ == '__main__':
    trajectory_length = 100
    state, action = 6, 4
    buf = ReplayBuffer(trajectory_length, state, action)

    for idx in range(100):
        buf.add_trajectory(generate_test_trajectory(state, action))

    buf.add_trajectories([generate_test_trajectory(state, action) for _ in range(100)])

    print(len(buf))
    print(buf.sample(2))

