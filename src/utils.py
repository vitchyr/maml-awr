from typing import NamedTuple, List

import numpy as np
import torch
import torch.nn as nn


class RunningEstimator(object):
    def __init__(self):
        self._mu = None
        self._mu2 = None
        self._n = 0

    def mean(self):
        return self._mu

    def var(self):
        return self._mu2 - self._mu ** 2
    
    def std(self):
        return (self.var() + 1e-8) ** 0.5

    def add(self, xs):
        if isinstance(xs, torch.Tensor):
            xs = xs.detach()
        if self._mu is None:
            self._mu = xs.mean()
            self._mu2 = (xs ** 2).mean()
        else:
            self._mu += ((xs - self._mu) * (1 / (self._n + 1))).mean()
            self._mu2 += ((xs**2 - self._mu2) * (1/(self._n+1))).mean()

        self._n += 1


def argmax(module: nn.Module, arg: torch.tensor):
    print('Computing argmax')
    arg.requires_grad = True
    opt = torch.optim.Adam([arg], lr=0.1)
    for idx in range(1000):
        out = module(arg)
        loss = -out
        prev_arg = arg.clone()
        loss.backward()
        opt.step()
        opt.zero_grad()
        module.zero_grad()
        d = (arg-prev_arg).norm(2)
        if d < 1e-4:
            print('breaking')
            break
    #print(f'Final d: {d}')
    return arg, out


def kld(p, q):
    p_mu = p[:,:p.shape[-1] // 2]
    q_mu = q[:,:q.shape[-1] // 2]

    p_std = (p[:,p.shape[-1] // 2:] / 2).exp()
    q_std = (q[:,q.shape[-1] // 2:] / 2).exp()
    dp = torch.distributions.Normal(p_mu, p_std)
    dq = torch.distributions.Normal(q_mu, q_std)

    return torch.distributions.kl_divergence(dp, dq).sum(-1)
    

class Experience(NamedTuple):
    state: np.ndarray
    action: np.ndarray
    next_state: np.ndarray
    reward: float
    done: bool
    #log_prob: float
    info: np.ndarray = None
    next_info: np.ndarray = None
    

class MiniBatch(object):
    def __init__(self, samples: np.ndarray, action_dim: int, observation_dim: int):
        self._samples = torch.tensor(samples).float()
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        
    def to(self, device: torch.device):
        self._samples = self._samples.to(device)

        return self

    def obs(self):
        return self._samples[:,:self._observation_dim]

    def act(self):
        return self._samples[:,self._observation_dim:self._observation_dim + self._action_dim]

    def next_obs(self):
        return self._samples[:,self._observation_dim + self._action_dim:self._observation_dim * 2 + self._action_dim]

    def terminal_obs(self):
        return self._samples[:,self._observation_dim * 2 + self._action_dim:self._observation_dim * 3 + self._action_dim]

    def log_prob(self):
        return self._samples[:,-5]

    def terminal_factor(self):
        return self._samples[:,-4]

    def done(self):
        return self._samples[:,-3]

    def reward(self):
        return self._samples[:,-2]

    def reward(self):
        return self._samples[:,-1]


class ReplayBuffer(object):
    @staticmethod
    #def join(buffers: List[ReplayBuffer]) -> ReplayBuffer:
    def join(buffers):
        b0 = buffers[0]
        trajectories = np.concatenate([b._trajectories for b in buffers])
        new_buffer = ReplayBuffer(trajectories.shape[0], b0._state_dim, b0._action_dim, b0._info_dim, b0._max_trajectories,
                                  b0._discount_factor, b0._immutable, b0._trim_suffix)
        new_buffer._trajectories = trajectories
        new_buffer._stored_trajectories = trajectories.shape[0]

        return new_buffer

    def __init__(self, trajectory_length: int, state_dim: int, action_dim: int, info_dim: int = 0, max_trajectories: int = 10000,
                 discount_factor: float = 0.99, immutable: bool = False, load_from: str = None, silent: bool = False, trim_suffix: int = 0):
        self._trajectories = np.empty((max_trajectories, trajectory_length, state_dim + action_dim + state_dim + state_dim + info_dim * 2 + 1 + 1 + 1 + 1), dtype=np.float32)
        self._trajectories.fill(np.float('nan'))
        self._stored_trajectories = 0
        self._new_trajectory_idx = 0
        self._max_trajectories = max_trajectories
        self._trajectory_length = trajectory_length
        self._state_dim = state_dim
        self._action_dim = action_dim
        self._info_dim = info_dim
        self._discount_factor = discount_factor
        self._immutable = immutable
        self._trim_suffix = trim_suffix
        if load_from is not None:
            if not silent:
                print(f'Loading trajectories from {load_from}')
            trajectories = np.load(load_from)
            if trajectories.shape[1:] != self._trajectories.shape[1:]:
                raise RuntimeError(f'Loaded old trajectories with mismatching shape (old/new {trajectories.shape}/{self._trajectories.shape})')
            n_seed_trajectories = min(trajectories.shape[0], self._max_trajectories)
            if trajectories.shape[0] != self._trajectories.shape[0]:
                if not silent:
                    print(f'Attempted to load {trajectories.shape[0]} offline trajectories into buffer of size {self._trajectories.shape[0]}.' \
                          f'Loading only {n_seed_trajectories} trajectories from offline buffer')
            self._trajectories[:n_seed_trajectories] = trajectories[:n_seed_trajectories]
            self._stored_trajectories = n_seed_trajectories
            self._new_trajectory_idx = n_seed_trajectories % self._max_trajectories

    def __len__(self):
        return self._stored_trajectories

    def save(self, location: str):
        np.save(location, self._trajectories[:self._stored_trajectories])
    
    def add_trajectory(self, trajectory: List[Experience], force: bool = False):
        if self._immutable and not force:
            raise ValueError('Cannot add trajectory to immutable replay buffer')

        #if len(trajectory) != self._trajectory_length:
        #    raise ValueError(f'Invalid trajectory length: {len(trajectory)}')

        mc_reward = 0
        terminal_state = None
        terminal_factor = 1
        missing_elements = self._trajectory_length - len(trajectory)
        for idx, experience in enumerate(trajectory[::-1]):
            if terminal_state is None:
                terminal_state = experience.next_state

            idx = idx + missing_elements
            slice_idx = 0
            self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + self._state_dim] = experience.state
            slice_idx += self._state_dim

            self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + self._action_dim] = experience.action
            slice_idx += self._action_dim

            self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + self._state_dim] = experience.next_state
            slice_idx += self._state_dim
            
            self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + self._state_dim] = terminal_state
            slice_idx += self._state_dim

            if experience.info is not None:
                self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + self._info_dim] = experience.info
                slice_idx += self._info_dim

                self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + self._info_dim] = experience.next_info
                slice_idx += self._info_dim

            #self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + 1] = experience.log_prob
            #slice_idx += 1
            
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

    def add_trajectories(self, trajectories: List[List[Experience]], force: bool = False):
        for trajectory in trajectories:
            self.add_trajectory(trajectory, force)

    def sample(self, batch_size, trajectory: bool = False, complete: bool = False):
        valid = (~np.isnan(self._trajectories)).all(-1)
        all_trajectory_idxs, all_time_steps = np.where(valid)
        idxs = np.random.choice(all_trajectory_idxs.shape[0], batch_size)
        trajectory_idxs = all_trajectory_idxs[idxs]
        time_steps = all_time_steps[idxs]
        #idxs = np.random.choice(np.arange(self._stored_trajectories * (self._trajectory_length - self._trim_suffix)), batch_size)
        #trajectory_idxs = idxs // (self._trajectory_length - self._trim_suffix)
        #time_steps = idxs % (self._trajectory_length - self._trim_suffix)

        batch = self._trajectories[trajectory_idxs, time_steps]
        if not trajectory:
            return batch
        else:
            if complete:
                if self._trim_suffix > 0:
                    trajectories = self._trajectories[trajectory_idxs,:-self._trim_suffix]
                else:
                    trajectories = self._trajectories[trajectory_idxs]
            else:
                trajectories = [self._trajectories[traj_idx, :time_step+1] for traj_idx, time_step in zip(trajectory_idxs, time_steps)]
            return batch, trajectories


def generate_test_trajectory(state_dim: int, action_dim: int, trajectory_length: int):
    trajectory = []
    next_state = np.random.uniform(0,1,(state_dim,))
    for idx in range(trajectory_length):
        state = next_state
        action = np.random.uniform(-1,0,(action_dim,))
        next_state = np.random.uniform(0,1,(state_dim,))
        reward = np.random.uniform()
        trajectory.append(Experience(state, action, next_state, reward, idx == trajectory_length - 1))

    return trajectory

if __name__ == '__main__':    
    trajectory_length = 20
    state, action = 6, 4
    buf = ReplayBuffer(trajectory_length, state, action, max_trajectories=5)

    for idx in range(2):
        buf.add_trajectory(generate_test_trajectory(state, action, 20))

    buf.add_trajectories([generate_test_trajectory(state, action, 10) for _ in range(2)])

    print(len(buf))
    print(buf.sample(2))
    import pdb; pdb.set_trace()
