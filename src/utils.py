from typing import NamedTuple, List

import h5py
import numpy as np
import tempfile
import torch
import torch.nn as nn
import os
import random


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


class NewReplayBuffer(object):
    def __init__(self, size: int, obs_dim: int, action_dim: int, discount_factor: float = 0.99,
                 immutable: bool = False, load_from: str = None, silent: bool = False, skip: int = 1,
                 stream_to_disk: bool = False):
        if size == -1 and load_from is None:
            print("Can't have size == -1 and no offline buffer - defaulting to 1M steps")
            size = 1000000

        self.immutable = immutable
        self.stream_to_disk = stream_to_disk
        
        if load_from is not None:
            f = h5py.File(load_from, 'r')
            if size == -1:
                size = f['obs'].shape[0]
        
        needs_to_load = True
        size //= skip
        if stream_to_disk:
            name = os.path.splitext(os.path.basename(os.path.normpath(load_from)))[0]
            if os.path.exists('/scr-ssd'):
                path = f'/scr-ssd/em7/{name}'
            else:
                path = f'/scr/em7/{name}'
            if os.path.exists(path):
                if not silent:
                    print(f'Using existing replay buffer memmap at {path}')
                needs_to_load = False
                self._obs = np.memmap(f'{path}/obs.array', mode='r', shape=(size, obs_dim), dtype=np.float32)
                self._actions = np.memmap(f'{path}/actions.array', mode='r', shape=(size, action_dim), dtype=np.float32)
                self._rewards = np.memmap(f'{path}/rewards.array', mode='r', shape=(size, 1), dtype=np.float32)
                self._mc_rewards = np.memmap(f'{path}/mc_rewards.array', mode='r', shape=(size, 1), dtype=np.float32)
                self._terminals = np.memmap(f'{path}/terminals.array', mode='r', shape=(size, 1), dtype=np.bool)
                self._terminal_obs = np.memmap(f'{path}/terminal_obs.array', mode='r', shape=(size, obs_dim), dtype=np.float32)
                self._terminal_discounts = np.memmap(f'{path}/terminal_discounts.array', mode='r', shape=(size, 1), dtype=np.float32)
                self._next_obs = np.memmap(f'{path}/next_obs.array', mode='r', shape=(size, obs_dim), dtype=np.float32)
            else:
                if not silent:
                    print(f'Creating replay buffer memmap at {path}')
                os.makedirs(path)
                self._obs = np.memmap(f'{path}/obs.array', mode='w+', shape=(size, obs_dim), dtype=np.float32)
                self._actions = np.memmap(f'{path}/actions.array', mode='w+', shape=(size, action_dim), dtype=np.float32)
                self._rewards = np.memmap(f'{path}/rewards.array', mode='w+', shape=(size, 1), dtype=np.float32)
                self._mc_rewards = np.memmap(f'{path}/mc_rewards.array', mode='w+', shape=(size, 1), dtype=np.float32)
                self._terminals = np.memmap(f'{path}/terminals.array', mode='w+', shape=(size, 1), dtype=np.bool)
                self._terminal_obs = np.memmap(f'{path}/terminal_obs.array', mode='w+', shape=(size, obs_dim), dtype=np.float32)
                self._terminal_discounts = np.memmap(f'{path}/terminal_discounts.array', mode='w+', shape=(size, 1), dtype=np.float32)
                self._next_obs = np.memmap(f'{path}/next_obs.array', mode='w+', shape=(size, obs_dim), dtype=np.float32)
                self._obs.fill(float('nan'))
                self._actions.fill(float('nan'))
                self._rewards.fill(float('nan'))
                self._mc_rewards.fill(float('nan'))
                self._terminals.fill(float('nan'))
                self._terminal_obs.fill(float('nan'))
                self._terminal_discounts.fill(float('nan'))
                self._next_obs.fill(float('nan'))
        else:
            self._obs = np.full((size, obs_dim), float('nan'), dtype=np.float32)
            self._actions = np.full((size, action_dim), float('nan'), dtype=np.float32)
            self._rewards = np.full((size, 1), float('nan'), dtype=np.float32)
            self._mc_rewards = np.full((size, 1), float('nan'), dtype=np.float32)
            self._terminals = np.full((size, 1), False, dtype=np.bool)
            self._terminal_obs = np.full((size, obs_dim), float('nan'), dtype=np.float32)
            self._terminal_discounts = np.full((size, 1), float('nan'), dtype=np.float32)
            self._next_obs = np.full((size, obs_dim), float('nan'), dtype=np.float32)

        self._size = size
        if load_from is None:
            self._stored_steps = 0
            self._discount_factor = discount_factor
        else:
            if f['obs'].shape[-1] != self.obs_dim:
                raise RuntimeError(f"Loaded data has different obs_dim from new buffer ({f['obs'].shape[-1]}, {self.obs_dim})")
            if f['actions'].shape[-1] != self.action_dim:
                raise RuntimeError(f"Loaded data has different action_dim from new buffer ({f['actions'].shape[-1]}, {self.action_dim})")

            stored = f['obs'].shape[0]
            n_seed = min(stored, self._size * skip)
            self._stored_steps = n_seed // skip

            if needs_to_load:
                if stored > self._size * skip:
                    if not silent:
                        print(f"Attempted to load {stored} offline steps into buffer of size {self._size}.")
                        print(f"Loading only the last {n_seed//skip} steps from offline buffer")
                if not silent:
                    print(f'Loading trajectories from {load_from}')

                self._discount_factor = f['discount_factor'][()]
                self._obs[:self._stored_steps] = f['obs'][-n_seed + int(skip > 1):][::skip]
                self._actions[:self._stored_steps] = f['actions'][-n_seed + int(skip > 1):][::skip]
                self._rewards[:self._stored_steps] = f['rewards'][-n_seed + int(skip > 1):][::skip]
                self._mc_rewards[:self._stored_steps] = f['mc_rewards'][-n_seed + int(skip > 1):][::skip]
                self._terminals[:self._stored_steps] = f['terminals'][-n_seed + int(skip > 1):][::skip]
                self._terminal_obs[:self._stored_steps] = f['terminal_obs'][-n_seed + int(skip > 1):][::skip]
                self._terminal_discounts[:self._stored_steps] = f['terminal_discounts'][-n_seed + int(skip > 1):][::skip]
                self._next_obs[:self._stored_steps] = f['next_obs'][-n_seed + int(skip > 1):][::skip]

            f.close()

        self._write_location = self._stored_steps % self._size
        self._valid = np.where(np.logical_and(~np.isnan(self._terminal_discounts[:,0]), self._terminal_discounts[:,0] < 0.35))[0]

    @property
    def obs_dim(self):
        return self._obs.shape[-1]

    @property
    def action_dim(self):
        return self._actions.shape[-1]

    def __len__(self):
        return self._stored_steps

    def save(self, location: str):
        f = h5py.File(location, 'w')
        f.create_dataset('obs', data=self._obs[:self._stored_steps], compression='lzf')
        f.create_dataset('actions', data=self._actions[:self._stored_steps], compression='lzf')
        f.create_dataset('rewards', data=self._rewards[:self._stored_steps], compression='lzf')
        f.create_dataset('mc_rewards', data=self._mc_rewards[:self._stored_steps], compression='lzf')
        f.create_dataset('terminals', data=self._terminals[:self._stored_steps], compression='lzf')
        f.create_dataset('terminal_obs', data=self._terminal_obs[:self._stored_steps], compression='lzf')
        f.create_dataset('terminal_discounts', data=self._terminal_discounts[:self._stored_steps], compression='lzf')
        f.create_dataset('next_obs', data=self._next_obs[:self._stored_steps], compression='lzf')
        f.create_dataset('discount_factor', data=self._discount_factor)
        f.close()
    
    def add_trajectory(self, trajectory: List[Experience], force: bool = False):
        if self.immutable and not force:
            raise ValueError('Cannot add trajectory to immutable replay buffer')

        mc_reward = 0
        terminal_obs = None
        terminal_factor = 1
        for idx, experience in enumerate(trajectory[::-1]):
            if terminal_obs is None:
                terminal_obs = experience.next_state

            self._obs[self._write_location] = experience.state
            self._next_obs[self._write_location] = experience.next_state
            self._actions[self._write_location] = experience.action
            self._rewards[self._write_location] = experience.reward
            self._terminals[self._write_location] = experience.done
            self._terminal_obs[self._write_location] = terminal_obs

            terminal_factor *= self._discount_factor
            self._terminal_discounts[self._write_location] = terminal_factor

            mc_reward = experience.reward + self._discount_factor * mc_reward
            self._mc_rewards[self._write_location] = mc_reward

            self._write_location += 1
            self._write_location = self._write_location % self._size
            
            if self._stored_steps < self._size:
                self._stored_steps += 1

        self._valid = np.where(np.logical_and(~np.isnan(self._terminal_discounts[:,0]), self._terminal_discounts[:,0] < 0.35))[0]

    def add_trajectories(self, trajectories: List[List[Experience]], force: bool = False):
        for trajectory in trajectories:
            self.add_trajectory(trajectory, force)

    def sample(self, batch_size, return_dict: bool = False, noise: bool = False):
        idxs = np.array(random.sample(range(self._stored_steps), batch_size))
        #idxs = np.random.choice(self._valid, batch_size)

        obs = self._obs[idxs]
        actions = self._actions[idxs]
        next_obs = self._next_obs[idxs]
        terminal_obs = self._terminal_obs[idxs]
        terminal_discounts = self._terminal_discounts[idxs]
        dones = self._terminals[idxs]
        rewards = self._rewards[idxs]
        mc_rewards = self._mc_rewards[idxs]
        
        if not return_dict:
            batch = np.concatenate((obs, actions, next_obs, terminal_obs, terminal_discounts, dones, rewards, mc_rewards), 1)
            if noise:
                std = batch.std(0) * np.sqrt(batch_size)
                mu = np.zeros(std.shape)
                noise = np.random.normal(mu, std, batch.shape).astype(np.float32)
                batch = batch + noise
            return batch
        else:
            return {
                'obs': obs,
                'actions': actions,
                'next_obs': next_obs,
                'terminal_obs': terminal_obs,
                'terminal_discounts': terminal_discounts,
                'dones': dones,
                'rewards': rewards,
                'mc_rewards': mc_rewards
            }


class ReplayBuffer(object):
    @staticmethod
    def join(buffers: List['ReplayBuffer']) -> 'ReplayBuffer':
        b0 = buffers[0]
        trajectories = np.concatenate([b._trajectories for b in buffers])
        new_buffer = ReplayBuffer(trajectories.shape[0], b0._state_dim, b0._action_dim, b0._info_dim, b0._max_trajectories,
                                  b0._discount_factor, b0._immutable, b0._trim_suffix)
        new_buffer._trajectories = trajectories
        new_buffer._stored_trajectories = trajectories.shape[0]

        return new_buffer

    def __init__(self, trajectory_length: int, state_dim: int, action_dim: int, info_dim: int = 0, max_trajectories: int = 10000,
                 discount_factor: float = 0.99, immutable: bool = False, load_from: str = None, silent: bool = False, trim_suffix: int = 0,
                 trim_obs: int = None, pad: bool = False):
        self._trajectories = np.zeros((max_trajectories, trajectory_length, state_dim + action_dim + state_dim + state_dim + info_dim * 2 + 1 + 1 + 1 + 1 + int(pad)), dtype=np.float32)
        #self._trajectories.fill(np.float('nan'))
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
        self._trim_obs = trim_obs
        self.has_nan = False
        if load_from is not None:
            if not silent:
                print(f'Loading trajectories from {load_from}')
            trajectories = np.load(load_from)

            if trajectories.shape[1:] != self._trajectories.shape[1:]:
                raise RuntimeError(f'Loaded old trajectories with mismatching shape [do you need to pad?] (old/new {trajectories.shape}/{self._trajectories.shape})')
            n_seed_trajectories = min(trajectories.shape[0], self._max_trajectories)
            if trajectories.shape[0] != self._trajectories.shape[0]:
                if not silent:
                    print(f'Attempted to load {trajectories.shape[0]} offline trajectories into buffer of size {self._trajectories.shape[0]}.' \
                          f'Loading only {n_seed_trajectories} trajectories from offline buffer')
            self._trajectories[:n_seed_trajectories] = trajectories[:n_seed_trajectories]
            self._stored_trajectories = n_seed_trajectories
            self._new_trajectory_idx = n_seed_trajectories % self._max_trajectories
            self.has_nan = np.isnan(self._trajectories).any()
            if (self._trajectories[:,:,-1] == 0).all():
                print('chopping off zeros')
                self._trajectories = self._trajectories[:,:,:-1]

            if self._immutable:
                valid = (~np.isnan(self._trajectories)).all(-1)
                self.all_traj_idxs, self.all_time_steps = np.where(valid)

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
        missing_elements = self._trajectory_length - len(trajectory)
        if missing_elements > 0:
            self.has_nan = True
        terminal_factor = 1 if missing_elements == 0 else 0 # For incomplete trajectories, we don't want any bootstrap value estimation
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

            #self._trajectories[self._new_trajectory_idx, -(idx + 1), slice_idx:slice_idx + 1] = 0experience.log_prob
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

    def sample(self, batch_size, trajectory: bool = False, complete: bool = False, train: bool = None):
        if self.has_nan:
            if self.all_traj_idxs is None:
                valid = (~np.isnan(self._trajectories)).all(-1)
                all_trajectory_idxs, all_time_steps = np.where(valid)
            else:
                all_trajectory_idxs, all_time_steps = self.all_traj_idxs, self.all_time_steps
            idxs = np.random.choice(all_trajectory_idxs.shape[0], batch_size)
            trajectory_idxs = all_trajectory_idxs[idxs]
            time_steps = all_time_steps[idxs]
        else:
            idxs = np.random.choice(np.arange(self._stored_trajectories * (self._trajectory_length - self._trim_suffix)), batch_size)
            trajectory_idxs = idxs // (self._trajectory_length - self._trim_suffix)
            time_steps = idxs % (self._trajectory_length - self._trim_suffix)

        if train is not None:    
            if train:
                odd = trajectory_idxs % 2 == 1
                trajectory_idxs[odd] = trajectory_idxs[odd] - 1
            else:
                even = trajectory_idxs % 2 == 0
                trajectory_idxs[even] = trajectory_idxs[even] - 1
                trajectory_idxs[trajectory_idxs < 0] = 1

        batch = self._trajectories[trajectory_idxs, time_steps]
        if not trajectory:
            if self._trim_obs is not None:
                batch[:,self._state_dim-self._trim_obs:self._state_dim] = 0
                batch[:,self._state_dim+self._action_dim+self._state_dim-self._trim_obs:self._state_dim+self._action_dim+self._state_dim] = 0
                batch[:,self._state_dim+self._action_dim+self._state_dim*2-self._trim_obs:self._state_dim+self._action_dim+self._state_dim*2] = 0
            if np.isnan(batch).any():
                import pdb; pdb.set_trace()
            assert not np.isnan(batch).any()
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


def generate_test_trajectory(length: int, state_dim: int, action_dim: int):
    trajectory = []
    next_state = np.random.uniform(0,1,(state_dim,))
    for idx in range(length):
        state = next_state
        action = np.random.uniform(-1,0,(action_dim,))
        next_state = np.random.uniform(0,1,(state_dim,))
        reward = np.random.uniform()
        trajectory.append(Experience(state, action, next_state, reward, idx == length - 1))

    return trajectory


def test_old_buffer():
    trajectory_length = 100
    state, action = 6, 4
    buf = ReplayBuffer(trajectory_length, state, action, max_trajectories=5)

    for idx in range(2):
        buf.add_trajectory(generate_test_trajectory(20, state, action))

    buf.add_trajectories([generate_test_trajectory(10, state, action) for _ in range(2)])

    print(len(buf))
    print(buf.sample(2))
    import pdb; pdb.set_trace()


def test_new_buffer():
    np.random.seed(0)
    size = 100000000
    state, action = 20, 6
    buf = NewReplayBuffer(size, state, action, stream_to_disk=True)

    t1 = generate_test_trajectory(3, state, action)
    buf.add_trajectory(t1)

    t2 = [generate_test_trajectory(3, state, action) for _ in range(4)]
    buf.add_trajectories(t2)

    print(len(buf))
    print('sample', buf.sample(20000).shape)

    buf.save('test_buf.h5')
    #buf2 = NewReplayBuffer(size, state, action, load_from='test_buf.h5')
    import pdb; pdb.set_trace()
    

if __name__ == '__main__':
    test_new_buffer()
