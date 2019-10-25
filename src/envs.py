#
# Toy environments for testing maml-rawr
#
import gym
from gym.spaces import Box
import imageio
import numpy as np
from typing import Optional, Tuple, List
from gym.envs.mujoco import HalfCheetahEnv as HalfCheetahEnv_

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

'''
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

    def render(self, *args):
        return self.env.render(*args)
    
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
'''


class HalfCheetahEnv(HalfCheetahEnv_):
    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).astype(np.float32).flatten()

    def viewer_setup(self):
        camera_id = self.model.camera_name2id('track')
        self.viewer.cam.type = 2
        self.viewer.cam.fixedcamid = camera_id
        self.viewer.cam.distance = self.model.stat.extent * 0.35
        # Hide the overlay
        self.viewer._hide_overlay = True

    def render(self, mode='human'):
        if mode == 'rgb_array':
            self._get_viewer().render()
            # window size used for old mujoco-py:
            width, height = 500, 500
            data = self._get_viewer(mode).read_pixels(width, height, depth=False)
            return data
        elif mode == 'human':
            self._get_viewer(mode).render()

class HalfCheetahVelEnv(HalfCheetahEnv):
    """Half-cheetah environment with target velocity, as described in [1]. The 
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand.py
    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each 
    time step a reward composed of a control cost and a penalty equal to the 
    difference between its current velocity and the target velocity. The tasks 
    are generated by sampling the target velocities from the uniform 
    distribution on [0, 2].
    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
        model-based control", 2012 
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """
    def __init__(self, task={}):
        self._task = task
        self._goal_vel = task.get('velocity', 0.0)
        super(HalfCheetahVelEnv, self).__init__()

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = -1.0 * abs(forward_vel - self._goal_vel)
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks):
        velocities = self.np_random.uniform(0.0, 2.0, size=(num_tasks,))
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def reset_task(self, task):
        self._task = task
        self._goal_vel = task['velocity']


class HalfCheetahDirEnv(HalfCheetahEnv):
    """Half-cheetah environment with target direction, as described in [1]. The 
    code is adapted from
    https://github.com/cbfinn/maml_rl/blob/9c8e2ebd741cb0c7b8bf2d040c4caeeb8e06cc95/rllab/envs/mujoco/half_cheetah_env_rand_direc.py
    The half-cheetah follows the dynamics from MuJoCo [2], and receives at each 
    time step a reward composed of a control cost and a reward equal to its 
    velocity in the target direction. The tasks are generated by sampling the 
    target directions from a Bernoulli distribution on {-1, 1} with parameter 
    0.5 (-1: backward, +1: forward).
    [1] Chelsea Finn, Pieter Abbeel, Sergey Levine, "Model-Agnostic 
        Meta-Learning for Fast Adaptation of Deep Networks", 2017 
        (https://arxiv.org/abs/1703.03400)
    [2] Emanuel Todorov, Tom Erez, Yuval Tassa, "MuJoCo: A physics engine for 
        model-based control", 2012 
        (https://homes.cs.washington.edu/~todorov/papers/TodorovIROS12.pdf)
    """
    def __init__(self, task_idx: Optional[int] = None, fix_random_task = False):
        if task_idx is None:
            task_idx = np.random.randint(2)
        self._task = task_idx
        self._goal_dir = [1,-1][task_idx]
        super(HalfCheetahDirEnv, self).__init__()
        self._max_episode_steps = 1000
        
    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]

        forward_vel = (xposafter - xposbefore) / self.dt
        forward_reward = self._goal_dir * forward_vel
        ctrl_cost = 0.5 * 1e-1 * np.sum(np.square(action))

        observation = self._get_obs()
        reward = forward_reward - ctrl_cost
        done = False
        infos = dict(reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost, task=self._task)
        return (observation, reward, done, infos)
