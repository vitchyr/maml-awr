import numpy as np

from . import register_env
from src.tp_envs.half_cheetah import HalfCheetahEnv


@register_env('cheetah-vel')
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
    def __init__(self, tasks=None, randomize_tasks=True):
        if tasks is None:
            tasks = [{}]
        self.tasks = tasks
        self._task = self.tasks[0]
        self._goal_vel = self._task.get('velocity', 0.0)
        self._goal = self._goal_vel
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
        infos = dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            goal_vel=self._goal_vel,
            forward_vel=forward_vel,
            xposbefore=xposbefore,
        )
        return (observation, reward, done, infos)

    def sample_tasks(self, num_tasks, seed: int = 1337):
        np.random.seed(seed)
        #velocities = np.random.uniform(0.0, 3.0, size=(num_tasks,))
        velocities = np.linspace(0.075,3,40)
        tasks = [{'velocity': velocity} for velocity in velocities]
        return tasks

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal_vel = self._task['velocity']
        self._goal = self._goal_vel
        self.reset()
