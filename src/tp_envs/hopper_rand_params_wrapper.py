import numpy as np
from typing import List, Dict
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv

from . import register_env


@register_env('hopper-rand-params')
class HopperRandParamsWrappedEnv(HopperRandParamsEnv):
    def __init__(self, tasks: List[Dict] = None, n_tasks: int = None, randomize_tasks=True):
        super(HopperRandParamsWrappedEnv, self).__init__()
        if tasks is None and n_tasks is None:
            raise Exception("Either tasks or n_tasks must be specified")

        if tasks is not None:
            self.tasks = tasks
        else:
            self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def reset_task(self, idx):
        self._task = self.tasks[idx]
        self._goal = idx
        self.set_task(self._task)
        self.reset()
