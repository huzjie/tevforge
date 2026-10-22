# -*- coding: utf-8 -*-
"""Dense (progress-only) reward, without the goal bonus.

Useful for ablations: it credits every correct step with +1/L, giving maximal
step density, at the cost of no explicit completion bonus.
"""


class DenseReward:
    def __init__(self):
        pass

    def step_reward(self, world, task_id, prefix, action):
        after = prefix + [action]
        before = world.progress(task_id, prefix)
        after_p = world.progress(task_id, after)
        return after_p - before

    def return_(self, world, task_id, actions):
        return world.progress(task_id, actions)
