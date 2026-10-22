# -*- coding: utf-8 -*-
"""Sparse reward: +1 only when the whole task is completed, else 0.

This is the reward the field has been using and the reason long-horizon GRPO
wastes so many rollouts: a group of G rollouts that all fail yields G identical
zero rewards, so the advantage is zero everywhere and the group contributes no
gradient.
"""


class SparseReward:
    def __init__(self, goal_bonus=1.0):
        self.goal_bonus = goal_bonus

    def return_(self, world, task_id, actions):
        return self.goal_bonus if world.is_complete(task_id, actions) else 0.0

    def step_reward(self, world, task_id, prefix, action):
        """Sparse step reward: only the final, completing action earns bonus."""
        after = prefix + [action]
        if len(after) == world.n_steps(task_id) and world.is_complete(task_id, after):
            return self.goal_bonus
        return 0.0
