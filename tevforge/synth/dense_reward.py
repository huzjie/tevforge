# -*- coding: utf-8 -*-
"""Dense reward annotation of synthetic tasks.

Given an answer world and a task, annotate every ``(task_id, step)`` with the
progress it would earn, so a task arrives pre-labelled with a dense reward
signal (the "progressive" part of Tev-Gym).
"""
from ..reward.pprs import PPRS


class DenseAnnotator:
    def __init__(self, pprs=None):
        self.pprs = pprs or PPRS()

    def annotate(self, world, task_id):
        """Return the per-step shaped-reward table for the ground-truth plan.

        The ground-truth plan is always correct, so its progress is 1/L, 2/L,
        ... and its shaped step reward is +1/L each step plus the goal bonus.
        """
        L = world.n_steps(task_id)
        table = []
        prefix = []
        for step in range(L):
            gt = world.ground_truth(task_id, step)
            r = self.pprs.step_reward(world, task_id, prefix, gt)
            table.append({"step": step, "action": gt, "reward": r})
            prefix.append(gt)
        return table
