# -*- coding: utf-8 -*-
"""Policies: the mock policy is trainable and fully deterministic.

``skill`` is the probability a policy reproduces the ground-truth action.
``skill = 0.5`` is a random walk (completion ~ 0.5**L ~ 0); ``skill = 1.0`` is
the perfect policy (completion = 1.0). The boost convention
``boost = (skill - 0.5) * 2`` keeps skill=0.5 at a clean random baseline and
skill=1.0 at a perfect ceiling.
"""
from ..utils.random import stable_bool


class Policy:
    """Abstract policy: map (task_id, step, history) -> action."""

    def act(self, world, task_id, step, history, salt=""):
        raise NotImplementedError


class MockPolicy(Policy):
    """Deterministic mock policy driven by a scalar ``skill``."""

    def __init__(self, skill=0.5, seed=20261022):
        self.skill = skill
        self.seed = seed

    def act(self, world, task_id, step, history, salt=""):
        gt = world.ground_truth(task_id, step)
        key = "act:%s:%d:%s" % (task_id, step, salt)
        if stable_bool(key, p=self.skill):
            return gt
        return world.actions.random_wrong(task_id, step)
