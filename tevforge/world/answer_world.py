# -*- coding: utf-8 -*-
"""The latent "answer world" that grounds every long-horizon task.

Tev's key idea is "answer-world-first" data synthesis: before you can train a
long-horizon agent you need tasks whose *answer* (the sequence of correct
actions) is known and whose *progress* can be measured at every step. This
class is that world: a deterministic, verifiable source of truth.
"""
from .action import ActionSpace
from ..utils.random import stable_int


class AnswerWorld:
    """Deterministic world model with a verifiable completion predicate.

    A task is identified by a string ``task_id``. Its horizon ``L`` and the
    ground-truth action at every step are derived deterministically from the
    id, so the same id always yields the same task. ``progress`` counts how
    many of the agent's actions so far match the ground truth; ``is_complete``
    is true only when the whole plan matches.
    """

    def __init__(self, action_space=None, horizon_min=8, horizon_max=24, seed=20261022):
        self.actions = action_space or ActionSpace(seed=seed)
        self.horizon_min = horizon_min
        self.horizon_max = horizon_max
        self.seed = seed

    def n_steps(self, task_id):
        """Horizon L for this task (deterministic, in [horizon_min, horizon_max))."""
        return stable_int("len:%s" % task_id, self.horizon_min, self.horizon_max)

    def ground_truth(self, task_id, step):
        return self.actions.ground_truth(task_id, step)

    def progress(self, task_id, actions):
        """Fraction of ``actions`` (a prefix of the plan) that are correct, in [0,1]."""
        if not actions:
            return 0.0
        correct = sum(1 for t, a in enumerate(actions)
                      if a == self.ground_truth(task_id, t))
        return correct / self.n_steps(task_id)

    def correct_count(self, task_id, actions):
        return sum(1 for t, a in enumerate(actions)
                   if a == self.ground_truth(task_id, t))

    def is_complete(self, task_id, actions):
        """True iff the agent reproduced the entire ground-truth plan."""
        L = self.n_steps(task_id)
        if len(actions) < L:
            return False
        return all(a == self.ground_truth(task_id, t) for t, a in enumerate(actions[:L]))
