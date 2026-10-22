# -*- coding: utf-8 -*-
"""Discrete action space shared by the world, policy and benchmarks."""
from ..utils.random import stable_int

DEFAULT_TOKENS = ["step", "grasp", "push", "pull", "rotate", "insert", "place", "align"]


class ActionSpace:
    """A finite set of symbolic actions.

    The answer world assigns every ``(task_id, step)`` a single ground-truth
    action token; a policy either reproduces it (correct) or not (wrong)."""

    def __init__(self, tokens=None, seed=20261022):
        self.tokens = list(tokens) if tokens else DEFAULT_TOKENS
        self.seed = seed

    def __len__(self):
        return len(self.tokens)

    def __contains__(self, token):
        return token in self.tokens

    def ground_truth(self, task_id, step):
        """The correct action for ``task_id`` at ``step`` (deterministic)."""
        idx = stable_int("gt:%s:%d" % (task_id, step), 0, len(self.tokens))
        return self.tokens[idx]

    def random_wrong(self, task_id, step):
        """A wrong action for ``task_id`` at ``step`` (deterministic)."""
        gt = self.ground_truth(task_id, step)
        others = [t for t in self.tokens if t != gt]
        idx = stable_int("wrong:%s:%d" % (task_id, step), 0, len(others))
        return others[idx]
