# -*- coding: utf-8 -*-
"""Progress Potential Reward Shaping (PPRS).

The sparse reward is replaced by a *potential-based* dense reward::

    r' = r_env + gamma * Phi(s') - Phi(s)

where Phi is a progress potential. Because it is potential-based, the shaped
MDP has the same optimal policy as the original (policy-invariance), but every
step that advances progress now emits non-zero reward. This is what turns a
"whole group fails -> zero advantage" scenario into a dense, informative
gradient.

Notes
-----
* With ``gamma == 1`` and ``Phi(empty) == 0``, the shaped return telescopes to
  ``sparse_return + Phi(s_T)``, i.e. final progress is always credited.
* We keep the goal bonus so completion is still the dominant term.
"""
from .potential import ProgressPotential


class PPRS:
    def __init__(self, gamma=1.0, goal_bonus=1.0, potential=None):
        self.gamma = gamma
        self.goal_bonus = goal_bonus
        self.potential = potential or ProgressPotential()

    def step_reward(self, world, task_id, prefix, action):
        """Dense potential-based reward for taking ``action`` after ``prefix``."""
        before = self.potential(world, task_id, prefix)
        after = self.potential(world, task_id, prefix + [action])
        r = self.gamma * after - before
        if len(prefix) + 1 == world.n_steps(task_id) and world.is_complete(task_id, prefix + [action]):
            r += self.goal_bonus
        return r

    def return_(self, world, task_id, actions):
        """Shaped episode return = sparse return + telescoped potential."""
        base = self.goal_bonus if world.is_complete(task_id, actions) else 0.0
        return base + self.potential(world, task_id, actions)
