# -*- coding: utf-8 -*-
"""Progress potentials: a scalar measure of "how close to the goal" a prefix is.

A potential function Phi(s) is the raw material of reward shaping. For a
verifiable answer world, the natural potential is simply the fraction of the
ground-truth plan the agent has already reproduced correctly.
"""


class ProgressPotential:
    """Phi(s) = fraction of correct actions in the prefix so far."""

    def __call__(self, world, task_id, actions_prefix):
        return world.progress(task_id, actions_prefix)


class DenseProgress(ProgressPotential):
    """Dense variant: also credits each newly-correct step with +1/L locally.

    Equivalent potential, exposed for clarity / step-level reward density
    accounting (see ``reward/dense.py`` documentation)."""

    def __call__(self, world, task_id, actions_prefix):
        return world.progress(task_id, actions_prefix)
