# -*- coding: utf-8 -*-
"""Group advantage estimation for GRPO.

GRPO normalizes rewards *within a group* of G rollouts of the same task. If
rewards are sparse and the group all fails, every reward is 0, the standard
deviation is 0, and all advantages collapse to 0: the group contributes no
gradient. Dense shaped rewards make the returns spread out, so even a group
with no completions carries signal.
"""
from ..utils.metrics import normalize


def grpo_advantages(rewards, eps=1e-9):
    """Return group advantages ``(r_i - mean) / std``.

    A constant reward vector (e.g. all-zero sparse failures) maps to all-zero
    advantages, which is exactly the "wasted group" case."""
    return normalize(rewards, eps=eps)


def wasted_group(rewards, eps=1e-9):
    """True iff this group yields no gradient (all advantages ~ 0)."""
    adv = grpo_advantages(rewards, eps=eps)
    return all(abs(a) < eps for a in adv)
