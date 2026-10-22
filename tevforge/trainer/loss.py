# -*- coding: utf-8 -*-
"""Policy-gradient loss over group advantages.

The reference implementation keeps the gradient math explicit: the policy
gradient is the average over the group of ``advantage_i * log p_i``. The mock
policy updates its scalar ``skill`` via a monotonic relaxation of this
gradient (see ``grpo.py``), which keeps training deterministic and analysable.
"""


def policy_gradient_loss(log_probs, advantages):
    """Return the (negative) expected reward loss = -mean(a_i * log p_i)."""
    if not log_probs:
        return 0.0
    return -sum(a * lp for a, lp in zip(advantages, log_probs)) / len(log_probs)


def group_log_probs(skill, n_correct_per_rollout, horizon):
    """Deterministic log-likelihood of each rollout under a skill-s policy.

    A rollout with ``c`` correct steps has log-prob ``c*ln(s) + (L-c)*ln(1-s)``.
    """
    import math
    out = []
    for c in n_correct_per_rollout:
        s = max(min(skill, 1.0 - 1e-9), 1e-9)
        out.append(c * math.log(s) + (horizon - c) * math.log(1.0 - s))
    return out
