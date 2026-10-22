# -*- coding: utf-8 -*-
"""Heldout evaluation and reward-hacking diagnostics.

Because PPRS is potential-based, the shaped return equals
``progress + completion_bonus`` exactly. ``progress_reward_gap`` verifies this
identity: a gap near zero means the reward is *not* hacked (it tracks genuine
progress), which is the property a long-horizon pipeline must preserve.
"""
from ..world.answer_world import AnswerWorld
from ..agent.policy import MockPolicy
from ..agent.rollout import rollout
from ..reward.pprs import PPRS
from ..reward.sparse import SparseReward
from ..utils.metrics import mean


def evaluate_heldout(world, policy, task_ids, n_rollouts=8):
    """Completion rate, mean return and mean progress over heldout tasks."""
    comps, returns, progs = [], [], []
    pprs = PPRS()
    for tid in task_ids:
        for i in range(n_rollouts):
            traj = rollout(policy, world, tid, pprs.step_reward, salt="ev%d" % i)
            comps.append(1 if traj.completed else 0)
            returns.append(pprs.return_(world, tid, traj.actions))
            progs.append(traj.final_progress)
    return {
        "completion_rate": mean(comps),
        "mean_return": mean(returns),
        "mean_progress": mean(progs),
        "n_rollouts": len(comps),
    }


def progress_reward_gap(world, policy, task_ids, n_rollouts=8):
    """Mean |return - (progress + completion_bonus)| — should be ~0 for PPRS."""
    pprs = PPRS()
    gaps = []
    for tid in task_ids:
        for i in range(n_rollouts):
            traj = rollout(policy, world, tid, pprs.step_reward, salt="gp%d" % i)
            r = pprs.return_(world, tid, traj.actions)
            expected = traj.final_progress + (1.0 if traj.completed else 0.0)
            gaps.append(abs(r - expected))
    return mean(gaps)


def reward_hacking_audit(world, task_ids, n_rollouts=8):
    """Contrast a hacking policy vs a genuine policy.

    A "hacked" policy (fixed skill 0.0, i.e. never progresses) would still earn
    zero return; a policy that blindly repeats one action also earns zero
    progress. The audit reports return and progress for a skill=0.0 policy, a
    skill=0.5 policy and a skill=1.0 policy, making it obvious that the reward
    cannot be gained without genuine progress.
    """
    out = {}
    for skill in (0.0, 0.5, 1.0):
        policy = MockPolicy(skill=skill)
        out[skill] = evaluate_heldout(world, policy, task_ids, n_rollouts)
    return out


def completion_vs_skill(world, task_ids, n_rollouts=8):
    """Completion and progress across the skill grid (for plotting / docs)."""
    rows = []
    for s in (0.0, 0.25, 0.5, 0.75, 1.0):
        policy = MockPolicy(skill=s)
        e = evaluate_heldout(world, policy, task_ids, n_rollouts)
        rows.append({"skill": s, **e})
    return rows
