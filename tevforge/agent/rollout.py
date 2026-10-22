# -*- coding: utf-8 -*-
"""Long-horizon rollout helpers."""

from .policy import Policy
from ..world.state import Trajectory, StepResult


def rollout(policy, world, task_id, reward_fn=None, salt=""):
    """Run a policy for one full episode and return a ``Trajectory``.

    ``reward_fn(world, task_id, prefix, action) -> float`` supplies the
    per-step reward (sparse or PPRS-shaped).
    """
    from ..reward.pprs import PPRS
    if reward_fn is None:
        reward_fn = PPRS().step_reward

    traj = Trajectory(task_id=task_id)
    L = world.n_steps(task_id)
    for step in range(L):
        a = policy.act(world, task_id, step, traj.actions, salt=salt)
        r = reward_fn(world, task_id, traj.actions, a)
        progress = world.progress(task_id, traj.actions + [a])
        correct = (a == world.ground_truth(task_id, step))
        traj.actions.append(a)
        traj.rewards.append(r)
        traj.steps.append(StepResult(step, a, r, progress, correct))
    traj.completed = world.is_complete(task_id, traj.actions)
    return traj


def rollout_return(policy, world, task_id, reward=None, salt=""):
    """Convenience: return only the episode return of a rollout."""
    from ..reward.pprs import PPRS
    rfn = reward.return_ if reward is not None else PPRS().return_
    traj = rollout(policy, world, task_id, salt=salt)
    return rfn(world, task_id, traj.actions)
