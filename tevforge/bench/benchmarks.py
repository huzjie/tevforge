# -*- coding: utf-8 -*-
"""Benchmarks that quantify long-horizon training quality.

All metrics are deterministic and share the answer world as the single oracle,
so they are reproducible and monotonic in ``skill`` where expected.

The headline comparison is sparse vs dense (PPRS) reward:

  * ``completion_rate`` : s**L  (random walk at s=0.5 -> ~0)
  * ``progress_rate``   : s     (linear)
  * ``reward_sparsity`` : fraction of zero-reward steps
  * ``advantage_density``: fraction of rollouts with a usable gradient
  * ``wasted_group_rate``: fraction of GRPO groups with all-zero advantage
"""
from ..world.answer_world import AnswerWorld
from ..agent.policy import MockPolicy
from ..agent.rollout import rollout
from ..reward.sparse import SparseReward
from ..reward.pprs import PPRS
from ..trainer.advantage import grpo_advantages
from ..utils.metrics import mean


def _make(skill=0.5, seed=20261022):
    world = AnswerWorld(seed=seed)
    policy = MockPolicy(skill=skill, seed=seed)
    return world, policy


def completion_rate(world, policy, task_ids, n_rollouts=8):
    """Fraction of rollouts that reproduce the full ground-truth plan."""
    done = 0
    for tid in task_ids:
        for i in range(n_rollouts):
            traj = rollout(policy, world, tid, salt="c%d" % i)
            done += 1 if traj.completed else 0
    return done / (len(task_ids) * n_rollouts)


def progress_rate(world, policy, task_ids, n_rollouts=8):
    """Mean final progress (fraction of correct actions) across rollouts."""
    progs = []
    for tid in task_ids:
        for i in range(n_rollouts):
            traj = rollout(policy, world, tid, salt="p%d" % i)
            progs.append(traj.final_progress)
    return mean(progs)


def reward_sparsity(world, policy, task_ids, n_rollouts=8, sparse=True):
    """Fraction of steps that yield exactly zero reward.

    Sparse reward is zero on every step except the completing one; PPRS is zero
    only when a step fails to advance progress."""
    rfn = SparseReward().step_reward if sparse else PPRS().step_reward
    zeros = 0
    total = 0
    for tid in task_ids:
        for i in range(n_rollouts):
            traj = rollout(policy, world, tid, rfn, salt="z%d" % i)
            zeros += sum(1 for r in traj.rewards if abs(r) < 1e-9)
            total += len(traj.rewards)
    return zeros / total


def advantage_density(world, policy, task_ids, group_size=8, sparse=False, eps=1e-9):
    """Fraction of (group, rollout) pairs with a non-zero advantage."""
    reward = SparseReward() if sparse else PPRS()
    non_zero = 0
    total = 0
    for tid in task_ids:
        rewards = []
        for i in range(group_size):
            traj = rollout(policy, world, tid, reward.step_reward, salt="a%d" % i)
            rewards.append(reward.return_(world, tid, traj.actions))
        adv = grpo_advantages(rewards, eps=eps)
        non_zero += sum(1 for a in adv if abs(a) > eps)
        total += len(adv)
    return non_zero / total


def wasted_group_rate(world, policy, task_ids, group_size=8, sparse=False, eps=1e-9):
    """Fraction of groups whose advantage is zero everywhere (no gradient)."""
    reward = SparseReward() if sparse else PPRS()
    wasted = 0
    for tid in task_ids:
        rewards = []
        for i in range(group_size):
            traj = rollout(policy, world, tid, reward.step_reward, salt="w%d" % i)
            rewards.append(reward.return_(world, tid, traj.actions))
        adv = grpo_advantages(rewards, eps=eps)
        if all(abs(a) < eps for a in adv):
            wasted += 1
    return wasted / len(task_ids)


def run_all(cfg=None):
    """Run the full benchmark suite and return a report dict."""
    from ..config import Config
    cfg = cfg or Config()
    seed = cfg.bench.seed
    n_tasks = cfg.bench.n_tasks
    G = cfg.bench.group_size
    task_ids = ["bench:%d" % i for i in range(n_tasks)]

    report = {"declared": cfg.declared.to_dict(), "n_tasks": n_tasks, "group_size": G}

    for skill in (0.5, 0.75, 1.0):
        world, policy = _make(skill=skill, seed=seed)
        report.setdefault("skill_sweep", {})[skill] = {
            "completion_rate": completion_rate(world, policy, task_ids, n_rollouts=G),
            "progress_rate": progress_rate(world, policy, task_ids, n_rollouts=G),
        }

    # sparse vs dense at training start (skill = 0.5) and ceiling (skill = 1.0)
    for skill in (0.5, 1.0):
        world, policy = _make(skill=skill, seed=seed)
        report.setdefault("sparse_vs_dense", {})[skill] = {
            "reward_sparsity_sparse": reward_sparsity(world, policy, task_ids, sparse=True),
            "reward_sparsity_dense": reward_sparsity(world, policy, task_ids, sparse=False),
            "advantage_density_sparse": advantage_density(world, policy, task_ids, group_size=G, sparse=True),
            "advantage_density_dense": advantage_density(world, policy, task_ids, group_size=G, sparse=False),
            "wasted_group_sparse": wasted_group_rate(world, policy, task_ids, group_size=G, sparse=True),
            "wasted_group_dense": wasted_group_rate(world, policy, task_ids, group_size=G, sparse=False),
        }

    return report
