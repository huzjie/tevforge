# -*- coding: utf-8 -*-
"""The core demonstration: dense (PPRS) reward beats sparse for long-horizon GRPO.

Run:  python examples/01_dense_vs_sparse.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tevforge.world.answer_world import AnswerWorld
from tevforge.agent.policy import MockPolicy
from tevforge.bench.benchmarks import (wasted_group_rate, reward_sparsity,
                                       advantage_density, completion_rate)

world = AnswerWorld(seed=20261022)
tasks = ["demo:%d" % i for i in range(64)]
policy = MockPolicy(skill=0.5)

print("=== skill = 0.5 (training start) ===")
print("completion_rate        : %.4f" % completion_rate(world, policy, tasks))
print("wasted_group  sparse    : %.4f" % wasted_group_rate(world, policy, tasks, sparse=True))
print("wasted_group  dense(PPRS): %.4f" % wasted_group_rate(world, policy, tasks, sparse=False))
print("advantage_density sparse: %.4f" % advantage_density(world, policy, tasks, sparse=True))
print("advantage_density dense : %.4f" % advantage_density(world, policy, tasks, sparse=False))
print("reward_sparsity sparse  : %.4f" % reward_sparsity(world, policy, tasks, sparse=True))
print("reward_sparsity dense   : %.4f" % reward_sparsity(world, policy, tasks, sparse=False))
