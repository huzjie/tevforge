# -*- coding: utf-8 -*-
"""Ablation: completion rate and waste across the skill range.

Run:  python examples/07_ablation.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tevforge.world.answer_world import AnswerWorld
from tevforge.agent.policy import MockPolicy
from tevforge.bench.benchmarks import completion_rate, wasted_group_rate

world = AnswerWorld(seed=20261022)
tasks = ["ab:%d" % i for i in range(32)]

print("skill | completion | waste_sparse | waste_dense")
for s in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
    p = MockPolicy(skill=s)
    comp = completion_rate(world, p, tasks)
    ws = wasted_group_rate(world, p, tasks, sparse=True)
    wd = wasted_group_rate(world, p, tasks, sparse=False)
    print(" %.2f |    %.4f   |   %.4f    |   %.4f" % (s, comp, ws, wd))
