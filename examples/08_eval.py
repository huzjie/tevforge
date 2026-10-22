# -*- coding: utf-8 -*-
"""Heldout evaluation + reward-hacking audit.

Run:  python examples/08_eval.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tevforge.world.answer_world import AnswerWorld
from tevforge.agent.policy import MockPolicy
from tevforge.eval.evaluate import (evaluate_heldout, progress_reward_gap,
                                    reward_hacking_audit)

world = AnswerWorld(seed=20261022)
tasks = ["eval:%d" % i for i in range(16)]

print("=== heldout evaluation (skill 0.5 vs 1.0) ===")
print("skill 0.5:", evaluate_heldout(world, MockPolicy(skill=0.5), tasks))
print("skill 1.0:", evaluate_heldout(world, MockPolicy(skill=1.0), tasks))

gap = progress_reward_gap(world, MockPolicy(skill=0.5), tasks)
print("progress-reward gap (should be ~0): %.6f" % gap)

print("=== reward-hacking audit ===")
for skill, e in reward_hacking_audit(world, tasks).items():
    print("skill %.1f -> return=%.3f progress=%.3f" % (skill, e["mean_return"], e["mean_progress"]))
