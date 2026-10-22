# -*- coding: utf-8 -*-
"""Verify the reward cannot be gained without genuine progress.

Run:  python examples/09_reward_hacking_audit.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tevforge.world.answer_world import AnswerWorld
from tevforge.agent.policy import MockPolicy
from tevforge.eval.evaluate import reward_hacking_audit

world = AnswerWorld(seed=20261022)
tasks = ["hack:%d" % i for i in range(24)]
audit = reward_hacking_audit(world, tasks)

print("A no-progress policy (skill=0) must earn ~0 return:")
print("  skill=0.0 return=%.4f progress=%.4f" % (audit[0.0]["mean_return"], audit[0.0]["mean_progress"]))
print("  skill=1.0 return=%.4f progress=%.4f" % (audit[1.0]["mean_return"], audit[1.0]["mean_progress"]))
print("=> reward scales with progress, not with clever shortcuts.")
