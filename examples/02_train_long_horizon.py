# -*- coding: utf-8 -*-
"""Train a long-horizon agent with PPRS + GRPO and watch skill climb 0.5 -> 1.0.

Run:  python examples/02_train_long_horizon.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tevforge.config import Config
from tevforge.engine.train_engine import TrainEngine

cfg = Config()
engine = TrainEngine(cfg)

print("skill before training: %.4f" % engine.policy.skill)
result = engine.fit(n_tasks=8, max_steps=40)
print("skill after  training: %.4f" % result["final_skill"])

first = result["history"][0]
last = result["history"][-1]
print("first iteration: skill=%.4f completion=%.4f" % (first[1], first[3]))
print("last  iteration: skill=%.4f completion=%.4f" % (last[1], last[3]))
