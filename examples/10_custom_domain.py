# -*- coding: utf-8 -*-
"""Add a custom Tev-Gym domain with a custom action space.

Run:  python examples/10_custom_domain.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tevforge.world.answer_world import AnswerWorld
from tevforge.world.action import ActionSpace
from tevforge.synth.gym import TevGym

custom_tokens = ["open", "read", "write", "close", "commit", "push"]
actions = ActionSpace(tokens=custom_tokens)
world = AnswerWorld(action_space=actions, horizon_min=4, horizon_max=12)
gym = TevGym(world=world, domains=["git", "editor"])

tasks = gym.build(tasks_per_domain=3)
for tid in tasks:
    print(tid, "horizon=%d" % world.n_steps(tid))
