# -*- coding: utf-8 -*-
"""Build a Tev-Gym training set and annotate a task with dense rewards.

Run:  python examples/04_tev_gym.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tevforge.world.answer_world import AnswerWorld
from tevforge.synth.gym import TevGym
from tevforge.synth.dense_reward import DenseAnnotator

world = AnswerWorld(seed=20261022)
gym = TevGym(world=world)
tasks = gym.build(tasks_per_domain=2)
train, heldout = gym.split(tasks, heldout_frac=0.25)

print("stats:", gym.stats())
print("total tasks:", len(tasks), " train:", len(train), " heldout:", len(heldout))

ann = DenseAnnotator()
table = ann.annotate(world, tasks[0])
print("task %s horizon=%d" % (tasks[0], world.n_steps(tasks[0])))
print("first 3 dense step-rewards:", [round(t["reward"], 4) for t in table[:3]])
