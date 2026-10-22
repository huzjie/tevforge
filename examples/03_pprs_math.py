# -*- coding: utf-8 -*-
"""Verify PPRS telescopes: shaped return = sparse return + final progress.

Run:  python examples/03_pprs_math.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tevforge.world.answer_world import AnswerWorld
from tevforge.reward.sparse import SparseReward
from tevforge.reward.pprs import PPRS

world = AnswerWorld(seed=20261022)
tid = "math:1"
L = world.n_steps(tid)
plan = [world.ground_truth(tid, t) for t in range(L)]

sparse = SparseReward().return_(world, tid, plan)
pprs = PPRS(gamma=1.0).return_(world, tid, plan)
print("sparse return       : %.4f" % sparse)
print("pprs shaped return  : %.4f" % pprs)
print("final progress      : %.4f" % world.progress(tid, plan))
print("shaped - sparse     : %.4f  (should equal final progress)" % (pprs - sparse))
