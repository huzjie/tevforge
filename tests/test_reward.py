# -*- coding: utf-8 -*-
import unittest

from tevforge.world.answer_world import AnswerWorld
from tevforge.reward.sparse import SparseReward
from tevforge.reward.pprs import PPRS
from tevforge.reward.dense import DenseReward


class TestRewards(unittest.TestCase):
    def setUp(self):
        self.world = AnswerWorld(seed=20261022)
        self.tid = "rw:test"

    def _plan(self):
        L = self.world.n_steps(self.tid)
        return [self.world.ground_truth(self.tid, t) for t in range(L)]

    def test_sparse_return(self):
        plan = self._plan()
        r = SparseReward()
        self.assertEqual(r.return_(self.world, self.tid, plan), 1.0)
        wrong = plan[:]
        wrong[-1] = self.world.actions.random_wrong(self.tid, len(plan) - 1)
        self.assertEqual(r.return_(self.world, self.tid, wrong), 0.0)

    def test_pprs_telescopes(self):
        # With gamma=1 and Phi(empty)=0 the shaped return == sparse + final progress.
        plan = self._plan()
        pprs = PPRS(gamma=1.0, goal_bonus=1.0)
        self.assertAlmostEqual(pprs.return_(self.world, self.tid, plan), 2.0, places=6)

    def test_pprs_step_rewards_sum_to_progress(self):
        plan = self._plan()
        pprs = PPRS(gamma=1.0, goal_bonus=1.0)
        prefix = []
        s = 0.0
        for a in plan:
            s += pprs.step_reward(self.world, self.tid, prefix, a)
            prefix.append(a)
        # telescoped potential + goal bonus = 1 (progress) + 1 (bonus)
        self.assertAlmostEqual(s, 2.0, places=6)

    def test_dense_reward_non_negative(self):
        plan = self._plan()
        d = DenseReward()
        prefix = []
        for a in plan:
            r = d.step_reward(self.world, self.tid, prefix, a)
            self.assertGreaterEqual(r, 0.0)
            prefix.append(a)


if __name__ == "__main__":
    unittest.main()
