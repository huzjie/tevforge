# -*- coding: utf-8 -*-
import unittest

from tevforge.world.answer_world import AnswerWorld
from tevforge.world.action import ActionSpace


class TestAnswerWorld(unittest.TestCase):
    def setUp(self):
        self.world = AnswerWorld(seed=20261022)

    def test_deterministic(self):
        a = self.world.ground_truth("t", 0)
        b = self.world.ground_truth("t", 0)
        self.assertEqual(a, b)

    def test_horizon_bounds(self):
        for i in range(50):
            L = self.world.n_steps("task:%d" % i)
            self.assertGreaterEqual(L, self.world.horizon_min)
            self.assertLess(L, self.world.horizon_max)

    def test_progress_and_completion(self):
        tid = "comp:test"
        L = self.world.n_steps(tid)
        plan = [self.world.ground_truth(tid, t) for t in range(L)]
        self.assertTrue(self.world.is_complete(tid, plan))
        self.assertAlmostEqual(self.world.progress(tid, plan), 1.0)
        # wrong plan
        wrong = plan[:]
        wrong[0] = self.world.actions.random_wrong(tid, 0)
        self.assertFalse(self.world.is_complete(tid, wrong))

    def test_action_space(self):
        s = ActionSpace(seed=20261022)
        self.assertEqual(len(s), 8)
        gt = s.ground_truth("x", 3)
        self.assertIn(gt, s.tokens)
        wrong = s.random_wrong("x", 3)
        self.assertNotEqual(wrong, gt)


class TestProgress(unittest.TestCase):
    def test_progress_is_fraction(self):
        world = AnswerWorld(seed=1)
        tid = "pg"
        L = world.n_steps(tid)
        plan = [world.ground_truth(tid, t) for t in range(L)]
        self.assertAlmostEqual(world.progress(tid, plan[: L // 2]), 0.5, places=6)


if __name__ == "__main__":
    unittest.main()
