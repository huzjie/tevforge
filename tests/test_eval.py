# -*- coding: utf-8 -*-
import unittest

from tevforge.world.answer_world import AnswerWorld
from tevforge.agent.policy import MockPolicy
from tevforge.eval.evaluate import (evaluate_heldout, progress_reward_gap,
                                    reward_hacking_audit)


class TestEval(unittest.TestCase):
    def setUp(self):
        self.world = AnswerWorld(seed=20261022)
        self.tasks = ["e%d" % i for i in range(8)]

    def test_heldout_bounds(self):
        e = evaluate_heldout(self.world, MockPolicy(skill=0.5), self.tasks, n_rollouts=4)
        self.assertGreaterEqual(e["completion_rate"], 0.0)
        self.assertLessEqual(e["completion_rate"], 1.0)

    def test_progress_reward_gap_is_zero(self):
        gap = progress_reward_gap(self.world, MockPolicy(skill=0.5), self.tasks, n_rollouts=4)
        self.assertAlmostEqual(gap, 0.0, places=6)

    def test_hacking_audit_monotonic(self):
        audit = reward_hacking_audit(self.world, self.tasks, n_rollouts=4)
        self.assertLess(audit[0.0]["mean_return"], audit[1.0]["mean_return"])


if __name__ == "__main__":
    unittest.main()
