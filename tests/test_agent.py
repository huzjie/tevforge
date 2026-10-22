# -*- coding: utf-8 -*-
import unittest

from tevforge.world.answer_world import AnswerWorld
from tevforge.agent.policy import MockPolicy
from tevforge.agent.rollout import rollout
from tevforge.reward.pprs import PPRS


class TestPolicy(unittest.TestCase):
    def setUp(self):
        self.world = AnswerWorld(seed=20261022)

    def test_perfect_policy_completes(self):
        p = MockPolicy(skill=1.0)
        traj = rollout(p, self.world, "perf", PPRS().step_reward)
        self.assertTrue(traj.completed)
        self.assertAlmostEqual(traj.final_progress, 1.0)

    def test_random_policy_is_wrong_sometimes(self):
        p = MockPolicy(skill=0.5)
        traj = rollout(p, self.world, "rand", PPRS().step_reward)
        # at skill 0.5 a random walk essentially never completes
        self.assertFalse(traj.completed)
        self.assertLess(traj.final_progress, 0.9)

    def test_rollout_length(self):
        p = MockPolicy(skill=0.5)
        tid = "len"
        traj = rollout(p, self.world, tid, PPRS().step_reward)
        self.assertEqual(traj.length, self.world.n_steps(tid))


if __name__ == "__main__":
    unittest.main()
