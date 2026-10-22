# -*- coding: utf-8 -*-
import unittest

from tevforge.world.answer_world import AnswerWorld
from tevforge.agent.policy import MockPolicy
from tevforge.trainer.grpo import GRPO
from tevforge.trainer.advantage import grpo_advantages, wasted_group


class TestGRPO(unittest.TestCase):
    def setUp(self):
        self.world = AnswerWorld(seed=20261022)
        self.policy = MockPolicy(skill=0.5)

    def test_skill_monotonic(self):
        grpo = GRPO(self.policy, self.world, group_size=8, learning_rate=0.2)
        grpo.train(["t1", "t2"], max_steps=20)
        skills = [h[1] for h in grpo.history]
        self.assertTrue(all(b >= a for a, b in zip(skills, skills[1:])))
        self.assertGreater(skills[-1], 0.99)

    def test_wasted_group_detection(self):
        self.assertTrue(wasted_group([0.0, 0.0, 0.0, 0.0]))
        self.assertFalse(wasted_group([0.0, 0.5, 1.0, 0.0]))

    def test_advantages_zero_mean(self):
        adv = grpo_advantages([0.0, 0.5, 1.0, 0.25])
        self.assertAlmostEqual(sum(adv), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
