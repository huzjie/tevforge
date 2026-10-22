# -*- coding: utf-8 -*-
import unittest

from tevforge.world.answer_world import AnswerWorld
from tevforge.agent.policy import MockPolicy
from tevforge.agent.rollout import rollout
from tevforge.reward.pprs import PPRS
from tevforge.data.dataset import TrajectoryStore


class TestTrajectoryStore(unittest.TestCase):
    def test_store(self):
        world = AnswerWorld(seed=20261022)
        policy = MockPolicy(skill=1.0)
        store = TrajectoryStore()
        for i in range(4):
            store.add(rollout(policy, world, "s%d" % i, PPRS().step_reward))
        self.assertEqual(len(store), 4)
        self.assertEqual(store.completed_count(), 4)
        self.assertAlmostEqual(store.completion_rate(), 1.0)
        recs = store.to_records()
        self.assertEqual(len(recs), 4)
        self.assertIn("progress", recs[0])


if __name__ == "__main__":
    unittest.main()
