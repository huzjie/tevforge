# -*- coding: utf-8 -*-
import unittest

from tevforge.world.answer_world import AnswerWorld
from tevforge.synth.gym import TevGym
from tevforge.synth.env_synth import EnvSynthesizer
from tevforge.synth.dense_reward import DenseAnnotator


class TestSynth(unittest.TestCase):
    def test_gym_builds_verifiable_tasks(self):
        world = AnswerWorld(seed=20261022)
        gym = TevGym(world=world)
        tasks = gym.build(tasks_per_domain=3)
        self.assertEqual(len(tasks), 3 * len(gym.domains))
        for tid in tasks:
            self.assertTrue(world.n_steps(tid) > 0)

    def test_split(self):
        world = AnswerWorld(seed=20261022)
        gym = TevGym(world=world)
        tasks = gym.build(tasks_per_domain=2)
        tr, held = gym.split(tasks, heldout_frac=0.25)
        self.assertGreater(len(held), 0)
        self.assertEqual(len(tr) + len(held), len(tasks))

    def test_annotator_dense(self):
        world = AnswerWorld(seed=20261022)
        ann = DenseAnnotator()
        table = ann.annotate(world, "ann:1")
        self.assertEqual(len(table), world.n_steps("ann:1"))
        # ground-truth step rewards are all positive (progress + eventual bonus)
        self.assertTrue(all(t["reward"] > 0 for t in table))


if __name__ == "__main__":
    unittest.main()
