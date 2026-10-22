# -*- coding: utf-8 -*-
import unittest

from tevforge.world.answer_world import AnswerWorld
from tevforge.agent.policy import MockPolicy
from tevforge.bench import benchmarks as B


class TestBenchmarks(unittest.TestCase):
    def setUp(self):
        self.world = AnswerWorld(seed=20261022)
        self.tasks = ["b%d" % i for i in range(16)]

    def test_completion_monotonic_in_skill(self):
        lo = B.completion_rate(self.world, MockPolicy(skill=0.5), self.tasks, n_rollouts=8)
        hi = B.completion_rate(self.world, MockPolicy(skill=1.0), self.tasks, n_rollouts=8)
        self.assertLess(lo, hi)
        self.assertAlmostEqual(hi, 1.0, places=6)

    def test_progress_monotonic(self):
        lo = B.progress_rate(self.world, MockPolicy(skill=0.5), self.tasks, n_rollouts=8)
        hi = B.progress_rate(self.world, MockPolicy(skill=1.0), self.tasks, n_rollouts=8)
        self.assertLess(lo, hi)

    def test_dense_beats_sparse_waste(self):
        # at skill 0.5, sparse GRPO wastes far more groups than dense PPRS
        sp = B.wasted_group_rate(self.world, MockPolicy(skill=0.5), self.tasks, sparse=True)
        dn = B.wasted_group_rate(self.world, MockPolicy(skill=0.5), self.tasks, sparse=False)
        self.assertGreater(sp, dn)

    def test_run_all(self):
        from tevforge.config import Config
        report = B.run_all(Config())
        self.assertIn("sparse_vs_dense", report)
        self.assertIn(0.5, report["sparse_vs_dense"])


if __name__ == "__main__":
    unittest.main()
