# -*- coding: utf-8 -*-
"""Answer-world-first environment synthesis.

The failure mode Tev attacks is that long-horizon *training* environments do
not exist: benchmark tasks are hand-written and only for eval. This synthesizer
flips the order --- it first builds (or reuses) a deterministic answer world,
then spawns any number of verifiable long-horizon tasks with known solutions.
"""
from ..world.answer_world import AnswerWorld


class EnvSynthesizer:
    """Generates task ids whose solution is verifiable in ``world``."""

    def __init__(self, world, domain="terminal", prefix="tev"):
        self.world = world
        self.domain = domain
        self.prefix = prefix

    def task_id(self, index):
        return "%s:%s:%d" % (self.prefix, self.domain, index)

    def synthesize(self, n_tasks, index_offset=0):
        """Return ``n_tasks`` task ids with known, verifiable solutions."""
        return [self.task_id(index_offset + i) for i in range(n_tasks)]

    def describe(self, task_id):
        """Human-readable summary of a task (for logging / docs)."""
        L = self.world.n_steps(task_id)
        return {
            "task_id": task_id,
            "domain": self.domain,
            "horizon": L,
            "action_space": len(self.world.actions),
            "verifiable": True,
        }
