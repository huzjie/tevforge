# -*- coding: utf-8 -*-
"""Tev-Gym: a synthetic training set of dense, progressive, verifiable tasks.

Tev-Gym (1,170 environments across 7 domains in the paper) is reproduced here
in miniature: a builder that composes multiple domains and yields tasks whose
reward is dense and whose completion is verifiable by construction.
"""
from ..world.answer_world import AnswerWorld
from .env_synth import EnvSynthesizer


class TevGym:
    DOMAINS = ["terminal", "swe", "search", "web", "os", "mcp", "android"]

    def __init__(self, world=None, domains=None):
        self.world = world or AnswerWorld()
        self.domains = list(domains) if domains else self.DOMAINS
        self.synth = {d: EnvSynthesizer(self.world, domain=d) for d in self.domains}

    def build(self, tasks_per_domain=8):
        """Build a training set: ``tasks_per_domain`` tasks per domain."""
        tasks = []
        for d in self.domains:
            tasks.extend(self.synth[d].synthesize(tasks_per_domain))
        return tasks

    def split(self, tasks, heldout_frac=0.2, seed=20261022):
        """Deterministic train / heldout split (heldout for eval only)."""
        n_hold = max(1, int(len(tasks) * heldout_frac))
        # stable split: last n_hold tasks are heldout
        return tasks[:-n_hold], tasks[-n_hold:]

    def stats(self):
        return {
            "domains": len(self.domains),
            "action_space": len(self.world.actions),
            "horizon_range": [self.world.horizon_min, self.world.horizon_max],
        }
