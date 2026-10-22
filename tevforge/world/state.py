# -*- coding: utf-8 -*-
"""Trajectory data structures for long-horizon rollouts."""
from dataclasses import dataclass, field
from typing import List


@dataclass
class StepResult:
    step: int
    action: str
    reward: float
    progress: float
    correct: bool


@dataclass
class Trajectory:
    task_id: str
    actions: List[str] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    steps: List[StepResult] = field(default_factory=list)
    completed: bool = False

    @property
    def length(self):
        return len(self.actions)

    @property
    def total_reward(self):
        return sum(self.rewards)

    @property
    def final_progress(self):
        if not self.steps:
            return 0.0
        return self.steps[-1].progress
