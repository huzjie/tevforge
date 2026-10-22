# -*- coding: utf-8 -*-
"""A lightweight trajectory store for offline analysis and replay.

Keeps rollouts in memory and exposes summary statistics and serialization, so a
training run can be inspected without any external dependency.
"""
from ..utils.metrics import mean


class TrajectoryStore:
    def __init__(self):
        self.trajectories = []

    def add(self, traj):
        self.trajectories.append(traj)

    def __len__(self):
        return len(self.trajectories)

    def completed_count(self):
        return sum(1 for t in self.trajectories if t.completed)

    def completion_rate(self):
        if not self.trajectories:
            return 0.0
        return self.completed_count() / len(self.trajectories)

    def mean_progress(self):
        return mean([t.final_progress for t in self.trajectories])

    def mean_return(self):
        return mean([t.total_reward for t in self.trajectories])

    def to_records(self):
        return [
            {"task_id": t.task_id, "length": t.length, "completed": t.completed,
             "progress": t.final_progress, "return": t.total_reward}
            for t in self.trajectories
        ]

    def summary(self):
        return {
            "n": len(self),
            "completion_rate": self.completion_rate(),
            "mean_progress": self.mean_progress(),
            "mean_return": self.mean_return(),
        }
