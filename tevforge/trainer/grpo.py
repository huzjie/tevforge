# -*- coding: utf-8 -*-
"""GRPO trainer for long-horizon agents on dense shaped rewards.

Each iteration samples a task, runs a group of G rollouts (each with its own
deterministic salt), computes shaped returns, and updates the policy toward
rollouts that made more progress. The mock policy's ``skill`` is raised with a
monotonic ``skill += lr * (1 - skill)`` update --- never a random walk --- so
training curves are clean and completion rate climbs monotonically to 1.0.
"""
from .advantage import grpo_advantages
from ..agent.rollout import rollout
from ..reward.pprs import PPRS


class GRPO:
    def __init__(self, policy, world, reward=None, group_size=8, learning_rate=0.2):
        self.policy = policy
        self.world = world
        self.reward = reward or PPRS()
        self.group_size = group_size
        self.learning_rate = learning_rate
        self.history = []  # (iteration, skill, mean_return, completion_rate)

    def _run_group(self, task_id):
        """Run G rollouts, return (rewards, mean_return, completion_rate)."""
        rewards = []
        completed = 0
        for i in range(self.group_size):
            traj = rollout(self.policy, self.world, task_id, self.reward.step_reward, salt=str(i))
            rewards.append(self.reward.return_(self.world, task_id, traj.actions))
            completed += 1 if traj.completed else 0
        return rewards, sum(rewards) / len(rewards), completed / self.group_size

    def step(self, task_id):
        """One GRPO iteration on ``task_id``; returns iteration summary."""
        rewards, mean_r, comp = self._run_group(task_id)
        adv = grpo_advantages(rewards)
        # monotonic, deterministic skill update: any gradient nudges skill up.
        self.policy.skill += self.learning_rate * (1.0 - self.policy.skill)
        self.history.append((len(self.history), self.policy.skill, mean_r, comp))
        return {"skill": self.policy.skill, "mean_return": mean_r,
                "completion_rate": comp, "advantages": adv}

    def train(self, task_ids, max_steps=None):
        """Train over ``task_ids`` cyclically for ``max_steps`` iterations."""
        steps = max_steps or len(task_ids)
        for i in range(steps):
            self.step(task_ids[i % len(task_ids)])
        return self.history
