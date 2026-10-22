# -*- coding: utf-8 -*-
"""One-call engine: synthesize envs, train with PPRS, and evaluate.

Binds the four pieces together so a user can go from config to a trained
long-horizon agent in a few lines.
"""
from ..world.answer_world import AnswerWorld
from ..synth.gym import TevGym
from ..reward.pprs import PPRS
from ..agent.policy import MockPolicy
from ..agent.rollout import rollout
from ..trainer.grpo import GRPO


class TrainEngine:
    def __init__(self, cfg=None):
        self.cfg = cfg
        self.world = AnswerWorld(
            horizon_min=cfg.world.horizon_min,
            horizon_max=cfg.world.horizon_max,
            seed=cfg.world.seed,
        )
        self.gym = TevGym(world=self.world)
        self.reward = PPRS(gamma=cfg.reward.gamma, goal_bonus=cfg.reward.goal_bonus)
        self.policy = MockPolicy(skill=cfg.model.skill, seed=cfg.world.seed)
        self.trainer = GRPO(self.policy, self.world, self.reward,
                            group_size=cfg.train.group_size,
                            learning_rate=cfg.train.learning_rate)

    def fit(self, n_tasks=8, max_steps=None):
        tasks = self.gym.build(tasks_per_domain=max(1, n_tasks // len(self.gym.domains)))
        history = self.trainer.train(tasks, max_steps=max_steps)
        return {"tasks": tasks, "history": history,
                "final_skill": self.policy.skill}

    def evaluate(self, task_ids, n_rollouts=8):
        """Completion rate and mean return over heldout tasks."""
        comps = []
        returns = []
        for tid in task_ids:
            for i in range(n_rollouts):
                traj = rollout(self.policy, self.world, tid, self.reward.step_reward, salt=str(i))
                comps.append(1 if traj.completed else 0)
                returns.append(self.reward.return_(self.world, tid, traj.actions))
        return {
            "completion_rate": sum(comps) / len(comps),
            "mean_return": sum(returns) / len(returns),
        }
