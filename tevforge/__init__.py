# -*- coding: utf-8 -*-
"""tevforge - long-horizon agent dense-reward RL training framework.

A zero-dependency Python reference implementation of the ideas behind the
UCLA x Adobe Research ``Tev`` line of work (arXiv 2610.18868): long-horizon
agents fail because (1) there is no *training* data for multi-hour tasks and
(2) rewards are sparse, so GRPO normalizes a group of rollouts that all fail
to zero advantage and the whole group is wasted.

tevforge provides four composable pieces:

  * world    - a deterministic "answer world" (latent world) that is the single
               source of truth for every task: ground-truth actions, progress
               and a verifiable completion predicate.
  * synth    - "answer-world-first" environment synthesis: build the verifiable
               world, then generate long-horizon tasks with dense, progressive,
               verifiable rewards (the Tev-Gym idea).
  * reward   - Progress Potential Reward Shaping (PPRS): a potential-based
               dense reward `r' = r + gamma*Phi(s') - Phi(s)` that provably
               preserves the optimal policy while giving signal every step.
  * trainer  - long-horizon GRPO over dense shaped returns, which eliminates
               the "whole group fails -> zero advantage" waste.

Everything is deterministic (md5-seeded) and shares a single oracle, so the
mock policy, the benchmarks and the trainer agree bit-for-bit. ``skill = 0.5``
is a random walk (completion ~ s^L ~ 0), ``skill = 1.0`` is a perfect policy
(completion = 1.0). Training raises skill monotonically via dense reward.
"""

__version__ = "1.0.0"

from .config import Config, load_config
from .world.answer_world import AnswerWorld
from .reward.pprs import PPRS
from .trainer.grpo import GRPO

__all__ = ["Config", "load_config", "AnswerWorld", "PPRS", "GRPO", "__version__"]
