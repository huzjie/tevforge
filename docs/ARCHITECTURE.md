# Architecture

`tevforge` is a zero-dependency Python reference implementation of the
UCLA x Adobe Research **Tev** approach to long-horizon agent training
(arXiv 2610.18868). It isolates four composable problems and solves each
independently.

## The two failure modes

1. **No training data.** Existing agent corpora are short (median < 2 min):
   a single PR fix or a brief terminal session. True long-horizon tasks exist
   only as hand-written benchmarks for eval, not for training.

2. **Sparse rewards.** GRPO normalizes rewards across G rollouts of the *same*
   task. If the group all fails, every reward is 0, the standard deviation is 0
   and every advantage collapses to 0. The wasted fraction is `(1 - p)^G`,
   which for `p = 0.01`, `G = 8` is 92%.

## Four pieces

```
                    +-----------------------+
                    |  synth / TevGym       |  answer-world-first env synthesis
                    +-----------+-----------+
                                | task ids (verifiable)
                                v
+---------------+      +--------+-----------+      +-----------------+
| world         |----->| reward / PPRS       |----->| trainer / GRPO   |
| answer world  |      | potential shaping   |      | dense advantages |
+---------------+      +--------------------+      +-----------------+
        ^                                                    |
        |                 agent / policy <-------------------+
        |                 rollout (long-horizon)
        +----------------------------------------------------+
```

* **`world`** — a deterministic "answer world" that is the single source of
  truth: ground-truth actions, a `progress` measure and a verifiable
  `is_complete` predicate.
* **`synth`** — `TevGym` builds verifiable tasks from the answer world; a task
  arrives pre-annotated with a dense, progressive reward.
* **`reward`** — `PPRS` (Progress Potential Reward Shaping) turns the sparse
  reward into a potential-based dense reward `r' = r + gamma*Phi(s') - Phi(s)`,
  which is provably policy-invariant.
* **`trainer`** — `GRPO` estimates group advantages over dense shaped returns,
  eliminating the "whole group fails -> zero advantage" waste.

## Determinism and the single oracle

Every component derives pseudo-randomness from `md5(key)`. The mock policy and
every benchmark share the answer world as the oracle, so `skill = 0.5` is
exactly a random walk (`completion ~ 0.5^L ~ 0`) and `skill = 1.0` is exactly
the perfect policy (`completion = 1.0`). Training is a monotonic
`skill += lr * (1 - skill)`.
