# Benchmark methodology

All numbers are measured by `python -m tevforge bench` over 64 synthesized
tasks with group size 8 and a fixed seed (20261022). They are deterministic and
reproducible.

## Metrics

* **completion_rate** — fraction of rollouts reproducing the full ground-truth
  plan. Equals `s^L`, so it is ~0 at the random baseline and 1.0 at the ceiling.
* **progress_rate** — mean final progress (fraction of correct actions). Equals
  `s`, linear in skill.
* **reward_sparsity** — fraction of steps with exactly zero reward. Sparse is
  ~1.0 (only the final step earns); PPRS is ~`1 - progress`.
* **advantage_density** — fraction of rollouts with a non-zero group advantage.
  Sparse is 0 at training start (all-zero returns); PPRS is ~0.96.
* **wasted_group_rate** — fraction of GRPO groups with all-zero advantage.
  Sparse is 1.0 (every group wasted) and PPRS is 0.0 at skill = 0.5.

## Interpretation

The sparse-vs-dense table at `skill = 0.5` is the headline: it shows exactly
why sparse reward fails long-horizon GRPO and what PPRS buys.
