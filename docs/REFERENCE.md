# API Reference

## `tevforge.world.answer_world.AnswerWorld`

* `n_steps(task_id) -> int` — horizon L.
* `ground_truth(task_id, step) -> str` — correct action.
* `progress(task_id, actions) -> float` — fraction correct in `[0, 1]`.
* `is_complete(task_id, actions) -> bool` — full plan reproduced.

## `tevforge.reward.pprs.PPRS`

* `PPRS(gamma=1.0, goal_bonus=1.0, potential=None)`
* `step_reward(world, task_id, prefix, action) -> float`
* `return_(world, task_id, actions) -> float`

## `tevforge.reward.sparse.SparseReward`

* `SparseReward(goal_bonus=1.0)`
* `step_reward(...)`, `return_(...)`

## `tevforge.trainer.grpo.GRPO`

* `GRPO(policy, world, reward=None, group_size=8, learning_rate=0.2)`
* `step(task_id)`, `train(task_ids, max_steps=None)`

## `tevforge.bench.benchmarks`

* `completion_rate`, `progress_rate`, `reward_sparsity`,
  `advantage_density`, `wasted_group_rate`, `run_all`
