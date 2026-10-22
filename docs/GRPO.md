# Long-horizon GRPO on dense rewards

## Standard GRPO (sparse)

For a task, sample G rollouts, compute sparse returns `r_i in {0, 1}`, then

```
advantage_i = (r_i - mean(r)) / std(r)
```

If the group all fails (`r_i = 0`), `std = 0` and every advantage is 0: the
group is **wasted**. At solve probability `p`, waste `= (1-p)^G`.

## GRPO on PPRS returns

Returns are now `progress_i + completion_i`, spread out across the group. Even
when nobody completes, rollouts differ in how much progress they made, so
advantages are non-zero and training proceeds from the very first batch.

## Implementation notes

* `grpo_advantages` maps a constant vector to all-zero advantages (the wasted
  group, detected by `wasted_group`).
* The mock policy update is a deterministic monotonic relaxation
  `skill += lr * (1 - skill)` so curves are clean and reproducible.
