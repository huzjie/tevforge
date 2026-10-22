# Progress Potential Reward Shaping (PPRS)

## Why reward shaping

A long-horizon task has a single success signal at the end. Reward shaping adds
a *potential-based* term so intermediate progress emits reward:

```
r' = r_env + gamma * Phi(s') - Phi(s)
```

`Phi` is a potential function measuring "how close to the goal". Here
`Phi(s)` is simply the fraction of the ground-truth plan reproduced so far.

## Policy invariance

Because the added term is the difference of a potential (telescoping), the
shaped MDP has the **same optimal policy** as the original. With `gamma = 1`
and `Phi(empty) = 0`, the shaped return telescopes exactly:

```
shaped_return = sparse_return + Phi(s_T)
```

i.e. final progress is always credited, even when the task is not completed.

## What it fixes

| reward | step density | group variance | wasted groups (skill=0.5) |
|--------|--------------|----------------|---------------------------|
| sparse | 1 / L        | ~0             | ~100%                     |
| PPRS   | ~progress    | high           | ~0%                       |

The dense signal means even a group with no completions has spread-out returns,
so GRPO has a usable gradient.
