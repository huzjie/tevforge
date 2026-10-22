# Prompt: answer-world-first environment synthesis

## Role
You are a reinforcement-learning environment designer.

## Task
Design a *verifiable* long-horizon agent task. Before writing any rollout
logic, define an "answer world" that can (a) generate ground-truth solutions,
(b) measure progress at every step, and (c) verify completion exactly.

## Output format
1. **Goal** (one sentence)
2. **Action space** (the finite set of primitive actions)
3. **Ground-truth generator** (how a solution is produced deterministically)
4. **Progress function** Phi(s) (scalar, monotonic in completion)
5. **Completion predicate** (exact, no LLM judge)

## Constraints
- Completion must be machine-verifiable, never judged by an LLM.
- Progress must be dense: non-zero on the majority of steps.
- Everything must be reproducible from a seed.

## Example stub
```
goal: sort the JSON array in-place using only `swap(i, j)`.
actions: [swap_i_j for all pairs]
ground_truth: insertion-sort order of swaps
progress: fraction of correctly positioned elements
completion: array is sorted
```
