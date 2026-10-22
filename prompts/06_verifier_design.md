# Prompt: write a completion verifier

## Role
You are a verification engineer.

## Task
Write a deterministic predicate `is_complete(task_id, actions) -> bool` that
returns true exactly when a rollout solves the task.

## Requirements
- Pure function of the rollout; no LLM call, no network, no randomness.
- Must agree with the ground-truth generator for the same `task_id`.
- Fast enough to run on millions of rollouts.

## Output
1. Pseudocode of the verifier.
2. Three positive and three negative examples.
