# Prompt: design an ablation study

## Role
You are a research engineer.

## Task
Design the minimal set of ablations to prove PPRS (dense shaping) beats sparse
reward for long-horizon GRPO.

## Output
For each ablation: `<name> | <what changes> | <metric to compare>`.

## Required ablations
- sparse vs dense step reward (reward sparsity)
- group advantage density
- wasted-group rate at skill = 0.5
- completion-rate monotonicity in skill
