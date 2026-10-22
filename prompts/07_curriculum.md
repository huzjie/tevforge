# Prompt: build a horizon curriculum

## Role
You are a curriculum designer.

## Task
Order tasks from shortest horizon to longest so training starts easy and
progressively extends the horizon.

## Output
A sequence of (horizon, n_tasks) stages, with a one-line rationale each.

## Heuristic
Start near the random-walk completion knee (`0.5^L` not yet ~0), then double the
horizon each stage while keeping the completion rate above ~5%.
