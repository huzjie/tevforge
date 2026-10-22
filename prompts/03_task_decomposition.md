# Prompt: decompose a long-horizon task into verifiable subgoals

## Role
You are a planning agent.

## Task
Decompose a multi-hour task into a sequence of subgoals, each of which is
independently verifiable. The subgoal chain is the ground-truth plan used for
progress annotation.

## Output format
For each subgoal: `<id> | <description> | <verification check>`

## Constraints
- Each subgoal must be checkable with a short, deterministic function.
- Subgoals must be ordered; later ones may depend on earlier ones.
- Keep the total number between 8 and 64 (a real long horizon).
