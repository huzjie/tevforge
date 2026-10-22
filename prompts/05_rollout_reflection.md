# Prompt: reflect on a failed long-horizon rollout

## Role
You are a self-improving agent.

## Task
Given a failed rollout (the sequence of actions and where progress stalled),
explain *why* progress stalled and propose one concrete fix.

## Output
1. **Stall point** (the step after which progress stopped)
2. **Root cause** (one sentence)
3. **Concrete fix** (one changed action or policy)
4. **New expected progress** (a number)

## Constraint
Do not restate the task. Focus on the *delta* that would resume progress.
