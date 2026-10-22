# Prompt: annotate a trajectory with dense progress

## Role
You are a reward annotator.

## Task
Given a ground-truth plan, produce the per-step shaped reward using the
potential Phi(s) = (number of correct steps so far) / (total steps).

## Output
A table `step | action | Phi(before) | Phi(after) | shaped_reward`.

## Note
The ground-truth plan is always correct, so its shaped step reward is +1/L each
step, plus the goal bonus on the final step.
