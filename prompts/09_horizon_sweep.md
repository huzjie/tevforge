# Prompt: run a horizon sweep

## Role
You are a training engineer.

## Task
Produce the command / config to sweep the task horizon and report completion
rate and wasted-group rate at each horizon.

## Output
A table `horizon | completion(sparse) | completion(dense) | wasted_sparse | wasted_dense`.

## Constraint
Keep the group size and seed fixed so only the horizon varies.
