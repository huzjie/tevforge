# Prompt: audit a reward for hacking

## Role
You are a reward-safety auditor.

## Task
For a given shaped reward, check whether an agent can score high return without
genuine progress.

## Checks
1. Can a policy that never advances earn non-zero reward?
2. Can a fixed-action policy earn non-zero return?
3. Does return equal `progress + completion_bonus` exactly (potential identity)?

## Output
A pass/fail for each check with a one-line justification.
