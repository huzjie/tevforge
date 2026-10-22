# Prompt: design a progress potential (PPRS)

## Role
You are a reward-shaping engineer.

## Task
Given a task with a sparse terminal reward, design a potential function Phi(s)
that turns it into a dense, potential-based shaped reward:

```
r' = r_env + gamma * Phi(s') - Phi(s)
```

## Requirements
- Phi(initial) = 0 and Phi(goal) = 1.
- Phi must be *policy-invariant*: it may not change the optimal policy.
- Phi must be cheap to compute at every step.

## Output
1. State features that define Phi.
2. The formula for Phi(s).
3. A proof sketch that the shaped reward telescopes to
   `sparse_return + Phi(s_T)` when gamma = 1.
4. Two example states with their Phi values.
