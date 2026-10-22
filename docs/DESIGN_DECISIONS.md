# Design decisions

## Why a scalar "skill" for the mock policy?

A single scalar makes training curves clean and monotonic and lets every
benchmark share one oracle. `skill = 0.5` is a random walk, `skill = 1.0` is
perfect; completion `= s^L` and progress `= s` are exactly predictable, so tests
can assert monotonicity instead of eyeballing plots.

## Why potential-based shaping instead of hand-written bonuses?

A hand-written dense reward ("+0.1 for every correct step") changes the optimal
policy. Potential-based shaping (`gamma*Phi(s') - Phi(s)`) provably does not.
That distinction is the entire point of PPRS, so the reference implementation
keeps it explicit and testable (`test_pprs_telescopes`).

## Why the answer world comes first?

If completion is not verifiable by construction, the reward must be an LLM
judge, which can be hacked and is non-deterministic. Building the world first
makes every task's answer known and its progress measurable for free.

## Why zero dependencies?

The reward math and determinism stay auditable; the pipeline runs anywhere
Python runs. The cloud backends are documented integration points, not hard
requirements.
