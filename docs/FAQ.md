# FAQ

**Is this the real Tev model?**
No. It is a zero-dependency *reference implementation* of the method (answer-
world-first synthesis + PPRS + long-horizon GRPO), not the trained weights.

**Why zero dependencies?**
So the whole pipeline runs anywhere Python runs, and the reward math stays
auditable. Swap the mock policy for a real model via the `model/backends.py`
integration points.

**Why is completion 0 at skill 0.5?**
A random walk has solve probability `0.5^L`. For a 16-step task that is
`1.5e-5`, which is the *point*: this is exactly why sparse reward wastes
groups at training start.

**Do the declared 27B/320B numbers mean anything?**
They are informational (from the paper) and never instantiated. The reference
implementation keeps tensors tiny on purpose.
