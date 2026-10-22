# Adding a new Tev-Gym domain

1. Decide the domain's action space (a finite list of primitive actions).
2. Ensure the answer world can generate a ground-truth plan for it
   (deterministic in `task_id`).
3. Add the domain to `TevGym.DOMAINS` (or pass a custom `domains` list).
4. Add a verifier: `is_complete` must be a pure function of the rollout.
5. Add a dense progress measure that is non-zero on most steps.

```python
from tevforge.world.answer_world import AnswerWorld
from tevforge.synth.gym import TevGym

world = AnswerWorld()                  # default action space
gym = TevGym(world=world, domains=["terminal", "swe", "my_domain"])
tasks = gym.build(tasks_per_domain=8)
```
