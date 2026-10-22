# Tev-Gym: answer-world-first environment synthesis

The paper's `Tev-Gym` (1,170 environments across 7 domains) is reproduced here
in miniature. The ordering matters: **build the answer world first, then
generate tasks.**

## Recipe

1. Define an answer world with a verifiable completion predicate and a
   progress measure.
2. For each domain, spawn task ids whose solution is known by construction.
3. Annotate each `(task_id, step)` with its dense shaped reward.

```python
from tevforge.world.answer_world import AnswerWorld
from tevforge.synth.gym import TevGym

world = AnswerWorld()
gym = TevGym(world=world)
tasks = gym.build(tasks_per_domain=8)     # 7 domains * 8 = 56 verifiable tasks
train, heldout = gym.split(tasks)          # heldout for eval only
```

Because completion is verifiable, the reward is *never* an LLM-judge guess: it
is exact, deterministic and free of reward-hacking.
