# -*- coding: utf-8 -*-
"""A thin policy model abstraction.

The reference implementation instantiates tiny structures; declared scale
numbers (27B / 320B parameters) live in ``Config.declared`` and are never
materialized.
"""


class PolicyModel:
    def __init__(self, model_id="tev-27b", backend="mock", skill=0.5):
        self.model_id = model_id
        self.backend = backend
        self.skill = skill

    def act(self, world, task_id, step, history, salt=""):
        raise NotImplementedError

    def to_dict(self):
        return {"model_id": self.model_id, "backend": self.backend,
                "skill": self.skill}
