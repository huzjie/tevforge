# -*- coding: utf-8 -*-
"""Configuration for tevforge.

A plain dataclass plus a minimal YAML loader. The managed Python runtime has no
pyyaml, so we fall back to a tiny hand-rolled YAML subset parser
(``utils/yamlish.py``). Declared scale numbers (total_params_b, ale_cm_pass5,
...) are informational and never instantiated as real tensors.
"""
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

try:  # pragma: no cover - optional
    import yaml as _pyyaml  # type: ignore
except Exception:  # pragma: no cover
    _pyyaml = None


@dataclass
class WorldConfig:
    action_space_size: int = 8
    horizon_min: int = 8
    horizon_max: int = 24
    seed: int = 20261022


@dataclass
class ModelConfig:
    backend: str = "mock"
    skill: float = 0.5
    model_id: str = "tev-27b"
    temperature: float = 0.0


@dataclass
class RewardConfig:
    shaping: str = "pprs"  # pprs | sparse | dense
    gamma: float = 1.0
    goal_bonus: float = 1.0


@dataclass
class TrainConfig:
    group_size: int = 8
    learning_rate: float = 0.2
    max_steps: int = 40
    seed: int = 20261022


@dataclass
class BenchConfig:
    n_tasks: int = 64
    group_size: int = 8
    seed: int = 20261022


@dataclass
class DeclaredConfig:
    total_params_b: float = 27.0
    ale_cm_pass5: float = 0.28
    env_domains: int = 7
    gym_envs: int = 1170

    def to_dict(self):
        return asdict(self)


@dataclass
class Config:
    world: WorldConfig = field(default_factory=WorldConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    bench: BenchConfig = field(default_factory=BenchConfig)
    declared: DeclaredConfig = field(default_factory=DeclaredConfig)

    @classmethod
    def from_dict(cls, d):
        d = d or {}
        def sub(key, factory):
            return factory(**{k: v for k, v in (d.get(key) or {}).items()})
        return cls(
            world=sub("world", WorldConfig),
            model=sub("model", ModelConfig),
            reward=sub("reward", RewardConfig),
            train=sub("train", TrainConfig),
            bench=sub("bench", BenchConfig),
            declared=sub("declared", DeclaredConfig),
        )

    def to_dict(self):
        return asdict(self)


def load_config(path=None):
    if path is None:
        here = Path(__file__).parent
        path = here / "configs" / "default.yaml"
        if not path.exists():
            path = here.parent / "configs" / "default.yaml"
    p = Path(path)
    if not p.exists():
        return Config()
    text = p.read_text(encoding="utf-8")
    data = None
    if _pyyaml is not None:
        try:
            data = _pyyaml.safe_load(text)
        except Exception:
            data = None
    if data is None:
        from .utils.yamlish import parse
        data = parse(text)
    return Config.from_dict(data)
