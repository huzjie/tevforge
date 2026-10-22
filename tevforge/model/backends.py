# -*- coding: utf-8 -*-
"""Pluggable backends: mock (trainable), cpu, openai, vllm, transformers.

Only ``mock`` and ``cpu`` are self-contained. The cloud backends are thin
adapters that delegate to a local mock policy with a note --- they document the
integration point (your token / endpoint goes in config) without requiring
network access in the reference implementation.
"""
from ..agent.policy import MockPolicy

_BACKENDS = {}


def register_backend(name):
    def deco(cls):
        _BACKENDS[name] = cls
        return cls
    return deco


@register_backend("mock")
class MockBackend:
    def __init__(self, cfg):
        self.policy = MockPolicy(skill=getattr(cfg, "skill", 0.5))

    def make_policy(self):
        return self.policy


@register_backend("cpu")
class CpuBackend:
    """Pure-python policy (same semantics as mock, no randomness difference)."""

    def __init__(self, cfg):
        self.policy = MockPolicy(skill=getattr(cfg, "skill", 0.5))

    def make_policy(self):
        return self.policy


@register_backend("openai")
class OpenAIBackend:
    """Adapter for an OpenAI-compatible endpoint (delegates to mock here)."""

    def __init__(self, cfg):
        self.policy = MockPolicy(skill=getattr(cfg, "skill", 0.5))
        self.endpoint = getattr(cfg, "endpoint", "https://api.openai.com/v1")

    def make_policy(self):
        return self.policy


@register_backend("vllm")
class VllmBackend:
    """Adapter for a vLLM OpenAI-compatible server (delegates to mock here)."""

    def __init__(self, cfg):
        self.policy = MockPolicy(skill=getattr(cfg, "skill", 0.5))
        self.base_url = getattr(cfg, "base_url", "http://localhost:8000/v1")

    def make_policy(self):
        return self.policy


@register_backend("transformers")
class TransformersBackend:
    """Adapter for HuggingFace transformers (delegates to mock here)."""

    def __init__(self, cfg):
        self.policy = MockPolicy(skill=getattr(cfg, "skill", 0.5))
        self.model_id = getattr(cfg, "model_id", "tev-27b")

    def make_policy(self):
        return self.policy


def get_backend(name, cfg=None):
    if name not in _BACKENDS:
        raise ValueError("unknown backend %r (available: %s)" % (name, list(_BACKENDS)))
    return _BACKENDS[name](cfg)


def list_backends():
    return sorted(_BACKENDS)
