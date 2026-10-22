# -*- coding: utf-8 -*-
"""Numeric helpers for advantage estimation and benchmarking."""
import math


def mean(xs):
    if not xs:
        return 0.0
    return sum(xs) / len(xs)


def variance(xs):
    if not xs:
        return 0.0
    m = mean(xs)
    return sum((x - m) ** 2 for x in xs) / len(xs)


def std(xs):
    return math.sqrt(variance(xs))


def normalize(xs, eps=1e-9):
    """Return a copy of ``xs`` standardized to zero mean / unit variance.

    If the input is constant the result is all zeros (no signal)."""
    m = mean(xs)
    s = std(xs)
    if s < eps:
        return [0.0 for _ in xs]
    return [(x - m) / s for x in xs]
