# -*- coding: utf-8 -*-
"""Benchmarks quantifying the dense-reward / PPRS advantage."""
from .benchmarks import (completion_rate, progress_rate, reward_sparsity,
                         advantage_density, wasted_group_rate, run_all)

__all__ = ["completion_rate", "progress_rate", "reward_sparsity",
           "advantage_density", "wasted_group_rate", "run_all"]
