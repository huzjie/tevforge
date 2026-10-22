# -*- coding: utf-8 -*-
"""List all backends and instantiate each.

Run:  python examples/05_backends.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tevforge.model.backends import get_backend, list_backends

for name in list_backends():
    b = get_backend(name, None)
    p = b.make_policy()
    print("backend %-14s -> policy skill=%.2f" % (name, p.skill))
