# -*- coding: utf-8 -*-
"""Serving layer: stdlib HTTP server and CLI entry point."""
from .http import make_server
from .cli import main

__all__ = ["make_server", "main"]
