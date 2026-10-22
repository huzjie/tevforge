# -*- coding: utf-8 -*-
"""``python -m tevforge`` entry point."""
import sys

from .serving.cli import main

if __name__ == "__main__":
    sys.exit(main())
