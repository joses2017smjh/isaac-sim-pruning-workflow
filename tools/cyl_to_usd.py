#!/usr/bin/env python3
"""Repository-local entry point for the cylinder-to-USD converter."""

from __future__ import annotations

import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "isaaclab_pruning"
sys.path.insert(0, str(SOURCE_ROOT))

from isaaclab_pruning.usd.cylinders import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
