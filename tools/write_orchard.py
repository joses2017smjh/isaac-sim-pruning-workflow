#!/usr/bin/env python3
"""Write the V-trellis orchard USDA (posts, wires, ground). Does not need pxr."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "isaaclab_pruning"
sys.path.insert(0, str(SOURCE_ROOT))

from isaaclab_pruning.usd.orchard import write_orchard_usda  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, nargs="?", default=Path("generated/orchard/v_trellis.usda"))
    parser.add_argument("--tree-usda", action="append", default=[], help="Optional tree USDA to reference")
    args = parser.parse_args(argv)
    path = write_orchard_usda(args.output, tree_usda_paths=args.tree_usda or None)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
