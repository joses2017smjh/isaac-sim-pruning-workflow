#!/usr/bin/env python3
"""Convert a directory of L-Py metadata JSON files to per-tree USDA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "isaaclab_pruning"
sys.path.insert(0, str(SOURCE_ROOT))

from isaaclab_pruning.assets import load_tree_splits  # noqa: E402
from isaaclab_pruning.usd.cylinders import main as convert_one  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metadata_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--include-held-out", action="store_true")
    args = parser.parse_args(argv)

    splits = load_tree_splits()
    files = sorted(args.metadata_dir.glob("*_metadata.json"))
    if not files:
        raise SystemExit(f"No *_metadata.json files in {args.metadata_dir}")

    converted = []
    for metadata in files:
        tree_id = metadata.stem.removesuffix("_metadata")
        if not args.include_held_out and splits.is_held_out(tree_id):
            continue
        output = args.output_dir / f"{tree_id}.usda"
        convert_one([str(metadata), str(output), "--tree-id", tree_id])
        converted.append(tree_id)
    print(json.dumps({"converted": converted, "count": len(converted)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
