#!/usr/bin/env python3
"""Fit mouth/failure OBBs from fetched pybullet-tree-sim cutter STLs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "isaaclab_pruning"
sys.path.insert(0, str(SOURCE_ROOT))

from isaaclab_pruning.geometry.cutter import fit_oriented_box_from_stl  # noqa: E402

DEFAULT_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "src" / "pybullet_tree_sim"
MOUTH = "pybullet_tree_sim/urdf/ur5e/collision/cutter-mouth-collision.stl"
FAILURE = "pybullet_tree_sim/urdf/ur5e/collision/cutter-failure-zone.stl"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args(argv)
    mouth_path = args.root / MOUTH
    failure_path = args.root / FAILURE
    if not mouth_path.is_file() or not failure_path.is_file():
        print(
            json.dumps(
                {
                    "fitted": False,
                    "reason": (
                        "Fetch OSUrobotics/pybullet-tree-sim first: "
                        "python tools/fetch_sources.py --source pybullet_tree_sim"
                    ),
                },
                indent=2,
            )
        )
        return 1
    mouth = fit_oriented_box_from_stl(mouth_path)
    failure = fit_oriented_box_from_stl(failure_path)
    print(
        json.dumps(
            {
                "fitted": True,
                "mouth": {
                    "center": mouth.center.tolist(),
                    "half_extents": mouth.half_extents.tolist(),
                    "vertices": mouth.vertex_count,
                },
                "failure": {
                    "center": failure.center.tolist(),
                    "half_extents": failure.half_extents.tolist(),
                    "vertices": failure.vertex_count,
                },
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
