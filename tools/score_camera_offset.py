#!/usr/bin/env python3
"""Score wrist-camera candidates by ray-casting cylinder colliders. No renderer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "isaaclab_pruning"
sys.path.insert(0, str(SOURCE_ROOT))

from isaaclab_pruning.geometry.cut_point import oracle_cut_candidates  # noqa: E402
from isaaclab_pruning.geometry.cylinders import load_cylinders, transform_cylinders  # noqa: E402
from isaaclab_pruning.robot import load_ur5e_pruner_spec  # noqa: E402
from isaaclab_pruning.sensors.wrist_camera import (  # noqa: E402
    CANDIDATES,
    RANGE_BAND_M,
    score_candidate_on_tree,
    select_wrist_camera,
)
from isaaclab_pruning.usd.cylinders import DEFAULT_TREE_TILT_X_DEG, _local_transform  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("/nfs/hpc/share/sanchej7/Computer_Vision/trees/metadata/lpy_envy_00000_metadata.json"),
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args(argv)

    cylinders = transform_cylinders(
        load_cylinders(args.metadata),
        _local_transform(DEFAULT_TREE_TILT_X_DEG, (0.0, 0.0, 0.0)),
    )
    cuts = oracle_cut_candidates(cylinders, organ_classes=("spur", "branch"))
    spec = load_ur5e_pruner_spec()
    intrinsic = np.array(
        [[500.0, 0.0, args.width / 2.0], [0.0, 500.0, args.height / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    scores = [
        score_candidate_on_tree(
            candidate,
            cylinders,
            cuts,
            mouth_offset_eef=spec.mouth_offset_m,
            mouth_half_extents=spec.mouth_half_extents_m,
            intrinsic=intrinsic,
            width=args.width,
            height=args.height,
        )
        for candidate in CANDIDATES
    ]
    winner = select_wrist_camera(scores)
    if winner is not None:
        for score in scores:
            score["selected"] = score["name"] == winner["name"]
    report = {
        "ok": winner is not None,
        "method": "raycast_cylinders",
        "renderer": None,
        "tree": str(args.metadata),
        "n_cylinders": len(cylinders),
        "n_cuts": len(cuts),
        "range_band_m": list(RANGE_BAND_M),
        "scores": scores,
        "selected": winner,
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text, flush=True)
    destination = args.json_out
    if destination is None:
        destination = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "camera_offset_raycast.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return 0 if winner is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
