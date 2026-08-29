#!/usr/bin/env python3
"""Score one Blender orchard pose against L-Py trunk cylinders (median < 2 mm)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[1] / "source" / "isaaclab_pruning"
sys.path.insert(0, str(SOURCE_ROOT))

from isaaclab_pruning.eval.blender_trunk import (  # noqa: E402
    load_blender_pose_centroids,
    score_blender_trunk,
)
from isaaclab_pruning.geometry.cylinders import load_cylinders  # noqa: E402
from isaaclab_pruning.usd.cylinders import DEFAULT_TREE_TILT_X_DEG  # noqa: E402

DEFAULT_METADATA = Path("/nfs/hpc/share/sanchej7/Computer_Vision/trees/metadata/lpy_envy_00000_metadata.json")
DEFAULT_POSE = Path(
    "/nfs/hpc/share/sanchej7/Computer_Vision/Data/full_spur/ann/bark_brown_02/"
    "lpy_envy_00000/box/lpy_envy_00000_shot01.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--blender-pose", type=Path, default=DEFAULT_POSE)
    parser.add_argument("--tilt-x-deg", type=float, default=DEFAULT_TREE_TILT_X_DEG)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args(argv)

    centroids, pose = load_blender_pose_centroids(args.blender_pose)
    report = score_blender_trunk(load_cylinders(args.metadata), centroids, tilt_x_deg=args.tilt_x_deg)
    report.update(
        {
            "ok": bool(report["pass"]),
            "metadata": str(args.metadata),
            "blender_pose": str(args.blender_pose),
            "tree_id": pose.get("tree_id"),
            "shot": pose.get("shot"),
            "camera_location": pose.get("camera", {}).get("location"),
            "note": (
                "cylinders_world is centroids-only and is not used as a cylinder sidecar; "
                "this is a rigid pose check after orchard tilt."
            ),
        }
    )
    text = json.dumps(report, indent=2) + "\n"
    print(text, flush=True)
    destination = args.json_out
    if destination is None:
        destination = Path(__file__).resolve().parents[1] / "docs" / "evidence" / "blender_trunk_mm_lpy_envy_00000.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(text, encoding="utf-8")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
