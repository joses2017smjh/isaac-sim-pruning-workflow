#!/usr/bin/env python3
"""Validate a tree0 replay render manifest and write the read-only evaluation plan.

Every frame the render registered is checked for its RGB, shared depth and
mask, and its recorded Isaac counterpart. A frame that is missing stays in the
plan's ``missing`` list and in the denominator. Nothing here looks at a
prediction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_generalization import sha256  # noqa: E402


def check_frame(frame):
    rgb = np.asarray(Image.open(frame["rgb"]).convert("RGB"))
    depth = np.load(frame["depth"], allow_pickle=False)
    mask = np.asarray(Image.open(frame["mask"])) > 0
    if not (rgb.shape[:2] == depth.shape == mask.shape == (frame["height"], frame["width"])):
        raise ValueError(f"RGB, depth, mask and recorded size disagree: {frame['rgb']}")
    if rgb.std() <= 3 or not mask.any():
        raise ValueError(f"Blank render or empty tree mask: {frame['rgb']}")
    isaac = np.load(frame["isaac_depth"], allow_pickle=False)
    if isaac.shape != depth.shape and frame.get("width") == 480:
        raise ValueError(f"Isaac depth shape differs from the render: {frame['isaac_depth']}")
    return float(np.median(depth[mask])) if mask.any() else None


def build(manifest_path, output, expected_frames=None):
    manifest_path, output = Path(manifest_path), Path(output)
    data = json.loads(manifest_path.read_text())
    frames, missing, by_condition = [], [], {}
    for frame in data["frames"]:
        try:
            check_frame(frame)
        except (OSError, ValueError) as error:
            missing.append({"view_id": frame["view_id"], "condition": frame["condition"], "reason": str(error)})
            continue
        frames.append(frame)
        by_condition[frame["condition"]] = by_condition.get(frame["condition"], 0) + 1
    registered = expected_frames if expected_frames is not None else len(data["frames"])
    if len(data["frames"]) < registered:
        missing.append(
            {
                "view_id": "*",
                "condition": "*",
                "reason": f"render registered {registered} frames and produced {len(data['frames'])}",
            }
        )
    plan = {
        "schema_version": 1,
        "scope": (
            "Renderer control: the original orchard tree0 rendered by Blender Cycles at the recorded Isaac "
            "wrist poses with the Isaac camera model, under the palm bark Isaac carried and the bark_brown_02 "
            "of the training renders. Not an Isaac render; the robot, tool and dome light are absent."
        ),
        "source_renders": [
            {
                "path": str(manifest_path),
                "sha256": sha256(manifest_path),
                "render_ok": bool(data.get("ok")),
                "runs": data.get("runs"),
                "lighting": data.get("lighting"),
            }
        ],
        "registered_frames": registered,
        "scored_frames": len(frames),
        "frames_by_condition": by_condition,
        "missing": missing,
        "frames": frames,
        "depth_convention": data.get("depth_convention"),
        "target_scope": (
            "The Stage A tracked pixel of the same frame when the Cycles depth agrees with the target's optical Z "
            "within 3 cm; otherwise the projected centre, and invisible projections are explicit failures"
        ),
        "temporal_scope": "Recorded approach frames in order; the render does not move, the recorded camera did",
    }
    with output.open("x") as stream:
        json.dump(plan, stream, indent=2)
    return plan


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=None, help="Frames the frozen plan registered")
    args = parser.parse_args(argv)
    plan = build(args.manifest, args.output, args.expected_frames)
    print(
        json.dumps(
            {"ok": not plan["missing"], "registered": plan["registered_frames"], "scored": plan["scored_frames"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
