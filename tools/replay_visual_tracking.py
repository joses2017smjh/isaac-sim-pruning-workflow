#!/usr/bin/env python3
"""Compare tracker maintenance on actual saved RGB-D; never replay robot control.

Frames recorded after a stopped robot do not prove counterfactual motion or
cutting. This diagnostic changes neither source captures nor task outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/isaaclab_pruning"))
from isaaclab_pruning.perception.visual_servo import VisualServoConfig, VisualServoTracker
from isaaclab_pruning.task.simulated_cut import tool_mouth_geometry


def replay(source):
    source = Path(source)
    report = json.loads((source / "report.json").read_text())
    frames = json.loads((source / "frames.json").read_text())["frames"]
    if not frames:
        raise ValueError("Capture contains no frames")
    camera = report["camera"]
    pose = report["initial_tool_pose_wxyz"][0]
    rotation = np.column_stack([tool_mouth_geometry(pose, closing_axis_tool=axis)[1] for axis in np.eye(3)])
    initial_transform = np.eye(4)
    initial_transform[:3, :3] = rotation @ np.asarray(camera["wrist_rotation_in_tool_ros"])
    initial_transform[:3, 3] = np.asarray(pose[:3]) + rotation @ camera["wrist_position_in_tool_m"]
    initial_rgb = np.array(Image.open(source / "preview_wrist.png").convert("RGB"))
    initial_depth = np.load(source / "preview_depth.npy", allow_pickle=False)
    output = {
        "source_job_id": report.get("job_id"),
        "source_task_outcome": report.get("task_outcome"),
        "recorded_frames": len(frames),
        "source_sha256": {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("report.json", "frames.json", "preflight.json")
        },
        "tracker_sha256": hashlib.sha256(
            Path(sys.modules[VisualServoTracker.__module__].__file__).read_bytes()
        ).hexdigest(),
        "scope": "Offline saved-image replay only; post-stop frames do not prove continued motion, release or retreat.",
        "results": [],
    }
    for maintain in (False, True):
        tracker = VisualServoTracker(
            VisualServoConfig(depth_radius_px=1, feature_quality_level=0.005, replenish_features=maintain)
        )
        tracker.initialize(initial_rgb, report["vision_initialization"]["pixel_xy"], initial_depth)
        result = tracker.update(initial_rgb, initial_depth, camera["wrist_intrinsics"], initial_transform)
        added = 0
        processed = 0
        digest = hashlib.sha256()
        for frame in frames:
            if result["state"] != "tracking":
                break
            index = frame["index"]
            transform = np.eye(4)
            transform[:3, :3] = frame["wrist_rotation_w_ros"]
            transform[:3, 3] = frame["wrist_position_w_m"]
            rgb_path = source / f"frames/wrist_{index:05d}.png"
            depth_path = source / f"frames/depth_{index:05d}.npy"
            for path in (rgb_path, depth_path):
                digest.update(path.name.encode())
                digest.update(hashlib.sha256(path.read_bytes()).digest())
            result = tracker.update(
                np.array(Image.open(rgb_path).convert("RGB")),
                np.load(depth_path, allow_pickle=False),
                camera["wrist_intrinsics"],
                transform,
            )
            added += len(result.get("pending_feature_pixels_for_next_frame", []))
            processed += 1
        output["results"].append(
            {
                "maintenance": maintain,
                "processed_frames": processed,
                "processed_image_digest": digest.hexdigest(),
                "new_candidates": added,
                "final_measurement": result,
            }
        )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing replay")
    result = replay(args.input_dir)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            [
                {key: item[key] for key in ("maintenance", "processed_frames", "new_candidates")}
                | {"state": item["final_measurement"]["state"], "features": item["final_measurement"]["feature_count"]}
                for item in result["results"]
            ]
        )
    )


if __name__ == "__main__":
    main()
