"""Build a small procedural capture on disk. No licensed asset, no simulator.

The scene is a textured patch on a flat depth plane, the same idea the tracker's
own unit tests use. It exists so the exporter, the reader and the node can be
tested anywhere, including in a clean CI container that has no orchard meshes,
no mock-pruner CAD and no recorded run.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

WIDTH, HEIGHT = 64, 48
SEED_PIXEL = (32.0, 24.0)


def _frame_image(shift_x, shift_y, rng):
    """A moving textured patch on a mid-grey background."""
    image = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
    patch = rng.integers(40, 240, (16, 12, 3), dtype=np.uint8)
    top, left = 16 + int(shift_y), 26 + int(shift_x)
    image[top : top + 16, left : left + 12] = patch
    return image


def write_capture(directory, frames=6, *, with_images=True, release_frame=None):
    """Write a capture directory the reader accepts. Returns its path."""
    directory = Path(directory)
    (directory / "frames").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    patch = rng.integers(40, 240, (16, 12, 3), dtype=np.uint8)

    records = []
    for index in range(frames):
        if with_images:
            import cv2

            image = np.full((HEIGHT, WIDTH, 3), 60, dtype=np.uint8)
            top, left = 16 + index, 26
            image[top : top + 16, left : left + 12] = patch
            # cv2 writes BGR, and the reader converts back, so round-tripping
            # through disk preserves the array the test built.
            cv2.imwrite(str(directory / "frames" / f"wrist_{index:05d}.png"), image[:, :, ::-1])
            np.save(directory / "frames" / f"depth_{index:05d}.npy", np.full((HEIGHT, WIDTH), 0.6, dtype=np.float32))

        tof = [[0.25 + 0.001 * (row * 8 + col) for col in range(8)] for row in range(8)]
        valid = [[(row + col) % 2 == 0 for col in range(8)] for row in range(8)]
        records.append(
            {
                "index": index,
                "time_s": round(0.1 * (index + 1), 3),
                "phase": "vision_approach",
                "tool_pose_wxyz": [0.4, 0.3, 0.75, 1.0, 0.0, 0.0, 0.0],
                "command_pose_root_wxyz": [0.4, 0.3, 0.75, 1.0, 0.0, 0.0, 0.0],
                "joint_position_rad": [0.0, -0.5, 1.2, -0.5, -0.7, 0.0],
                "joint_velocity_rad_s": [0.0] * 6,
                "wrist_position_w_m": [0.5, 0.4, 0.62],
                "wrist_rotation_w_ros": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "tof_left_m": tof,
                "tof_right_m": tof,
                "tof_left_valid": valid,
                "tof_right_valid": valid,
                "contact_force_n": 0.0,
                "controller_source_frame_index": index - 1,
                "detachment_requested_after_capture": bool(release_frame is not None and index == release_frame),
                "sensor_stop_reason": None,
                "live_vision": {
                    "measurement": {
                        "state": "tracking",
                        "reason": None,
                        "pixel_xy": list(SEED_PIXEL),
                        "feature_count": 20,
                        "confidence": 0.9,
                        "target_position_world_m": [0.25, 0.63, 0.81],
                    }
                },
                "visual_servo_decision": {
                    "state": "tracking",
                    "reason": None,
                    "delta_world_m": [0.0, 0.0, 0.0],
                },
            }
        )

    report = {
        "job_id": "fixture",
        "frame_count": frames,
        "photometric_normalization": "raw",
        "camera": {
            "wrist_resolution": [WIDTH, HEIGHT],
            "wrist_intrinsics": [[40.0, 0.0, WIDTH / 2], [0.0, 40.0, HEIGHT / 2], [0.0, 0.0, 1.0]],
        },
        "blender_scene": {
            "daylight": {"preset": "source"},
            "target": {
                "id": "fixture_SPUR_component_1",
                "axis_w": [0.0, 0.0, 1.0],
                "radius_m": 0.006,
                "identity_source": "procedural_test_fixture_not_a_classifier",
            },
        },
    }
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (directory / "frames.json").write_text(json.dumps({"fps": 10, "frames": records}, indent=2), encoding="utf-8")
    return directory
