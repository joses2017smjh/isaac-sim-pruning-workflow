#!/usr/bin/env python3
"""Record actual DDS graph parity and received-message fault controls as JSON."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if ROOT.name == "tools":
    ROOT = ROOT.parent
sys.path[:0] = [str(ROOT / "ros2/pruning_sil"), str(ROOT / "ros2/pruning_sil/test")]
import rclpy  # noqa: E402
from pruning_sil import topics as T  # noqa: E402
from pruning_sil.capture import Capture  # noqa: E402
from pruning_sil.export_rosbag2 import _stamp  # noqa: E402
from pruning_sil.replay import compare_decision, invalid_frame_is_held  # noqa: E402
from pruning_sil.servo_node import array_to_image, image_to_array  # noqa: E402
from test_stream_graph import Graph  # noqa: E402


def fault_control(capture, name):
    graph = Graph(capture.directory)
    try:
        graph.publish(capture, 0)

        def corrupt(topic, message):
            if name == "rgb_blackout" and topic == T.TOPIC_WRIST_IMAGE:
                return array_to_image(
                    np.zeros_like(image_to_array(message)),
                    "rgb8",
                    message.header.frame_id,
                    message.header.stamp,
                )
            if topic == T.TOPIC_WRIST_DEPTH:
                if name in ("depth_dropout", "negative_depth"):
                    value = np.nan if name == "depth_dropout" else -1.0
                    return array_to_image(
                        np.full_like(image_to_array(message), value),
                        "32FC1",
                        message.header.frame_id,
                        message.header.stamp,
                    )
                if name == "wrong_depth_stamp":
                    message.header.stamp = _stamp(capture.stamp_ns(0))
                elif name == "wrong_camera_frame":
                    message.header.frame_id = "unregistered_camera"
                elif name == "malformed_depth":
                    message.data = bytes(4)
            if name == "invalid_intrinsics" and topic == T.TOPIC_WRIST_CAMERA_INFO:
                message.k = [0.0] * 9
            return message

        for index in range(1, 12):
            graph.publish(
                capture,
                index,
                change=corrupt,
                omit=(T.TOPIC_WRIST_DEPTH,) if name == "missing_depth" else (),
            )
        graph.spin(0.4)
        decisions = [d for d in graph.decisions if d["frame_index"] >= 2]
        authorized = [d for d in decisions if not invalid_frame_is_held(d)]
        nonzero = [d for d in decisions if np.linalg.norm(d.get("delta_world_m", [0, 0, 0])) > 1e-12]
        return {
            "fault": name,
            "affected_frame_indices": list(range(2, 12)),
            "received_decisions": len(decisions),
            "held": sum(invalid_frame_is_held(d) for d in decisions),
            "authorized_motion_states": authorized,
            "nonzero_command_deltas": nonzero,
            "published_poses": len(graph.poses),
            "per_frame": decisions,
            "passed": len(decisions) == 10 and not authorized and not nonzero,
        }
    finally:
        graph.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite existing evidence")
    capture = Capture(args.capture_dir)
    rclpy.init()
    try:
        graph = Graph(capture.directory)
        try:
            # Images are loaded by the publisher; disk access from the servo is forbidden.
            def refuse(*args):
                raise AssertionError("Graph servo attempted to read image files")

            graph.servo.capture.rgb = refuse
            graph.servo.capture.depth = refuse
            for index in range(len(capture)):
                graph.publish(capture, index, reverse=bool(index % 2))
            graph.spin(0.4)
            compared = []
            for decision in graph.decisions:
                index = decision["frame_index"]
                if index == 0:
                    continue
                comparison = compare_decision(
                    decision,
                    capture.recorded_decision(index),
                    position_tolerance_m=0.002,
                )
                compared.append({"frame_index": index, **comparison})
            errors = [r["delta_error_m"] for r in compared if r["delta_error_m"] is not None]
            stamps = [p.header.stamp.sec * 10**9 + p.header.stamp.nanosec for p in graph.poses]
            normal = {
                "recorded_frames": len(capture),
                "received_decisions": len(graph.decisions),
                "frames_compared": len(compared),
                "states_agree": sum(r["state_matches"] for r in compared),
                "command_tolerance_m": 0.002,
                "commands_within_tolerance": all(r["delta_matches"] for r in compared),
                "maximum_command_delta_error_m": max(errors) if errors else None,
                "published_poses": len(graph.poses),
                "pose_stamps_match_capture": stamps == [capture.stamp_ns(i) for i in range(1, len(capture))],
                "file_image_reads_in_servo": 0,
                "manual_step_frame_calls": 0,
                "per_frame": compared,
            }
            normal["passed"] = (
                len(compared) == len(capture) - 1
                and all(r["matches"] for r in compared)
                and normal["pose_stamps_match_capture"]
            )
        finally:
            graph.close()
        faults = [
            fault_control(capture, name)
            for name in (
                "rgb_blackout",
                "depth_dropout",
                "negative_depth",
                "missing_depth",
                "wrong_depth_stamp",
                "wrong_camera_frame",
                "malformed_depth",
                "invalid_intrinsics",
            )
        ]
    finally:
        rclpy.try_shutdown()
    source_files = [
        ROOT / "ros2/pruning_sil/pruning_sil/replay.py",
        ROOT / "ros2/pruning_sil/pruning_sil/servo_node.py",
        ROOT / "ros2/pruning_sil/test/test_stream_graph.py",
        Path(__file__).resolve(),
    ]
    document = {
        "schema_version": 1,
        "evaluation_type": "actual_ros2_dds_graph_parity_and_fault_injection",
        "capture_dir": str(capture.directory),
        "scope": (
            "Advisory RGB-D visual servo ROS2 graph on one recorded simulator capture. Incoming ROS RGB, "
            "optical-Z depth, intrinsics and tool poses drive decisions. Target identity, camera world transforms "
            "and contact context remain recorded metadata. ToF is telemetry only in this ROS node; Isaac "
            "environment ToF/release gates are outside this test. No hardware or actuation."
        ),
        "normal_graph": normal,
        "fault_controls": faults,
        "source_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files},
        "passed": normal["passed"] and all(r["passed"] for r in faults),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {
                "normal_graph": {k: v for k, v in normal.items() if k != "per_frame"},
                "fault_controls": [{k: v for k, v in c.items() if k != "per_frame"} for c in faults],
                "passed": document["passed"],
            },
            indent=2,
        )
    )
    return 0 if document["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
