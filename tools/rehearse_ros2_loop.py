"""Rehearse the live Isaac loop's ROS 2 side on CPU, with a recorded capture standing in for Isaac.

Starts ``pruning_sil.live_bringup`` (controller + C++ safety gate), then drives it
through ``hpc/inner/ros2_loop_bridge.RosLoopBridge`` — the same class Isaac Sim
uses — with a recorded capture's sensor bundles in the recorder's causal order:
the preview at t = 0, then each frame's capture. Because the recorded poses do
not respond to commands this is open loop: it checks the message path, the
lockstep handshake and that the ROS 2 controller reproduces the recorded
in-process decisions. It does not show closed-loop behaviour; the GPU runs do.

The t = 0 camera pose is rebuilt from the recorded tool pose and wrist mount, and
the initial ToF sample (not recorded) is frame 0's. Run inside the SIL container.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "hpc/inner"))

from ros2_loop_bridge import RosLoopBridge  # noqa: E402


def quat_to_matrix(wxyz):
    w, x, y, z = (float(v) for v in wxyz)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def load_png_rgb(path):
    import cv2

    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(path)
    return np.ascontiguousarray(image[:, :, ::-1])


def grid(frame, side):
    ranges = np.asarray(
        [[np.nan if v is None else float(v) for v in row] for row in frame[f"tof_{side}_m"]], dtype=np.float32
    )
    return {"ranges": ranges, "valid": np.asarray(frame[f"tof_{side}_valid"], dtype=bool)}


def config_from(report):
    target = report["blender_scene"]["target"]
    approach = report.get("approach_strategy") or {}
    tracker = report.get("tracker_config") or {}
    closing = report["blender_scene"].get("visual_proxy_closing_axis_tool") or report["camera_mount_selection"][
        "closing_axis_tool"
    ]
    return {
        "target_id": target["id"],
        "target_axis_w": target["axis_w"],
        "target_radius_m": target["radius_m"],
        "home_pose_wxyz": report["initial_tool_pose_wxyz"][0],
        "closing_axis_tool": closing,
        "mouth_offset_tool_m": [0.0, 0.0, 0.07],
        "photometric_normalization": report.get("photometric_normalization", "raw"),
        "approach": {
            "mode": approach.get("mode", "straight"),
            "standoff_m": approach.get("standoff_m", 0.0),
            "max_step_m": approach.get("max_step_m", 0.004),
            "max_rotation_deg": approach.get("max_rotation_deg", 1.5),
            "planned_tool_quat_wxyz": approach.get("planned_tool_quat_wxyz"),
        },
        "motion_model": tracker.get("motion_model", "translation"),
        "jaw_self_mask": bool((report.get("jaw_self_mask") or {}).get("enabled")),
        "closure_hold": bool((report.get("closure_hold") or {}).get("enabled")),
        "depth_appearance": bool((report.get("depth_appearance") or {}).get("enabled")),
        "depth_appearance_arm": (report.get("depth_appearance") or {}).get("arm", "strict"),
        "fps": 10,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--capture", required=True, type=Path)
    parser.add_argument("--gate", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=None)
    args = parser.parse_args(argv)
    report = json.loads((args.capture / "report.json").read_text())
    frames = json.loads((args.capture / "frames.json").read_text())["frames"]
    count = len(frames) if args.frames is None else min(len(frames), args.frames)
    work = args.output.parent / (args.output.stem + "_bringup")
    work.mkdir(parents=True, exist_ok=True)

    bringup_log = open(work / "bringup.log", "w")  # noqa: SIM115 - the child writes to it until the end
    bringup = subprocess.Popen(
        [sys.executable, "-m", "pruning_sil.live_bringup", "--gate", args.gate, "--output-dir", str(work),
         "--config-timeout-s", "120", "--input-timeout-s", "10"],
        stdout=bringup_log, stderr=subprocess.STDOUT,
    )  # fmt: skip
    bridge = RosLoopBridge(wait_s=60.0)
    config = config_from(report)
    bridge.publish_config(config)
    connect = bridge.wait_for_graph(120.0)
    del connect

    camera = report["camera"]
    camera_matrix = np.asarray(camera["wrist_intrinsics"], dtype=float)
    mount = np.asarray(camera["wrist_position_in_tool_m"], dtype=float)
    mount_rotation = np.asarray(camera["wrist_rotation_in_tool_ros"], dtype=float)
    home = np.asarray(report["initial_tool_pose_wxyz"][0], dtype=float)
    rotation = quat_to_matrix(home[3:7])
    t0 = np.eye(4)
    t0[:3, :3], t0[:3, 3] = rotation @ mount_rotation, home[:3] + rotation @ mount

    rows = []

    def cycle(stamp_ns, marker, rgb, depth, transform, tool, frame):
        started = time.monotonic()
        bridge.publish_cycle(
            stamp_ns,
            marker,
            rgb=rgb,
            depth=depth,
            camera_matrix=camera_matrix,
            world_from_optical=transform,
            tool_pose_wxyz=tool,
            tof={"left": grid(frame, "left"), "right": grid(frame, "right")},
            contact_force_w=[float(frame.get("contact_force_n") or 0.0), 0.0, 0.0],
            joints={"position": frame["joint_position_rad"], "velocity": frame["joint_velocity_rad_s"]},
        )
        result = bridge.wait(stamp_ns)
        result["wall_s"] = time.monotonic() - started
        return result

    # t = 0: the preview, seeded from the recorded pixel.
    initial = cycle(
        0,
        {"time_s": 0.0, "index": -1, "world_from_optical": t0.tolist(), "initialize": True,
         "seed_pixel": report["vision_initialization"]["pixel_xy"], "visible": True},
        load_png_rgb(args.capture / "preview_wrist.png"),
        np.load(args.capture / "preview_depth.npy").astype(np.float32),
        t0, home, frames[0],
    )  # fmt: skip
    rows.append({"index": -1, "result": initial})
    for index in range(count):
        frame = frames[index]
        transform = np.eye(4)
        transform[:3, :3] = np.asarray(frame["wrist_rotation_w_ros"], dtype=float)
        transform[:3, 3] = np.asarray(frame["wrist_position_w_m"], dtype=float)
        time_s = float(frame["live_vision"]["measurement_time_s"])
        result = cycle(
            int(round(time_s * 1e9)),
            {"time_s": time_s, "index": index, "world_from_optical": transform.tolist()},
            load_png_rgb(args.capture / "frames" / f"wrist_{index:05d}.png"),
            np.load(args.capture / "frames" / f"depth_{index:05d}.npy").astype(np.float32),
            transform,
            frame["tool_pose_wxyz"],
            frame,
        )
        rows.append({"index": index, "result": result})
    bridge.end()
    try:
        bringup.wait(timeout=60)
    except subprocess.TimeoutExpired:
        bringup.kill()

    # The proposal after observing frame j is the command the recorder applied in frame j + 1.
    agree_state = agree_phase = compared = cut_agree = 0
    disagreements = []
    for row in rows:
        evidence = row["result"]["evidence"]
        decision = evidence.get("decision") or {}
        nxt = row["index"] + 1
        if nxt < count:
            recorded = frames[nxt].get("visual_servo_decision") or {}
            compared += 1
            same_state = decision.get("state") == recorded.get("state")
            same_phase = evidence.get("phase") == frames[nxt].get("phase")
            agree_state += same_state
            agree_phase += same_phase
            if not (same_state and same_phase) and len(disagreements) < 20:
                disagreements.append(
                    {"after_frame": row["index"], "state": [decision.get("state"), recorded.get("state")],
                     "phase": [evidence.get("phase"), frames[nxt].get("phase")]}
                )  # fmt: skip
        if row["index"] >= 0:
            recorded_cut = (frames[row["index"]].get("live_vision") or {}).get("cut") or {}
            live_cut = (evidence.get("live_vision") or {}).get("cut") or {}
            cut_agree += live_cut.get("phase") == recorded_cut.get("phase")
    walls = np.asarray([row["result"]["wall_s"] for row in rows])
    trips = np.asarray([row["result"]["round_trip_s"] for row in rows if row["result"]["round_trip_s"] is not None])
    held = [row["index"] for row in rows if not row["result"]["decision"]["passed"]]
    summary = {
        "schema_version": 1,
        "tool": "tools/rehearse_ros2_loop.py",
        "capture": str(args.capture),
        "scope": "open-loop rehearsal of the live graph on recorded bundles; not closed-loop evidence",
        "cycles_published": len(rows),
        "cycles_answered": len(rows),
        "decision_state_agreement": [agree_state, compared],
        "motion_phase_agreement": [agree_phase, compared],
        "cut_phase_agreement": [cut_agree, count],
        "gate_held_cycles": held,
        "cycle_wall_s": {
            "p50": float(np.median(walls)),
            "p95": float(np.percentile(walls, 95)),
            "max": float(walls.max()),
        },
        "sensor_to_gated_command_s": {
            "p50": float(np.median(trips)),
            "p95": float(np.percentile(trips, 95)),
            "max": float(trips.max()),
        },
        "disagreements": disagreements,
        "bringup": json.loads((work / "ros2_loop_bringup.json").read_text())
        if (work / "ros2_loop_bringup.json").exists()
        else None,
    }
    args.output.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k not in ("disagreements", "bringup")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
