"""Score the live ROS 2 loop batch against docs/EVAL_PROTOCOL_ROS2_LIVE_LOOP_2026-11-05.md (R1-R5).

Reads only what the runs recorded: each run's experiment result, report, frames
and sequence grade; the gate's decision log (``ros2_loop_decisions.jsonl``) and,
for the panel row, the panel's own record (``operator_panel_record.jsonl``).
Committed with the protocol before any registered run exists. CPU only.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_vision_experiment import run_label  # noqa: E402

TOLERANCE_M = 1e-5
FPS = 10


def _load(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def _jsonl(path):
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        return None
    return [json.loads(line) for line in lines if line.strip()]


def _rotate(quaternion_xyzw, vector):
    x, y, z, w = quaternion_xyzw
    q = np.array([x, y, z])
    v = np.asarray(vector, dtype=float)
    t = 2.0 * np.cross(q, v)
    return v + w * t + np.cross(q, t)


def _quat_mul_wxyz(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ]
    )


def applied_world(action_root_wxyz, root_xyzw):
    """The root-frame action the recorder applied, expressed in world as x y z w qx qy qz."""
    root_position, root_quat = np.asarray(root_xyzw[:3], dtype=float), np.asarray(root_xyzw[3:7], dtype=float)
    action = np.asarray(action_root_wxyz, dtype=float)
    position = root_position + _rotate(root_quat, action[:3])
    root_wxyz = np.array([root_quat[3], root_quat[0], root_quat[1], root_quat[2]])
    return np.concatenate([position, _quat_mul_wxyz(root_wxyz, action[3:7])])


def _same_pose(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if a.shape != (7,) or b.shape != (7,):
        return False
    position = float(np.linalg.norm(a[:3] - b[:3])) <= TOLERANCE_M
    orientation = min(float(np.linalg.norm(a[3:] - b[3:])), float(np.linalg.norm(a[3:] + b[3:]))) <= TOLERANCE_M
    return position and orientation


def score_r1(row, report, frames):
    """ROS 2 drove the arm: loop on, ToF guard off, a gated answer every cycle, applied == gated."""
    loop = (report or {}).get("ros2_loop") or {}
    problems = []
    if loop.get("enabled") is not True:
        problems.append("loop_not_enabled")
    if loop.get("delay_steps") != row["ros2_loop"]["delay_steps"]:
        problems.append("delay_mismatch")
    if not str(loop.get("in_process_sensor_guard", "")).startswith("disabled"):
        problems.append("in_process_tof_guard_not_disabled")
    if loop.get("timed_out"):
        problems.append("timed_out")
    if not frames:
        return {"ok": False, "problems": problems + ["no_frames"]}
    root = ((report.get("initial_control_state") or {}).get("root_pose_w_xyzw") or [[None]])[0]
    sim_stop = str(report.get("stopped_reason") or "").startswith(("tool_tracking_error", "ros2_loop_timeout"))
    gated = [(loop.get("initial_cycle") or {}).get("gated_command_wxyz")]
    gated += [((frame.get("ros2_loop") or {}).get("gated_command_next_cycle_wxyz")) for frame in frames[:-1]]
    missing = [
        i for i, frame in enumerate(frames) if not (frame.get("ros2_loop") or {}).get("gate_decision_next_cycle")
    ]
    mismatched, overridden = [], []
    for index, (frame, command) in enumerate(zip(frames, gated)):
        if command is None or root is None or len(root) != 7:
            mismatched.append(index)
            continue
        if not _same_pose(applied_world(frame["command_pose_root_wxyz"], root), command):
            (overridden if sim_stop and frame.get("phase") == "stopped_failure" else mismatched).append(index)
    if missing:
        problems.append("frames_without_gate_decision")
    if mismatched:
        problems.append("applied_command_not_gated")
    return {
        "ok": not problems,
        "problems": problems,
        "frames": len(frames),
        "frames_without_gate_decision": missing[:20],
        "applied_not_gated_frames": mismatched[:20],
        "simulator_stop_overrides": overridden,
        "round_trip_s": loop.get("round_trip_s"),
        "gate_held_cycles": loop.get("gate_held_cycles"),
    }


def _first_stop(report, frames):
    for frame in frames or []:
        if frame.get("phase") == "stopped_failure":
            return {"frame": frame["index"], "reason": (frame.get("visual_servo_decision") or {}).get("reason")}
    return {"frame": None, "reason": report.get("stopped_reason") if report else None}


def score_r4_r5(run_dir, frames):
    panel = _jsonl(run_dir / "operator_panel_record.jsonl")
    decisions = _jsonl(run_dir / "ros2_loop_decisions.jsonl")
    if panel is None or decisions is None or not frames:
        return {"ok": False, "reason": "missing panel record, decision log or frames"}, {"ok": False}
    presses = [event for event in panel if event.get("event") == "hold_pressed"]
    clicks = [event for event in panel if event.get("event") == "synthesized_click"]
    press = presses[0]["wall_ns"] if presses else None
    held = [d for d in decisions if d.get("operator_hold")]
    first = held[0] if held else None
    r4 = {
        "synthesized_click": bool(clicks),
        "hold_published": bool(presses),
        "first_held_cycle_stamp_ns": first and first["stamp_ns"],
        "first_held_reason": first and first["reason"],
    }
    if press is not None and first is not None:
        after = [d for d in decisions if d["publish_wall_ns"] > press]
        r4["passing_decisions_after_press"] = sum(1 for d in after if d["passed"])
        index = int(round(first["stamp_ns"] * FPS / 1e9))
        by_index = {frame["index"]: frame for frame in frames}
        applied = (by_index.get(index) or {}).get("ros2_loop", {}).get("applied_command_wall_ns")
        r4["press_to_applied_hold_ms"] = None if applied is None else (applied - press) / 1e6
        passing_applied_after = 0
        for frame in frames:
            loop = frame.get("ros2_loop") or {}
            stamp = int(round(frame["index"] * 1e9 / FPS))
            decision = next((d for d in decisions if d["stamp_ns"] == stamp), None)
            if loop.get("applied_command_wall_ns", 0) > press and decision is not None and decision["passed"]:
                passing_applied_after += 1
        r4["passing_commands_applied_after_press"] = passing_applied_after
        r4["control_cycles_to_hold"] = passing_applied_after
        still = [frame["index"] for frame in frames if frame["index"] >= index]
        moved = []
        for frame in frames:
            if frame["index"] >= index and frame.get("phase") != "stopped_failure":
                moved.append(frame["index"])
        r4["frames_from_first_hold"] = len(still)
        r4["frames_not_held_after_first_hold"] = moved[:20]
        r4["ok"] = bool(
            clicks
            and presses
            and first["reason"] == "operator_hold"
            and r4["passing_decisions_after_press"] == 0
            and passing_applied_after <= 1
            and not moved
        )
    else:
        r4["ok"] = False
    shown = {event["stamp_ns"]: event for event in panel if event.get("event") == "decision"}
    renders = [event for event in panel if event.get("event") == "rendered"]
    started = next((event["wall_ns"] for event in panel if event.get("event") == "start"), None)
    logged = [d for d in decisions if started is None or d["publish_wall_ns"] >= started]
    missing = [d["stamp_ns"] for d in logged if d["stamp_ns"] not in shown]
    changes = [event for event in shown.values() if event.get("pass_hold_changed")]
    ordered = sorted(shown.values(), key=lambda e: e["applied_wall_ns"])
    unrendered = []
    for position, event in enumerate(ordered):
        if not event.get("pass_hold_changed"):
            continue
        until = ordered[position + 1]["applied_wall_ns"] if position + 1 < len(ordered) else math.inf
        if not any(
            r["shown_decision_stamp_ns"] == event["stamp_ns"] and event["applied_wall_ns"] <= r["wall_ns"] <= until
            for r in renders
        ):
            unrendered.append(event["stamp_ns"])
    frames_seen = [event for event in panel if event.get("event") == "frame"]
    receive = np.asarray(
        [
            (shown[d["stamp_ns"]]["receive_wall_ns"] - d["publish_wall_ns"]) / 1e6
            for d in logged
            if d["stamp_ns"] in shown
        ]
        or [np.nan]
    )
    r5 = {
        "decisions_logged_while_panel_ran": len(logged),
        "decisions_missing_from_panel": missing[:20],
        "pass_hold_changes": len(changes),
        "changes_without_a_repaint_before_the_next_decision": unrendered,
        "wrist_frames_received": len(frames_seen),
        "wrist_frames_published": len(frames) + 1,
        "decision_receive_latency_ms": {
            "p50": float(np.nanpercentile(receive, 50)),
            "p95": float(np.nanpercentile(receive, 95)),
        },
    }
    r5["ok"] = not missing and not unrendered and len(frames_seen) >= len(frames) + 1 and bool(logged)
    return r4, r5


def score(batch):
    batch = Path(batch)
    plan = _load(batch / "plan.json")
    runs = []
    for row in plan["runs"]:
        run_dir = batch / run_label(row["index"], row)
        result = _load(run_dir / "experiment_result.json", {})
        report = _load(run_dir / "report.json", {})
        frames_doc = _load(run_dir / "frames.json", {})
        frames = frames_doc.get("frames") if isinstance(frames_doc, dict) else frames_doc
        grade = _load(run_dir / "sequence_grade.json", {})
        entry = {
            "index": row["index"],
            "run": run_dir.name,
            "delay_steps": row["ros2_loop"]["delay_steps"],
            "delay_ms": row["ros2_loop"]["delay_ms"],
            "panel": row["ros2_loop"].get("operator_hold_after_cycles") is not None,
            "pass": result.get("ok") is True,
            "failed_checks": grade.get("failed_checks"),
            "task_outcome": report.get("task_outcome"),
            "first_stop": _first_stop(report, frames),
            "r1": score_r1(row, report, frames),
        }
        if entry["panel"]:
            entry["r4"], entry["r5"] = score_r4_r5(run_dir, frames)
        runs.append(entry)
    delay_runs = [run for run in runs if not run["panel"]]
    by_delay = {}
    for run in delay_runs:
        by_delay.setdefault(run["delay_ms"], []).append(run)
    table = {
        str(delay): {"passes": sum(r["pass"] for r in group), "runs": len(group)}
        for delay, group in sorted(by_delay.items())
    }
    failing = [delay for delay, group in sorted(by_delay.items()) if sum(r["pass"] for r in group) < len(group)]
    zero = by_delay.get(0.0, [])
    panel = next((run for run in runs if run["panel"]), None)
    verdicts = {
        "R1_ros2_drove_the_arm": {
            "supported": all(run["r1"]["ok"] for run in runs),
            "runs_ok": sum(run["r1"]["ok"] for run in runs),
            "runs": len(runs),
        },
        "R2_same_result_at_0ms": {
            "supported": bool(zero) and all(run["pass"] for run in zero),
            "passes": sum(run["pass"] for run in zero),
            "runs": len(zero),
        },
        "R3_delay": {"pass_counts_by_delay_ms": table, "first_failing_delay_ms": failing[0] if failing else None},
        "R4_hold_stops_the_arm": None if panel is None else {"supported": panel["r4"]["ok"], **panel["r4"]},
        "R5_panel_keeps_up": None if panel is None else {"supported": panel["r5"]["ok"], **panel["r5"]},
    }
    return {
        "schema_version": 1,
        "protocol": plan.get("protocol"),
        "batch": batch.name,
        "code_revision": plan.get("code_revision"),
        "verdicts": verdicts,
        "runs": runs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    result = score(args.batch)
    args.output.write_text(json.dumps(result, indent=2, default=str) + "\n", encoding="utf-8")
    print(json.dumps(result["verdicts"], indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
