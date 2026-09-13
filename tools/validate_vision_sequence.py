#!/usr/bin/env python3
"""Grade captured vision → detachment → home-return evidence independently.

This reads report.json and frames.json, not images or simulator state. A passing
grade is capture-rate evidence of a simulated sequence, not continuous contact
safety, material fracture, learned recognition, or an actuated physical blade.
Incomplete captures produce a failed grade and exit 1, not an exception.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

THRESHOLDS = {
    "minimum_post_detach_drop_m": 0.020,
    "maximum_pre_detach_piece_motion_m": 0.002,
    "minimum_home_directed_retreat_m": 0.020,
    "maximum_final_home_error_m": 0.003,
    "maximum_captured_contact_force_n": 5.0,
}


def _number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def _vector(value, size):
    if not isinstance(value, list) or len(value) != size or any(_number(item) is None for item in value):
        return None
    return np.asarray(value, dtype=float)


def _mapping(value):
    return value if isinstance(value, dict) else {}


def _strict_input(report, frames):
    if not isinstance(report, dict):
        raise ValueError("report must be a JSON object")
    if isinstance(frames, dict):
        if "frames" not in frames:
            raise ValueError("frames document must contain a frames list")
        frames = frames["frames"]
    if not isinstance(frames, list) or any(not isinstance(frame, dict) for frame in frames):
        raise ValueError("frames must be a list of JSON objects")
    # Python's JSON decoder otherwise accepts NaN/Infinity, which must never
    # become evidence that passes a comparison accidentally.
    json.dumps([report, frames], allow_nan=False)
    return frames


def _stops(report, frames):
    reasons = []
    for label, record in [("report", report), *((f"frame[{index}]", frame) for index, frame in enumerate(frames))]:
        live = _mapping(record.get("live_vision", record.get("final_live_vision")))
        cut = _mapping(live.get("cut"))
        for key, value in (
            ("stopped_reason", record.get("stopped_reason")),
            ("sensor_stop_reason", record.get("sensor_stop_reason")),
            ("external_stop_reason", live.get("external_stop_reason")),
            ("cut.stopped_reason", cut.get("stopped_reason")),
        ):
            if value is not None and value != "":
                reasons.append(f"{label}.{key}: {value}")
        if record.get("phase") in ("stopped", "stopped_failure") or cut.get("phase") == "stopped":
            reasons.append(f"{label}: stopped phase")
    return reasons


def _piece_metrics(report, frames, event_index):
    initial = _vector(report.get("initial_piece_pose_wxyz"), 7)
    poses = [_vector(frame.get("selected_piece_pose_wxyz"), 7) for frame in frames]
    metrics = {"maximum_pre_detach_piece_motion_m": None, "maximum_post_detach_drop_m": None}
    complete = initial is not None and bool(poses) and all(pose is not None for pose in poses)
    if not complete or event_index is None:
        return complete, metrics
    positions = np.asarray(poses)[:, :3]
    metrics["maximum_pre_detach_piece_motion_m"] = float(
        np.linalg.norm(positions[: event_index + 1] - initial[:3], axis=1).max()
    )
    if event_index + 1 < len(frames):
        metrics["maximum_post_detach_drop_m"] = float(initial[2] - positions[event_index + 1 :, 2].min())
    return complete, metrics


def _retreat_metrics(report, frames, event_index):
    home_value = report.get("initial_tool_pose_wxyz")
    if isinstance(home_value, list) and len(home_value) == 1:
        home_value = home_value[0]
    home = _vector(home_value, 7)
    positions = [_vector(frame.get("tool_position_m"), 3) for frame in frames]
    metrics = {
        "final_home_error_m": None,
        "post_detach_home_distance_reduction_m": None,
        "post_detach_home_direction_displacement_m": None,
        "post_detach_retreat_frames": 0,
    }
    complete = home is not None and bool(positions) and all(position is not None for position in positions)
    if not complete:
        return complete, metrics
    positions = np.asarray(positions)
    distances = np.linalg.norm(positions - home[:3], axis=1)
    metrics["final_home_error_m"] = float(distances[-1])
    if event_index is None or event_index + 1 >= len(frames):
        return complete, metrics
    metrics["post_detach_retreat_frames"] = sum(frame.get("phase") == "retreat" for frame in frames[event_index + 1 :])
    metrics["post_detach_home_distance_reduction_m"] = float(distances[event_index] - distances[-1])
    toward_home = home[:3] - positions[event_index]
    norm = float(np.linalg.norm(toward_home))
    metrics["post_detach_home_direction_displacement_m"] = (
        float(np.dot(positions[-1] - positions[event_index], toward_home / norm)) if norm > 0 else 0.0
    )
    return complete, metrics


def grade_sequence(report, frames):
    """Return strict-JSON checks/metrics/ok; absent evidence cannot pass a gate.

    ``frames`` accepts either the frame list or its {"frames": [...]} envelope.
    Malformed top-level input or nonfinite JSON raises ValueError/TypeError.
    Missing telemetry, including an empty/incomplete capture, returns ok=False.
    """
    frames = _strict_input(report, frames)
    cuts = [_mapping(_mapping(frame.get("live_vision")).get("cut")) for frame in frames]
    events = [index for index, cut in enumerate(cuts) if cut.get("detach_event") is True]
    requests = [index for index, frame in enumerate(frames) if frame.get("detachment_requested_after_capture") is True]
    matched = len(events) == len(requests) == 1 and events == requests
    event_index = events[0] if matched else None
    event = cuts[event_index] if event_index is not None else {}
    certificate = _mapping(event.get("certificate"))
    piece_complete, piece_metrics = _piece_metrics(report, frames, event_index)
    robot_complete, retreat_metrics = _retreat_metrics(report, frames, event_index)
    reasons = _stops(report, frames)
    contacts = [_number(frame.get("contact_force_n")) for frame in frames]
    contact_complete = bool(contacts) and all(value is not None and value >= 0 for value in contacts)
    maximum_contact = max(contacts) if contact_complete else None
    times = [_number(frame.get("time_s")) for frame in frames]
    commands = sum(
        frame.get("phase") == "vision_approach"
        and _mapping(frame.get("visual_servo_decision")).get("state") == "tracking"
        for frame in frames
    )
    supplied_commands = _number(report.get("vision_command_count"))
    indexes = [frame.get("index") for frame in frames]
    indexes_valid = (
        bool(frames) and all(type(index) is int for index in indexes) and indexes == list(range(len(frames)))
    )
    drop = piece_metrics["maximum_post_detach_drop_m"]
    pre_motion = piece_metrics["maximum_pre_detach_piece_motion_m"]
    reduction = retreat_metrics["post_detach_home_distance_reduction_m"]
    direction = retreat_metrics["post_detach_home_direction_displacement_m"]
    home_error = retreat_metrics["final_home_error_m"]
    checks = {
        "capture_complete_and_nonblank": bool(frames)
        and report.get("stage") == "complete"
        and report.get("rendering_ok") is True
        and type(report.get("frame_count")) is int
        and report["frame_count"] == len(frames)
        and _mapping(report.get("checks")).get("frames_complete") is True,
        "ordered_contiguous_frame_indexes": indexes_valid,
        "strictly_increasing_capture_times": bool(times)
        and all(value is not None and value >= 0 for value in times)
        and all(right > left for left, right in zip(times, times[1:])),
        "causal_previous_frame_commands": indexes_valid
        and all(
            type(frame.get("controller_source_frame_index")) is int
            and frame["controller_source_frame_index"] == index - 1
            for index, frame in enumerate(frames)
        ),
        "one_matching_detach_event_and_request": matched,
        "detach_has_completed_certified_closure": matched
        and _number(event.get("closure_progress")) == 1.0
        and event.get("detached") is True
        and event.get("phase") == "retreat"
        and certificate.get("ready_to_close") is True
        and certificate.get("safe_to_approach") is True
        and certificate.get("reasons") == [],
        "detached_state_consistent_with_event": matched
        and all(cut.get("detached") is (index >= event_index) for index, cut in enumerate(cuts)),
        "selected_piece_pose_evidence_complete": piece_complete,
        "piece_stationary_before_release": pre_motion is not None
        and pre_motion <= THRESHOLDS["maximum_pre_detach_piece_motion_m"],
        "piece_dropped_after_release": drop is not None and drop >= THRESHOLDS["minimum_post_detach_drop_m"],
        "robot_position_evidence_complete": robot_complete,
        "post_detach_retreat_toward_home": retreat_metrics["post_detach_retreat_frames"] > 0
        and reduction is not None
        and reduction > THRESHOLDS["minimum_home_directed_retreat_m"]
        and direction is not None
        and direction > THRESHOLDS["minimum_home_directed_retreat_m"],
        "returned_home_and_final_phase_complete": bool(frames)
        and frames[-1].get("phase") == "complete"
        and home_error is not None
        and home_error <= THRESHOLDS["maximum_final_home_error_m"],
        "vision_commands_applied_and_counted": commands > 0 and supplied_commands == commands,
        "captured_contact_within_limit": maximum_contact is not None
        and maximum_contact <= THRESHOLDS["maximum_captured_contact_force_n"],
        "no_recorded_stops": not reasons,
        "capture_did_not_advance_physics": bool(frames)
        and all(
            _number(frame.get("capture_physics_steps_advanced")) == 0
            and _number(frame.get("capture_timeline_advanced_s")) is not None
            and abs(frame["capture_timeline_advanced_s"]) < 1e-8
            for frame in frames
        ),
    }
    result = {
        "schema_version": 1,
        "ok": all(checks.values()),
        "source_job_id": report.get("job_id"),
        "source_task_outcome": report.get("task_outcome"),
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
        "thresholds": dict(THRESHOLDS),
        "metrics": {
            "recorded_frames": len(frames),
            "detach_event_frame_indexes": events,
            "detachment_request_frame_indexes": requests,
            "applied_vision_command_frames": commands,
            "reported_vision_command_count": supplied_commands,
            "maximum_captured_contact_force_n": maximum_contact,
            "recorded_stops": reasons,
            **piece_metrics,
            **retreat_metrics,
        },
        "evidence_scope": (
            "Independent grading of recorded JSON at capture rate; camera completeness/nonblankness is taken "
            "from the renderer report, not remeasured from images. PhysX piece poses and robot positions are "
            "recorded measurements. This does not certify continuous/subframe contact safety, individual ToF "
            "sensor availability, material fracture, learned recognition, or an actuated blade."
        ),
    }
    json.dumps(result, allow_nan=False)
    return result


def _read_json(path):
    def reject_constant(value):
        raise ValueError(f"Nonfinite JSON value {value} in {path}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir", type=Path, required=True, help="Capture directory containing report.json and frames.json"
    )
    parser.add_argument("--output", type=Path, help="Optional new JSON file; existing files are never overwritten")
    args = parser.parse_args(argv)
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
        result = grade_sequence(_read_json(args.input_dir / "report.json"), _read_json(args.input_dir / "frames.json"))
        serialized = json.dumps(result, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
    except (OSError, ValueError, TypeError) as error:
        parser.error(str(error))
    print(serialized, end="")
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
