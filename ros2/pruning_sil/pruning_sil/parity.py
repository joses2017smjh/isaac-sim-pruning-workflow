"""Check that the ROS 2 node reproduces a recorded capture, and say where it does not.

Parity is reported at two points, because they fail for different reasons:

``measurement``  the tracker's estimated branch position. The tracker is
                 iterative, so its feature set evolves from frame 0 and any
                 floating-point difference between this host and the recording
                 host accumulates here.
``command``      the bounded Cartesian delta the controller proposes. Given the
                 same measurement this is closed-form, so a command difference
                 that is larger than its measurement difference would point at
                 the control path rather than at accumulated drift.

Decision *state* is reported separately and is not given a tolerance. A replay
that proposed ``tracking`` where the recording held, or the reverse, has not
reproduced the run at all, however small its numbers are.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from .capture import Capture
from .replay import ControllerReplay, compare_decision, invalid_frame_is_held

SCHEMA_VERSION = 1

#: Worst-case command agreement this replay is asserted to hold to. It is a
#: reporting threshold, not a control tolerance, and the measured distribution
#: is always recorded beside it.
DEFAULT_TOLERANCE_M = 2.0e-3

SCOPE = (
    "Software-in-the-loop parity between the ROS 2 node and a recorded Isaac capture. "
    "Every input is recorded simulator output. This is not hardware-in-the-loop, no "
    "physical sensor or robot is involved, and no command is actuated."
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _position_error(proposed, recorded):
    if proposed is None or recorded is None:
        return None
    return float(np.linalg.norm(np.asarray(proposed, dtype=float) - np.asarray(recorded, dtype=float)))


def _distribution(values):
    finite = np.asarray([value for value in values if value is not None], dtype=float)
    if finite.size == 0:
        return {"count": 0}
    return {
        "count": int(finite.size),
        "exact_zero": int((finite == 0.0).sum()),
        "median_m": float(np.median(finite)),
        "p95_m": float(np.percentile(finite, 95)),
        "p99_m": float(np.percentile(finite, 99)),
        "max_m": float(finite.max()),
        "over_0p1mm": int((finite > 1e-4).sum()),
        "over_1mm": int((finite > 1e-3).sum()),
    }


def run_parity(capture_dir, tolerance_m=DEFAULT_TOLERANCE_M):
    capture = Capture(capture_dir)
    replay = ControllerReplay(capture)
    replay.initialize()

    frames, command_errors, measurement_errors = [], [], []
    states_agree = 0
    compared = 0
    skipped_no_source = 0

    for index in range(len(capture)):
        proposal = replay.step(index)
        if proposal is None:
            skipped_no_source += 1
            continue
        recorded = capture.recorded_decision(index)
        comparison = compare_decision(proposal["decision"], recorded, position_tolerance_m=tolerance_m)

        measurement_error = _position_error(
            (replay.controller.measurement or {}).get("target_position_world_m"),
            capture.recorded_measurement(capture.source_frame_index(index)).get("target_position_world_m"),
        )
        compared += 1
        states_agree += int(bool(comparison["state_matches"]))
        command_errors.append(comparison["delta_error_m"])
        measurement_errors.append(measurement_error)
        frames.append(
            {
                "frame_index": index,
                "controller_source_frame_index": capture.source_frame_index(index),
                "proposed_state": comparison["proposed_state"],
                "recorded_state": comparison["recorded_state"],
                "state_matches": comparison["state_matches"],
                "command_delta_error_m": comparison["delta_error_m"],
                "measurement_error_m": measurement_error,
                "within_tolerance": comparison["delta_matches"],
            }
        )

    command = _distribution(command_errors)
    measurement = _distribution(measurement_errors)
    within = all(frame["within_tolerance"] for frame in frames)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "evaluation_type": "software_in_the_loop_replay_parity",
        "capture_dir": str(Path(capture_dir).resolve()),
        "source_job_id": capture.job_id,
        "recorded_frames": len(capture),
        "frames_compared": compared,
        "frames_skipped_no_source_image": skipped_no_source,
        "decision_state_agreement": f"{states_agree}/{compared}",
        "decision_states_all_agree": states_agree == compared,
        "command_tolerance_m": tolerance_m,
        "command_within_tolerance": within,
        "command_delta_error": command,
        "tracker_measurement_error": measurement,
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "opencv": __import__("cv2").__version__,
            "platform": platform.platform(),
        },
        "per_frame": frames,
    }


def run_negative_controls(capture_dir):
    """Corrupted observations must hold. They must never approach or release.

    The corruption schedules are the ones the CPU replay benchmark already uses:
    an RGB blackout and a depth dropout. The assertion is not that the tracker
    survives them, it is that failing to see the branch never authorizes motion.
    """
    capture = Capture(capture_dir)
    results = []
    for name in ("rgb_blackout", "depth_dropout"):
        replay = ControllerReplay(capture)
        replay.initialize()
        controller = replay.controller
        violations, held, evaluated = [], 0, 0

        for index in range(1, min(len(capture), 40)):
            source = capture.source_frame_index(index)
            if source < 0:
                continue
            frame = capture.frames[index]
            rgb = capture.rgb(source)
            depth = capture.depth(source)
            if name == "rgb_blackout":
                rgb = np.zeros_like(rgb)
            else:
                depth = np.full_like(depth, np.nan)
            controller.observe(
                rgb,
                depth,
                capture.camera_matrix,
                capture.world_from_optical(source),
                float(capture.frames[source]["time_s"]),
                np.asarray(frame["tool_pose_wxyz"], dtype=float),
            )
            _, phase, decision = controller.command(
                np.asarray(frame["tool_pose_wxyz"], dtype=float), float(frame["time_s"])
            )
            evaluated += 1
            if invalid_frame_is_held(decision):
                held += 1
            else:
                violations.append({"frame_index": index, "state": decision.get("state"), "phase": phase})

        results.append(
            {
                "control": name,
                "frames_evaluated": evaluated,
                "frames_held": held,
                "authorized_approach_or_release": violations,
                "passed": not violations,
            }
        )
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    parser.add_argument("--tolerance-m", type=float, default=DEFAULT_TOLERANCE_M)
    parser.add_argument("--skip-negative-controls", action="store_true")
    args = parser.parse_args(argv)

    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")

    document = run_parity(args.capture_dir, args.tolerance_m)
    document["negative_controls"] = [] if args.skip_negative_controls else run_negative_controls(args.capture_dir)
    document["negative_controls_passed"] = all(item["passed"] for item in document["negative_controls"])

    summary = {key: value for key, value in document.items() if key != "per_frame"}
    serialized = json.dumps(document, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps(summary, indent=2))

    ok = document["decision_states_all_agree"] and document["command_within_tolerance"]
    ok = ok and (args.skip_negative_controls or document["negative_controls_passed"])
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
