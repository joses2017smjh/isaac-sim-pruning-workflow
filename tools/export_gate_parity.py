"""Rebuild the safety-gate inputs of every recorded control cycle and run the Python checks.

Each Isaac vision capture records, per frame, the fresh time-of-flight grids the
recorder's ``sensor_guard`` saw, the tool pose, the contact force and the tracker
measurement that ``VisionPruningDemo.observe`` turned into a ``CutObservation``.
This tool rebuilds those inputs without images, runs ``SafetyGateReference``
(``sensor_guard`` + ``evaluate_cut_gate`` + the recorder's stop rules), and checks
the reconstruction against what the recording itself logged:

* the ToF missing-frame counter and the ToF stop reason;
* the cut certificate (pass flags, reasons, mouth distance, alignment, age);
* the run's first stop, when that stop came from one of the two checks.

It writes three things: a JSONL of every cycle (inputs + Python outputs) that the
ROS 2 graph harness replays into the C++ node; a summary; and, optionally, a
small text fixture of contiguous segments for the C++ gtest.

CPU only. No ROS, no Isaac, no images.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source/isaaclab_pruning"))

from isaaclab_pruning.sim.vision_demo_controller import demo_cut_config  # noqa: E402
from isaaclab_pruning.task.safety_gate import (  # noqa: E402
    GATE_LOST_DURING_CLOSURE,
    TOF_REASONS,
    GateCycle,
    SafetyGateReference,
)
from isaaclab_pruning.task.simulated_cut import (  # noqa: E402
    CutObservation,
    tool_mouth_geometry,
)

#: The recorder's hazard threshold: ``hazard_contact=contact_force_n > 5.0``.
HAZARD_CONTACT_FORCE_N = 5.0
#: Reasons a cut certificate can carry; a recorded stop with one of these came from the cut check.
CUT_GATE_REASONS = (
    "invalid_vision_source",
    "vision_invalid",
    "wrong_target_id",
    "target_not_branch",
    "hazard_contact",
    "vision_stale_or_future",
    "invalid_branch_radius",
    "branch_exceeds_demo_cut_limit",
    "invalid_geometry",
)
GATE_DERIVED_STOPS = set(TOF_REASONS) | set(CUT_GATE_REASONS) | {GATE_LOST_DURING_CLOSURE}
CONTINUOUS_TOLERANCE = 1e-9


def _nan_list(values, size):
    out = [math.nan if value is None else float(value) for value in (values or [])]
    return out if len(out) == size else None


def _grid(frame, side):
    grid, valid = frame.get(f"tof_{side}_m"), frame.get(f"tof_{side}_valid")
    if grid is None or valid is None:
        return None, None
    flat = [cell for row in grid for cell in row]
    mask = [bool(cell) for row in valid for cell in row]
    ranges = _nan_list(flat, 64)
    if ranges is None or len(mask) != 64:
        return None, None
    return ranges, mask


def stamp_ns(seconds):
    return int(round(float(seconds) * 1e9))


def _float_or_none(value):
    return None if value is None or not math.isfinite(float(value)) else float(value)


def _close(left, right):
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= CONTINUOUS_TOLERANCE * max(1.0, abs(float(right)))


def certificate_dict(certificate):
    return {
        "safe_to_approach": bool(certificate.safe_to_approach),
        "ready_to_close": bool(certificate.ready_to_close),
        "reasons": list(certificate.reasons),
        "mouth_distance_m": certificate.mouth_distance_m,
        "perpendicularity_error_deg": certificate.perpendicularity_error_deg,
        "vision_age_s": certificate.vision_age_s,
        "vision_source": certificate.vision_source,
        "waived_checks": list(certificate.waived_checks),
    }


def certificate_matches(python, recorded):
    """Field-by-field agreement with the recorded certificate; returns the fields that differ."""
    differ = []
    for key in ("safe_to_approach", "ready_to_close"):
        if bool(python[key]) != bool(recorded.get(key)):
            differ.append(key)
    if list(python["reasons"]) != list(recorded.get("reasons") or []):
        differ.append("reasons")
    if python["vision_source"] != recorded.get("vision_source", "live"):
        differ.append("vision_source")
    if list(python["waived_checks"]) != list(recorded.get("waived_checks") or []):
        differ.append("waived_checks")
    for key in ("mouth_distance_m", "perpendicularity_error_deg", "vision_age_s"):
        if not _close(python[key], recorded.get(key)):
            differ.append(key)
    return differ


def discover(roots, max_depth=3):
    """Captures up to ``max_depth`` levels below each root (artifacts/<family>/<batch>/<run>).

    A depth-limited glob, not a recursive walk: batch directories link ``code/artifacts``
    back to the artifacts tree, and a recursive walk that follows links never ends.
    """
    captures = set()
    for root in roots:
        for depth in range(1, max_depth + 1):
            for frames in Path(root).glob("/".join(["*"] * depth) + "/frames.json"):
                if "code" not in frames.parts and (frames.parent / "report.json").is_file():
                    captures.add(frames.parent)
    return sorted(captures)


class Skip(Exception):
    pass


def capture_cycles(directory):
    """Rebuild one capture: (records, cut config, seeded ToF counter); raise Skip if not a vision capture."""
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    document = json.loads((directory / "frames.json").read_text(encoding="utf-8"))
    frames = document["frames"] if isinstance(document, dict) else document
    if not frames:
        raise Skip("no_frames")
    target = (report.get("blender_scene") or {}).get("target") or {}
    if not target.get("id") or target.get("axis_w") is None or target.get("radius_m") is None:
        raise Skip("no_selected_target")
    if any(not isinstance(frame.get("live_vision"), dict) for frame in frames):
        raise Skip("not_a_live_vision_capture")
    config = demo_cut_config(target["id"])
    initial = (report.get("initial_live_vision") or {}).get("cut") or {}
    prior_phase = initial.get("phase") or "approach"
    prior_detached = bool(initial.get("detached", False))

    # Seed the ToF counter with the pre-loop guard's state, which is not recorded directly:
    # frame 0's counter is that state + 1 when frame 0 has no usable zone, else 0.
    first_ranges = []
    for side in ("left", "right"):
        ranges, mask = _grid(frames[0], side)
        if ranges is None:
            raise Skip("no_tof_grids")
        first_ranges += [r for r, v in zip(ranges, mask) if v and math.isfinite(r) and r > 0.0]
    recorded_first = frames[0].get("sensor_guard_consecutive_missing_frames")
    seeded_missing = 0 if first_ranges or recorded_first is None else max(0, int(recorded_first) - 1)

    reference = SafetyGateReference(config)
    reference.missing = seeded_missing
    closure_reference = None  # (target position, vision timestamp) of the closure-start observation
    records = []
    for frame in frames:
        live = frame["live_vision"]
        cut = live.get("cut") or {}
        measurement = live.get("measurement") or {}
        time_s = live.get("measurement_time_s")
        tool = frame.get("tool_pose_wxyz")
        if time_s is None or tool is None or len(tool) != 7:
            raise Skip("frame_without_observation_inputs")
        left, left_valid = _grid(frame, "left")
        right, right_valid = _grid(frame, "right")
        if left is None or right is None:
            raise Skip("no_tof_grids")
        ranges = np.asarray(left + right, dtype=float)
        valid = np.asarray(left_valid + right_valid, dtype=bool)

        position = measurement.get("target_position_world_m")
        target_position = tuple(float(v) for v in position) if position is not None else (math.nan,) * 3
        mouth_offset = tuple(float(v) for v in live.get("mouth_offset_tool_m", (0.0, 0.0, 0.07)))
        closing_axis_tool = tuple(float(v) for v in live.get("closing_axis_tool", (1.0, 0.0, 0.0)))
        mouth, closing_axis = tool_mouth_geometry(tuple(float(v) for v in tool), mouth_offset, closing_axis_tool)
        contact_force = float(frame.get("contact_force_n") or 0.0)
        hold = live.get("closure_hold") or {}
        observation = CutObservation(
            time_s=float(time_s),
            vision_timestamp_s=float(time_s),
            target_id=str(target["id"]),
            target_semantic="branch",
            vision_valid=measurement.get("state") == "tracking",
            target_position_w=target_position,
            target_axis_w=tuple(float(v) for v in target["axis_w"]),
            target_radius_m=float(target["radius_m"]),
            mouth_position_w=mouth,
            cutter_closing_axis_w=closing_axis,
            hazard_contact=contact_force > HAZARD_CONTACT_FORCE_N,
        )
        held_stamp = None
        if hold.get("held"):
            if closure_reference is None:
                raise Skip("held_frame_without_closure_reference")
            observation = replace(
                observation,
                target_position_w=closure_reference[0],
                vision_valid=True,
                vision_source="closure_hold",
                held_target_timestamp_s=closure_reference[1],
            )
            held_stamp = closure_reference[1]
        cycle = GateCycle(
            tof_ranges=ranges,
            tof_valid=valid,
            observation=observation,
            prior_cut_phase=prior_phase,
            prior_detached=prior_detached,
        )
        verdict = reference.step(cycle)
        if cut.get("phase") == "closing" and closure_reference is None:
            # The cutter froze this observation as its closure reference.
            closure_reference = (observation.target_position_w, observation.vision_timestamp_s)
        record = {
            "index": int(frame["index"]),
            "time_s": float(time_s),
            "stamp_ns": stamp_ns(time_s),
            "tof_left": [None if not (v and math.isfinite(r)) else r for r, v in zip(left, left_valid)],
            "tof_right": [None if not (v and math.isfinite(r)) else r for r, v in zip(right, right_valid)],
            "tool_pose_wxyz": [float(v) for v in tool],
            "contact_force_n": contact_force,
            "target": {
                "id": observation.target_id,
                "semantic": observation.target_semantic,
                "valid": bool(observation.vision_valid),
                "source": observation.vision_source,
                "vision_time_s": observation.vision_timestamp_s,
                "held_time_s": held_stamp,
                "position": [_float_or_none(v) for v in observation.target_position_w],
                "axis": list(observation.target_axis_w),
                "radius_m": observation.target_radius_m,
                "tracker_state": measurement.get("state"),
            },
            "mouth_offset_tool_m": list(mouth_offset),
            "closing_axis_tool": list(closing_axis_tool),
            "prior_cut_phase": prior_phase,
            "prior_detached": prior_detached,
            "proposal": {
                "cut_phase": cut.get("phase"),
                "cut_detached": bool(cut.get("detached", False)),
                "closure_progress": float(cut.get("closure_progress") or 0.0),
                "state": (frame.get("visual_servo_decision") or {}).get("state"),
            },
            "python": {
                "passed": verdict.passed,
                "reason": verdict.reason,
                "latched_reason": verdict.latched_reason,
                "tof_reason": verdict.tof_reason,
                "tof_missing_frames": verdict.tof_missing_frames,
                "tof_min_range_m": verdict.tof_min_range_m,
                "closure_authorized": verdict.closure_authorized,
                "certificate": certificate_dict(verdict.certificate),
            },
            "recorded": {
                "tof_missing_frames": frame.get("sensor_guard_consecutive_missing_frames"),
                "sensor_stop_reason": frame.get("sensor_stop_reason"),
                "cut_phase": cut.get("phase"),
                "cut_stopped_reason": cut.get("stopped_reason"),
                "certificate": cut.get("certificate"),
            },
        }
        records.append(record)
        prior_phase = cut.get("phase") or prior_phase
        prior_detached = bool(cut.get("detached", prior_detached))
    return records, config, seeded_missing


def check_capture(records):
    """Compare the reconstruction with the recording, frame by frame and at the first stop."""
    counts = Counter()
    first_recorded_stop = None
    first_reference_latch = None
    for record in records:
        python, recorded = record["python"], record["recorded"]
        counts["cycles"] += 1
        if recorded["tof_missing_frames"] is not None:
            counts["tof_counter_compared"] += 1
            counts["tof_counter_match"] += int(python["tof_missing_frames"] == recorded["tof_missing_frames"])
        if recorded["certificate"] is not None:
            counts["certificate_compared"] += 1
            differ = certificate_matches(python["certificate"], recorded["certificate"])
            counts["certificate_match"] += int(not differ)
            for key in differ:
                counts[f"certificate_differs_{key}"] += 1
        if first_recorded_stop is None and recorded["cut_phase"] == "stopped":
            first_recorded_stop = (record["index"], recorded["cut_stopped_reason"])
        if first_reference_latch is None and python["latched_reason"] is not None:
            first_reference_latch = (record["index"], python["latched_reason"])
    if first_recorded_stop is not None and first_recorded_stop[1] in GATE_DERIVED_STOPS:
        stop_class = "gate_derived"
        stop_agrees = first_reference_latch == first_recorded_stop
    elif first_recorded_stop is not None:
        stop_class = "controller_or_runner"
        stop_agrees = first_reference_latch is None or first_reference_latch[0] >= first_recorded_stop[0]
    else:
        stop_class = "no_stop"
        stop_agrees = first_reference_latch is None
    return {
        **counts,
        "first_recorded_stop": first_recorded_stop,
        "first_reference_latch": first_reference_latch,
        "recorded_stop_class": stop_class,
        "first_stop_agrees": bool(stop_agrees),
    }


def _token(value):
    if value is None:
        return "-"
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, float):
        return repr(value) if math.isfinite(value) else "nan"
    text = str(value)
    if not text or any(ch.isspace() for ch in text):
        raise ValueError(f"Fixture token must be a nonempty word: {text!r}")
    return text


def fixture_segments(by_capture, limit_cycles):
    """Contiguous windows around every distinct gate outcome, so stateful counters stay honest."""
    chosen, seen, used = [], set(), 0
    for label, (records, config) in by_capture.items():
        for position, record in enumerate(records):
            python = record["python"]
            signature = (
                python["passed"],
                python["reason"],
                python["tof_reason"],
                tuple(python["certificate"]["reasons"]),
                python["certificate"]["vision_source"],
                record["prior_cut_phase"],
            )
            if signature in seen:
                continue
            start, stop = max(0, position - 4), min(len(records), position + 2)
            if used + (stop - start) > limit_cycles:
                continue
            seen.add(signature)
            used += stop - start
            chosen.append((label, config, records, start, stop))
    return chosen


def _cycle_tokens(record):
    target = record["target"]
    held = math.nan if target["held_time_s"] is None else target["held_time_s"]
    position = [math.nan if v is None else v for v in target["position"]]
    return [
        "cycle",
        str(record["index"]),
        *map(_token, (record["time_s"], target["vision_time_s"], float(held))),
        *map(_token, (target["id"], target["semantic"], target["valid"], target["source"])),
        *map(_token, [float(v) for v in position + target["axis"] + [target["radius_m"]]]),
        *map(_token, [float(v) for v in record["tool_pose_wxyz"]]),
        *map(_token, [float(v) for v in record["mouth_offset_tool_m"] + record["closing_axis_tool"]]),
        _token(float(record["contact_force_n"])),
        _token(record["prior_cut_phase"]),
        _token(record["prior_detached"]),
    ]


def _expect_tokens(python):
    certificate = python["certificate"]

    def number(value):
        return _token(math.nan if value is None else float(value))

    return [
        "expect",
        *map(_token, (python["passed"], python["reason"], python["latched_reason"], python["tof_reason"])),
        str(int(python["tof_missing_frames"])),
        number(python["tof_min_range_m"]),
        *map(_token, (certificate["safe_to_approach"], certificate["ready_to_close"], python["closure_authorized"])),
        number(certificate["mouth_distance_m"]),
        number(certificate["perpendicularity_error_deg"]),
        number(certificate["vision_age_s"]),
        str(len(certificate["reasons"])),
        *map(_token, certificate["reasons"]),
    ]


def write_fixture(path, segments, seeded_missing_by_label):
    """Line format, read by ros2/pruning_sil_cpp/test/test_safety_gates.cpp:

    segment <label> <first index> <cycles> <target id> <missing before> <latched before|->
    config <max radius> <mouth tolerance> <alignment deg> <vision max age> <min clearance>
    cycle <index> <time> <vision time> <held time|nan> <id> <semantic> <valid> <source>
          <target xyz> <axis xyz> <radius> <tool pose xyz wxyz> <mouth offset xyz> <closing axis xyz>
          <contact N> <prior cut phase> <prior detached>
    tof <128 ranges, nan = no usable zone>
    expect <passed> <reason|-> <latched|-> <tof reason|-> <missing> <min range|nan> <safe> <ready>
           <closure authorized> <mouth distance|nan> <alignment deg|nan> <age|nan> <n reasons> <reasons...>
    """
    lines = [
        "pruning_safety_gate_fixture 1",
        "# Generated by tools/export_gate_parity.py from recorded Isaac captures.",
        "# Expected values are the Python checks (isaaclab_pruning.task.safety_gate), not the C++ port.",
    ]
    for label, config, records, start, stop in segments:
        before = records[start - 1]["python"] if start else None
        lines.append(
            " ".join(
                [
                    "segment",
                    _token(label),
                    str(records[start]["index"]),
                    str(stop - start),
                    _token(config.selected_target_id),
                    str(before["tof_missing_frames"] if before else seeded_missing_by_label[label]),
                    _token(before["latched_reason"] if before else None),
                ]
            )
        )
        limits = (
            config.max_branch_radius_m,
            config.mouth_position_tolerance_m,
            config.alignment_tolerance_deg,
            config.vision_max_age_s,
            0.06,
        )
        lines.append(" ".join(["config", *(_token(float(v)) for v in limits)]))
        for record in records[start:stop]:
            lines.append(" ".join(_cycle_tokens(record)))
            tof = [math.nan if v is None else v for v in record["tof_left"] + record["tof_right"]]
            lines.append(" ".join(["tof", *(_token(float(v)) for v in tof)]))
            lines.append(" ".join(_expect_tokens(record["python"])))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--roots", nargs="+", default=[str(REPO / "artifacts")])
    parser.add_argument("--cycles-out", required=True, help="gzip JSONL of every rebuilt cycle (keep outside git)")
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--fixture-out", help="Small text fixture for the C++ gtest")
    parser.add_argument("--fixture-cycles", type=int, default=120)
    args = parser.parse_args(argv)

    captures = discover(args.roots)
    summary = {"schema_version": 1, "tool": "tools/export_gate_parity.py", "captures": {}, "skipped": {}}
    totals = Counter()
    by_capture = {}
    initial_missing = {}
    digest = hashlib.sha256()
    cycles_path = Path(args.cycles_out)
    cycles_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(cycles_path, "wt", encoding="utf-8") as stream:
        for directory in captures:
            label = str(directory.relative_to(REPO)) if directory.is_relative_to(REPO) else str(directory)
            try:
                records, config, seeded = capture_cycles(directory)
            except Skip as reason:
                summary["skipped"][label] = str(reason)
                totals[f"skipped_{reason}"] += 1
                continue
            except (KeyError, TypeError, ValueError) as error:
                summary["skipped"][label] = f"unreadable: {type(error).__name__}: {error}"
                totals["skipped_unreadable"] += 1
                continue
            check = check_capture(records)
            check["seeded_tof_missing_frames"] = seeded
            summary["captures"][label] = check
            initial_missing[label] = seeded
            for key in (
                "cycles",
                "tof_counter_compared",
                "tof_counter_match",
                "certificate_compared",
                "certificate_match",
            ):
                totals[key] += check.get(key, 0)
            totals[f"stop_class_{check['recorded_stop_class']}"] += 1
            totals[f"first_stop_agrees_{check['first_stop_agrees']}"] += 1
            for record in records:
                record["capture"] = label
                line = json.dumps(record, allow_nan=False, sort_keys=True)
                digest.update(line.encode())
                stream.write(line + "\n")
                python = record["python"]
                totals["python_hold_cycles"] += int(not python["passed"])
                if python["reason"] is not None:
                    totals[f"python_reason_{python['reason']}"] += 1
            by_capture[label] = (records, config)
    summary["totals"] = dict(sorted(totals.items()))
    summary["cycles_sha256"] = digest.hexdigest()
    summary["cycles_path"] = str(cycles_path)
    summary["notes"] = [
        "Python = isaaclab_pruning.task.safety_gate.SafetyGateReference on inputs rebuilt from frames.json.",
        "Recorded = what the Isaac recorder logged at run time, possibly under an older frozen code copy.",
        "The pre-loop ToF guard state is not recorded; frame 0's counter seeds it (see capture_cycles).",
    ]
    Path(args.summary_out).write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if args.fixture_out:
        segments = fixture_segments(by_capture, args.fixture_cycles)
        write_fixture(Path(args.fixture_out), segments, initial_missing)
    print(json.dumps(summary["totals"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
