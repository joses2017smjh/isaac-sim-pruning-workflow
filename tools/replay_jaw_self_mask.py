#!/usr/bin/env python3
"""Drive the real VisionPruningDemo over recorded runs, with or without the jaw arms.

CPU only. For each recorded run this rebuilds the controller the runner built
(target, radius, home pose, closing axis, approach strategy, tracker options),
seeds it with the recorded seed pixel and preview, then replays every wrist
frame in order with what the live runner gave ``observe``: the saved RGB and
optical-Z depth, the recorded camera pose, the recorded tool pose, ``contact >
5 N`` as the hazard, and the external stops (initial visibility, ToF guard,
tracking error, and the runner's re-assertion of the controller's own stop)
injected where the runner injected them. ``command`` and ``command_applied``
are called as the runner called them, so the controller's state evolves the
same way.

With both flags off the replay must reproduce the recorded ``live_vision``:
state, reason and pixel_xy exactly and patch_correlation within 1e-6, and the
recorded cut phase, stop reason and detach event, through the recorded stop.
With a flag on the output is a counterfactual on recorded images: once the
replayed cut decision differs from the recorded one, later frames were rendered
for a robot that acted on the recorded decision, and once the replayed closure
progress differs from the rendered one, the jaw in the image is not the jaw the
controller predicts. Both are marked per frame, and neither is evidence of a
live outcome. Only tools/validate_vision_sequence.py grades live runs.

Inputs are read only. The output must be a new file outside the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/isaaclab_pruning"))
from isaaclab_pruning.perception.visual_servo import VisualServoConfig
from isaaclab_pruning.sim.vision_demo_controller import ApproachStrategy, VisionPruningDemo
from isaaclab_pruning.task.simulated_cut import tool_mouth_geometry

VISION_ROBUSTNESS = ROOT / "artifacts/vision_robustness"
#: Batches recorded after the jaw-in-view study (with other scenes or with the labelled arms on); --all leaves
#: them out, so --all is always the 129 earlier runs.
EXCLUDED_BATCH_PREFIXES = ("jaw-shadow-", "jaw-hold-", "depth-loop-")
CORRELATION_TOLERANCE = 1e-6
#: Commands within 1 um of the recorded one count as the recorded command. The replay rebuilds the preview
#: camera pose from the initial tool pose, which moves the frozen standoff point, and so every planned step, by
#: nanometres (0.1-5 nm on the planned-pose runs with both arms off).
COMMAND_TOLERANCE_MM = 1e-3
HAZARD_CONTACT_N = 5.0
SOURCE_FILES = (
    "source/isaaclab_pruning/isaaclab_pruning/perception/depth_appearance.py",
    "source/isaaclab_pruning/isaaclab_pruning/perception/visual_servo.py",
    "source/isaaclab_pruning/isaaclab_pruning/perception/jaw_self_mask.py",
    "source/isaaclab_pruning/isaaclab_pruning/sim/vision_demo_controller.py",
    "source/isaaclab_pruning/isaaclab_pruning/task/simulated_cut.py",
    "tools/replay_jaw_self_mask.py",
)


def discover_runs(base=VISION_ROBUSTNESS):
    """Every recorded run with frames under the vision batches, except the excluded batches."""
    runs = []
    for path in sorted(Path(base).glob("*/run_*")):
        if path.parent.name.startswith(EXCLUDED_BATCH_PREFIXES):
            continue
        if not ((path / "report.json").is_file() and (path / "frames.json").is_file()):
            continue
        if not (path / "frames").is_dir() or not any((path / "frames").glob("wrist_*.png")):
            continue  # refused at startup: no frame was ever recorded
        runs.append(path)
    return runs


def output_refusal(path):
    """Why the output path is not allowed (inside the repository or artifacts/, or existing), else None."""
    resolved = Path(path).expanduser().resolve()
    for forbidden in (ROOT.resolve(), (ROOT / "artifacts").resolve()):
        if resolved == forbidden or forbidden in resolved.parents:
            return f"refusing to write inside {forbidden}: {resolved}"
    if resolved.exists():
        return f"refusing to overwrite an existing file: {resolved}"
    return None


def _transform(position, rotation):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def preview_transform(report):
    """world_from_optical of the preview frame, rebuilt from the initial tool pose (replay_visual_tracking.py)."""
    camera = report["camera"]
    pose = report["initial_tool_pose_wxyz"][0]
    rotation = np.column_stack([tool_mouth_geometry(pose, closing_axis_tool=axis)[1] for axis in np.eye(3)])
    return _transform(
        np.asarray(pose[:3]) + rotation @ camera["wrist_position_in_tool_m"],
        rotation @ np.asarray(camera["wrist_rotation_in_tool_ros"]),
    )


def genuine_external_stops(report, frames):
    """Frame -> stop reason injected by the runner before ``observe`` (-1 = before the preview observation).

    A recorded ``external_stop_reason`` that first appears while the recorded cut
    was not yet stopped is an outside event (ToF guard, tracking error, initial
    visibility). Once the cut has stopped itself the runner re-asserts that same
    reason every frame; the replay reproduces that from its own controller.
    """
    stops = {}
    initial = report.get("initial_live_vision") or {}
    if initial.get("external_stop_reason"):
        stops[-1] = initial["external_stop_reason"]
    for index, frame in enumerate(frames):
        reason = frame["live_vision"].get("external_stop_reason")
        if index == 0:
            previous_stopped = (initial.get("cut") or {}).get("phase") == "stopped"
            previous_reason = None
        else:
            previous_stopped = frames[index - 1]["live_vision"]["cut"]["phase"] == "stopped"
            previous_reason = frames[index - 1]["live_vision"].get("external_stop_reason")
        if reason and not previous_stopped and not previous_reason:
            stops[index] = reason
    return stops


def build_demo(report, jaw_self_mask, closure_hold, depth_appearance=False):
    """The controller exactly as the runner constructed it, plus the requested arms."""
    target = report["blender_scene"]["target"]
    initial = report["initial_live_vision"]
    strategy = report.get("approach_strategy")
    tracker_config = report.get("tracker_config") or initial["tracker_config"]
    return VisionPruningDemo(
        target["id"],
        target["axis_w"],
        target["radius_m"],
        report["initial_tool_pose_wxyz"][0],
        closing_axis_tool=tuple(float(value) for value in initial["closing_axis_tool"]),
        photometric_normalization=report.get("photometric_normalization", "raw"),
        approach=ApproachStrategy(**strategy) if strategy else ApproachStrategy(),
        motion_model=tracker_config.get("motion_model", "translation"),
        jaw_self_mask=jaw_self_mask,
        closure_hold=closure_hold,
        depth_appearance=depth_appearance,
    )


def _json_normalized(value):
    return json.loads(json.dumps(value))


def tracker_config_matches(demo, report):
    """Every recorded tracker option equals the rebuilt one; an option added since holds its default."""
    recorded = _json_normalized(report.get("tracker_config") or report["initial_live_vision"]["tracker_config"])
    built = _json_normalized(demo.evidence()["tracker_config"])
    defaults = _json_normalized(asdict(VisualServoConfig()))
    if not set(recorded) <= set(built):
        return False
    return all(recorded[key] == value if key in recorded else value == defaults[key] for key, value in built.items())


def same_measurement(replayed, recorded):
    """The replay criterion: state, reason and pixel_xy exactly, patch_correlation within the tolerance."""
    if replayed.get("state") != recorded.get("state") or replayed.get("reason") != recorded.get("reason"):
        return False
    if replayed.get("pixel_xy") != recorded.get("pixel_xy"):
        return False
    a, b = replayed.get("patch_correlation"), recorded.get("patch_correlation")
    if (a is None) != (b is None):
        return False
    return a is None or abs(a - b) <= CORRELATION_TOLERANCE


def _cut_decision(cut):
    cut = cut or {}
    return (cut.get("phase"), cut.get("stopped_reason"), bool(cut.get("detach_event")))


def _first(frames, predicate):
    return next((frame["index"] for frame in frames if predicate(frame)), None)


def _rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def _command_difference_m(decision, recorded_decision):
    """World-frame difference between two applied command steps (0 for two holds); None if only one moved."""
    mine = (decision or {}).get("delta_world_m")
    theirs = (recorded_decision or {}).get("delta_world_m")
    if mine is None and theirs is None:
        return 0.0
    if mine is None or theirs is None:
        return None
    return float(np.linalg.norm(np.asarray(mine, dtype=float) - np.asarray(theirs, dtype=float)))


def _frame_row(index, time_s, evidence, recorded_frame, rendered_progress, command_phase, injected, decision):
    measurement = evidence["measurement"] or {}
    cut = evidence["cut"] or {}
    certificate = cut.get("certificate") or {}
    jaw = evidence.get("jaw_self_mask")
    recorded = recorded_frame["live_vision"]
    recorded_measurement = recorded.get("measurement") or {}
    recorded_cut = recorded.get("cut") or {}
    recorded_progress = recorded_frame.get("visual_jaw_closure_progress")
    command_difference = _command_difference_m(decision, recorded_frame.get("visual_servo_decision"))
    return {
        "index": index,
        "time_s": time_s,
        "state": measurement.get("state"),
        "reason": measurement.get("reason"),
        "pixel_xy": measurement.get("pixel_xy"),
        "patch_correlation": measurement.get("patch_correlation"),
        "patch_correlation_kept": measurement.get("jaw_mask_patch_correlation_kept"),
        "unmasked_patch_elements": measurement.get("jaw_mask_patch_unmasked"),
        "masked_patch_elements_prev_cur": None
        if "jaw_mask_patch_masked_prev" not in measurement
        else [measurement["jaw_mask_patch_masked_prev"], measurement["jaw_mask_patch_masked_cur"]],
        "mask_dropped_valid_features": measurement.get("jaw_mask_dropped_valid"),
        "unmasked_flow_would_pass": measurement.get("jaw_mask_unmasked_flow_would_pass"),
        "feature_count": measurement.get("feature_count"),
        "flow_reason": measurement.get("flow_reason"),
        "target_position_world_m": measurement.get("target_position_world_m"),
        "cut_phase": cut.get("phase"),
        "stopped_reason": cut.get("stopped_reason"),
        "closure_progress": cut.get("closure_progress"),
        "detach_event": cut.get("detach_event"),
        "mouth_distance_m": certificate.get("mouth_distance_m"),
        "vision_source": certificate.get("vision_source"),
        "held_target_age_s": certificate.get("held_target_age_s"),
        "waived_checks": list(certificate.get("waived_checks") or []),
        "closure_hold": evidence.get("closure_hold"),
        "cut_without_hold": evidence.get("cut_without_hold"),
        "depth_appearance": measurement.get("depth_appearance"),
        "jaw_mask_pixel_count": None if jaw is None else jaw["mask_pixel_count"],
        "closure_progress_used": rendered_progress,
        "recorded_closure_progress": recorded_progress,
        "rendered_progress_matches_recording": recorded_progress is not None
        and abs(float(recorded_progress) - rendered_progress) <= 1e-12,
        "external_stop_injected": injected,
        "command_phase": command_phase,
        "command_difference_mm": None if command_difference is None else 1e3 * command_difference,
        "recorded": {
            "state": recorded_measurement.get("state"),
            "reason": recorded_measurement.get("reason"),
            "pixel_xy": recorded_measurement.get("pixel_xy"),
            "patch_correlation": recorded_measurement.get("patch_correlation"),
            "cut_phase": recorded_cut.get("phase"),
            "stopped_reason": recorded_cut.get("stopped_reason"),
            "detach_event": recorded_cut.get("detach_event"),
            "phase": recorded_frame.get("phase"),
        },
        "measurement_matches_recording": same_measurement(measurement, recorded_measurement),
        "cut_matches_recording": _cut_decision(cut) == _cut_decision(recorded_cut),
        "target_difference_mm": _target_difference_mm(measurement, recorded_measurement),
    }


def _target_difference_mm(measurement, recorded):
    mine, theirs = measurement.get("target_position_world_m"), recorded.get("target_position_world_m")
    if mine is None or theirs is None:
        return None
    return float(1e3 * np.linalg.norm(np.asarray(mine, dtype=float) - np.asarray(theirs, dtype=float)))


def _comparison(rows):
    correlation = [
        abs(row["patch_correlation"] - row["recorded"]["patch_correlation"])
        for row in rows
        if row["patch_correlation"] is not None and row["recorded"]["patch_correlation"] is not None
    ]
    return {
        "frames_compared": len(rows),
        "measurement_mismatch_frames": [row["index"] for row in rows if not row["measurement_matches_recording"]],
        "cut_mismatch_frames": [row["index"] for row in rows if not row["cut_matches_recording"]],
        "command_phase_mismatch_frames": [
            row["index"] for row in rows if row["command_phase"] != row["recorded"]["phase"]
        ],
        "max_abs_patch_correlation_difference": max(correlation, default=0.0),
    }


def replay_run(path, *, jaw_self_mask=False, closure_hold=False, depth_appearance=False):
    """Replay one recorded run; returns the per-run summary with its per-frame rows."""
    path = Path(path)
    report = json.loads((path / "report.json").read_text())
    document = json.loads((path / "frames.json").read_text())
    frames, fps = document["frames"], float(document["fps"])
    if [frame["index"] for frame in frames] != list(range(len(frames))):
        raise ValueError(f"{path}: frame indexes are not 0..N-1")
    camera_matrix = np.asarray(report["camera"]["wrist_intrinsics"], dtype=float)
    initial_pose = np.asarray(report["initial_tool_pose_wxyz"][0], dtype=float)
    stops = genuine_external_stops(report, frames)
    demo = build_demo(report, jaw_self_mask, closure_hold, depth_appearance)

    # Seed and preview, in the runner's order: visibility stop, initialize, preview observe at t = 0.
    preview_rgb = _rgb(path / "preview_wrist.png")
    preview_depth = np.load(path / "preview_depth.npy", allow_pickle=False)
    preview_camera = preview_transform(report)
    if -1 in stops:
        demo.stop(stops[-1])
    recorded_init = (report.get("vision_initialization") or {}).get("tracker") or {}
    if recorded_init.get("state") == "initialization_rejected":
        initialized = {"state": "initialization_rejected", "feature_count": 0}
    else:
        initialized = demo.initialize(
            preview_rgb,
            preview_depth,
            report["vision_initialization"]["pixel_xy"],
            camera_matrix=camera_matrix,
            world_from_optical=preview_camera,
            tool_pose_wxyz=initial_pose,
        )
    preview = demo.observe(preview_rgb, preview_depth, camera_matrix, preview_camera, 0.0, initial_pose)
    recorded_preview = report.get("initial_live_vision") or {}

    rows = []
    stopped_reason = None
    missing_contact = []
    for frame in frames:
        index = frame["index"]
        before = np.asarray(frames[index - 1]["tool_pose_wxyz"] if index else initial_pose, dtype=float)
        _, command_phase, decision = demo.command(before, index / fps)
        if stopped_reason is not None:
            command_phase, decision = "stopped_failure", {"state": "hold", "reason": stopped_reason}
            demo.stop(stopped_reason)
        demo.command_applied(command_phase, decision)
        injected = None
        if stopped_reason is None and index in stops:
            stopped_reason = injected = stops[index]
            if injected.startswith("tool_tracking_error"):
                command_phase = "stopped_failure"
        if stopped_reason:
            demo.stop(stopped_reason)
        contact = frame.get("contact_force_n")
        if contact is None:
            missing_contact.append(index)
        rendered_progress = demo.cut_step.closure_progress if demo.cut_step else 0.0
        transform = _transform(frame["wrist_position_w_m"], frame["wrist_rotation_w_ros"])
        evidence = demo.observe(
            _rgb(path / f"frames/wrist_{index:05d}.png"),
            np.load(path / f"frames/depth_{index:05d}.npy", allow_pickle=False),
            camera_matrix,
            transform,
            (index + 1) / fps,
            np.asarray(frame["tool_pose_wxyz"], dtype=float),
            hazard_contact=contact is not None and float(contact) > HAZARD_CONTACT_N,
        )
        if demo.cut_step.phase == "stopped" and stopped_reason is None:
            stopped_reason = demo.cut_step.stopped_reason
        rows.append(
            _frame_row(index, (index + 1) / fps, evidence, frame, rendered_progress, command_phase, injected, decision)
        )

    recorded_stop = _first(frames, lambda f: f["live_vision"]["cut"]["phase"] == "stopped")
    recorded_detach = _first(frames, lambda f: f["live_vision"]["cut"].get("detach_event"))
    last_compared = recorded_stop if recorded_stop is not None else len(frames) - 1
    through = _comparison(rows[: last_compared + 1])
    after = _comparison(rows[last_compared + 1 :])
    first_divergence = next((row["index"] for row in rows if not row["cut_matches_recording"]), None)
    first_render_mismatch = next((row["index"] for row in rows if not row["rendered_progress_matches_recording"]), None)
    # A command decided from frame k-1 moves the robot before frame k is captured, so frame k itself already
    # shows the recorded command's outcome, not the replayed one's.
    first_command_divergence = next(
        (
            row["index"]
            for row in rows
            if row["command_phase"] != row["recorded"]["phase"]
            or row["command_difference_mm"] is None
            or row["command_difference_mm"] > COMMAND_TOLERANCE_MM
        ),
        None,
    )
    for row in rows:
        row["after_recorded_stop"] = recorded_stop is not None and row["index"] > recorded_stop
        row["after_first_cut_decision_divergence"] = first_divergence is not None and row["index"] > first_divergence
        row["evaluable"] = not (
            row["after_recorded_stop"]
            or row["after_first_cut_decision_divergence"]
            or not row["rendered_progress_matches_recording"]
        )
        # Exact: every command up to this frame equals the recorded one, so the image is exactly what this
        # controller would have seen. Evaluable but not exact frames are approximate counterfactuals.
        row["exact_counterfactual"] = row["evaluable"] and (
            first_command_divergence is None or row["index"] < first_command_divergence
        )
    preview_matches = same_measurement(preview["measurement"], recorded_preview.get("measurement") or {}) and (
        _cut_decision(preview["cut"]) == _cut_decision(recorded_preview.get("cut"))
    )
    held = [row["index"] for row in rows if (row["closure_hold"] or {}).get("held")]
    shadow_stop = next(
        (row for row in rows if (row["cut_without_hold"] or {}).get("phase") == "stopped"),
        None,
    )
    initialization_matches = initialized.get("state") == recorded_init.get("state") and initialized.get(
        "feature_count"
    ) == recorded_init.get("feature_count", 0)
    reproduces = (
        initialization_matches
        and preview_matches
        and not through["measurement_mismatch_frames"]
        and not through["cut_mismatch_frames"]
    )
    return {
        "run": f"{path.parent.name}/{path.name}",
        "path": str(path),
        "frames": len(frames),
        "flags": {"jaw_self_mask": jaw_self_mask, "closure_hold": closure_hold, "depth_appearance": depth_appearance},
        "tracker_config_matches_recording": tracker_config_matches(demo, report),
        "external_stops_injected": {str(key): value for key, value in sorted(stops.items())},
        "contact_force_missing_frames": missing_contact,
        "recorded": {
            "task_outcome": report.get("task_outcome"),
            "stop_frame": recorded_stop,
            "stop_reason": None
            if recorded_stop is None
            else frames[recorded_stop]["live_vision"]["cut"]["stopped_reason"],
            "detach_frame": recorded_detach,
            "closure_start_frame": _first(frames, lambda f: f["live_vision"]["cut"]["phase"] == "closing"),
        },
        "replay": {
            "stop_frame": _first(rows, lambda r: r["cut_phase"] == "stopped"),
            "stop_reason": next((r["stopped_reason"] for r in rows if r["cut_phase"] == "stopped"), None),
            "detach_frame": _first(rows, lambda r: r["detach_event"]),
            "closure_start_frame": _first(rows, lambda r: r["cut_phase"] == "closing"),
            "held_frames": held,
            "cut_without_hold_stop_frame": None if shadow_stop is None else shadow_stop["index"],
            "cut_without_hold_stop_reason": None
            if shadow_stop is None
            else shadow_stop["cut_without_hold"]["stopped_reason"],
            "first_jaw_mask_occluded_frame": _first(rows, lambda r: r["state"] == "jaw_mask_occluded"),
        },
        "initialization": {
            "replayed_state": initialized.get("state"),
            "recorded_state": recorded_init.get("state"),
            "replayed_feature_count": initialized.get("feature_count"),
            "recorded_feature_count": recorded_init.get("feature_count"),
            "jaw_self_mask": initialized.get("jaw_self_mask"),
            "matches": initialization_matches,
        },
        "preview": {
            "matches_recording": preview_matches,
            "state": preview["measurement"].get("state"),
            "reason": preview["measurement"].get("reason"),
            "pixel_xy": preview["measurement"].get("pixel_xy"),
            "patch_correlation": preview["measurement"].get("patch_correlation"),
            "depth_appearance": preview["measurement"].get("depth_appearance"),
            "cut_phase": (preview["cut"] or {}).get("phase"),
            "jaw_mask_pixel_count": (preview.get("jaw_self_mask") or {}).get("mask_pixel_count"),
        },
        "compared_through_frame": last_compared,
        "comparison_through_recorded_stop": through,
        "comparison_after_recorded_stop": after,
        "first_measurement_divergence_frame": next(
            (row["index"] for row in rows if not row["measurement_matches_recording"]), None
        ),
        "first_cut_decision_divergence_frame": first_divergence,
        "first_command_divergence_frame": first_command_divergence,
        "max_command_difference_mm_while_evaluable": max(
            (row["command_difference_mm"] or 0.0 for row in rows if row["evaluable"]), default=0.0
        ),
        "max_target_difference_mm_while_evaluable": max(
            (row["target_difference_mm"] or 0.0 for row in rows if row["evaluable"]), default=0.0
        ),
        "first_rendered_progress_mismatch_frame": first_render_mismatch,
        "reproduces_recording_through_recorded_stop": reproduces,
        "frames_detail": rows,
    }


def _provenance():
    hashes = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    try:
        revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(
            subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--", *SOURCE_FILES], text=True)
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"code_revision": revision, "source_files_differ_from_revision": dirty, "source_sha256": hashes}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-dir", type=Path, action="append", help="A recorded run directory (repeatable)")
    selection.add_argument("--all", action="store_true", help="Every recorded run with frames, except jaw-shadow-*")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    parser.add_argument("--jaw-self-mask", action="store_true", help="Enable the jaw self-mask arm")
    parser.add_argument("--closure-hold", action="store_true", help="Enable the closure-hold arm")
    parser.add_argument("--depth-appearance", action="store_true", help="Enable the D_strict + J arm")
    args = parser.parse_args(argv)
    refusal = output_refusal(args.output)
    if refusal:
        parser.error(refusal)
    import cv2

    cv2.setNumThreads(1)
    runs = discover_runs() if args.all else [Path(path) for path in args.run_dir]
    results = []
    for path in runs:
        try:
            result = replay_run(
                path,
                jaw_self_mask=args.jaw_self_mask,
                closure_hold=args.closure_hold,
                depth_appearance=args.depth_appearance,
            )
        except Exception as error:  # noqa: BLE001 - a failed replay is recorded, never dropped
            result = {"run": f"{Path(path).parent.name}/{Path(path).name}", "path": str(path), "error": repr(error)}
        results.append(result)
        if "error" in result:
            print(f"{result['run']}: ERROR {result['error']}", flush=True)
            continue
        print(
            f"{result['run']}: reproduces={result['reproduces_recording_through_recorded_stop']} "
            f"through={result['compared_through_frame']} "
            f"mismatch m/c={len(result['comparison_through_recorded_stop']['measurement_mismatch_frames'])}/"
            f"{len(result['comparison_through_recorded_stop']['cut_mismatch_frames'])} "
            f"recorded stop={result['recorded']['stop_frame']},{result['recorded']['stop_reason']} "
            f"detach={result['recorded']['detach_frame']} | replay stop={result['replay']['stop_frame']},"
            f"{result['replay']['stop_reason']} detach={result['replay']['detach_frame']} "
            f"held={result['replay']['held_frames']} shadow={result['replay']['cut_without_hold_stop_frame']},"
            f"{result['replay']['cut_without_hold_stop_reason']}",
            flush=True,
        )
    ok = [r for r in results if "error" not in r]
    document = {
        "schema_version": 1,
        "tool": "tools/replay_jaw_self_mask.py",
        "flags": {
            "jaw_self_mask": args.jaw_self_mask,
            "closure_hold": args.closure_hold,
            "depth_appearance": args.depth_appearance,
        },
        **_provenance(),
        "criterion": (
            "Through each run's recorded stop (the first frame whose recorded cut phase is 'stopped'; every frame "
            "otherwise): state, reason and pixel_xy exactly, patch_correlation within 1e-6, and the cut phase, stop "
            "reason and detach event exactly; plus the seed initialization and the preview observation."
        ),
        "scope": (
            "Offline replay of recorded images through the repository controller. With a flag on, frames after the "
            "first cut-decision divergence, after the recorded stop, or rendered with a different closure progress "
            "are not evaluable; evaluable frames rendered after the first differing command are approximate "
            "(exact_counterfactual=false). No frame here is a live outcome or a grade."
        ),
        "summary": {
            "runs": len(results),
            "errors": [r["run"] for r in results if "error" in r],
            "reproduce_recording_through_recorded_stop": sum(
                r["reproduces_recording_through_recorded_stop"] for r in ok
            ),
            "do_not_reproduce": [r["run"] for r in ok if not r["reproduces_recording_through_recorded_stop"]],
            "tracker_config_mismatch": [r["run"] for r in ok if not r["tracker_config_matches_recording"]],
        },
        "runs": results,
    }
    with args.output.open("x") as stream:
        json.dump(_json_safe(document), stream, allow_nan=False)
        stream.write("\n")
    print(json.dumps(document["summary"]), flush=True)
    return 0


def _json_safe(value):
    """Strict JSON: non-finite floats become null (as the runner records them), numpy values become Python."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value


if __name__ == "__main__":
    raise SystemExit(main())
