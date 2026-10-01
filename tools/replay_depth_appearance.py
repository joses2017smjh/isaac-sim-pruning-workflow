#!/usr/bin/env python3
"""Offline replay of the depth-aware appearance check ``D_strict + J`` and its labelled ``agreement`` arm.

Implements exactly the variant registered in
docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_HELDOUT_2026-10-01.md (H1-H4). CPU only.
The tracker itself lives in ``perception/depth_appearance.py``, where it was moved
unchanged for the closed-loop test (tools/check_depth_loop_c0.py checks the move
against the published replay); this tool replays recorded runs through it.

Scope. A subclass of the repository ``VisualServoTracker`` that differs from it
only when the grey NCC appearance check fails (below ``min_patch_correlation``,
0.35). It then runs a static-world depth test on the previous 13 x 13 patch and
the jaw guard J. On acceptance the 0.35 gate passes. In the primary arm,
``strict``, the raw NCC stays in the unchanged confidence
``min(1, n/n0) * NCC * depth_valid_fraction >= 0.15``. The secondary arm,
``agreement``, replaces the NCC by the depth-agreement fraction in that
confidence, which changes the confidence gate's input; it is reported
separately and never pooled with ``strict``. Every other check (optical flow,
depth window, world jump, replenishment) is the repository code, called
unchanged through ``super()``.

The depth test (registered thresholds; module constants, not options):
  * back-project the previous patch rint(p) + [-6..6]^2 with the previous
    frame's optical-Z and pose (pixel index u has its centre at u + 0.5);
  * reproject it with the current pose and sample the current depth with the
    3 x 3 median rule (median of the valid samples, invalid unless at least
    ``min_depth_valid_fraction`` of the window is valid; unlike the tracker's
    own depth window it has no spread or centre check). As in the study's
    median3 path (depthcheck.py ``sample_depth``, run as D_median3 at the 8
    earlier grey-check failures; its 129-run regression used nearest-pixel
    sampling), the same rule also gives the previous patch's optical-Z;
  * keep the elements within 25 mm of the tracked depth (the median of the
    tracker's own depth window at p in the previous frame): the same-surface
    elements;
  * accept only if (1) >= 75% of the same-surface elements are verified (land
    in the image with a valid current depth), (2) >= 20 same-surface elements
    remain, (3) <= 5% of the verified elements are nearer than predicted by
    more than 10 mm, (4) the median |measured - predicted| over the verified
    elements is <= 3 mm, (5) the propagated pixel is within 3 px of the
    static-world prediction of p, and
    (J) the committed 2.0 px jaw silhouette
    (``perception/jaw_self_mask.jaw_mask``) touches none of the previous,
    reprojected or propagated patches. "Touches" is any mask pixel in the
    patch footprint: the ``getRectSubPix`` footprint (any nonzero weight, as the
    committed tracker samples a mask) for the previous patch (previous frame's
    mask) and the propagated patch (current mask), and the rounded reprojected
    same-surface elements in the current mask. J fails closed when the
    silhouette is undefined (a jaw corner at or behind the camera) or no jaw
    pose was supplied.

Label. Simulator depth: RTX optical-Z ground truth, noise-free, perfectly
registered, exact camera motion; not a sensor. The variant changes the 0.35
appearance gate's rule.

Replay recipe (tools/replay_visual_tracking.py, tools/replay_jaw_self_mask.py):
the tracker config is frames[0].live_vision.tracker_config; the seed
initialization uses the preview image and depth (skipped when the recording
rejected it); the preview update uses the camera pose rebuilt from the initial
tool pose; every frame update uses the recorded wrist pose. Jaw boxes use the
recorded tool pose, the attachment roll ``proxy_roll_rad(closing_axis_tool)``,
the closure progress rendered into the frame and the scene's branch radius.

Offline-only limits. Only decisions up to each run's recorded stop (the first
frame whose recorded cut phase is ``stopped``; else the last frame) are
replayed: frames after a stop show a stopped robot. A divergence shows that the
recorded stop would not have happened at that frame, not what follows. No
sensor noise, pose error or time-sync error is modelled, and same-depth
occluders were never recorded. Nothing here is a live outcome or a grade; only
tools/validate_vision_sequence.py grades live runs. Inputs are read only; the
output must be a new file outside the repository (and artifacts/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/isaaclab_pruning"))
from isaaclab_pruning.perception.depth_appearance import (  # noqa: F401 - re-exported for the tests and scorers
    ARM_LABELS,
    ARMS,
    CONDITION_REASONS,
    MAX_MEDIAN_ABS_M,
    MAX_NEAR_FRACTION,
    MAX_PIXEL_DISAGREEMENT_PX,
    MIN_SAME_SURFACE_PX,
    MIN_VERIFIED_FRACTION,
    NEAR_TOLERANCE_M,
    SAME_SURFACE_M,
    SIMULATOR_DEPTH_LABEL,
    DepthAppearanceTracker,
    depth_test,
    frame_jaw_boxes,
    jaw_contact,
)
from isaaclab_pruning.perception.jaw_self_mask import MASK_MARGIN_PX, proxy_roll_rad
from isaaclab_pruning.perception.visual_servo import VisualServoConfig, VisualServoTracker
from isaaclab_pruning.task.simulated_cut import tool_mouth_geometry

VISION_ROBUSTNESS = ROOT / "artifacts/vision_robustness"
HELDOUT_BATCH_GLOB = "jaw-shadow-*-20260930"
#: Batches recorded after the protocol (the held-out counterfactual and the live jaw-in-view runs).
REGRESSION_EXCLUDED_PREFIXES = ("jaw-shadow-", "jaw-hold-")
#: The registered data sets: 12 held-out recordings and the 129 earlier runs with frames.
EXPECTED_RUNS = {"heldout": 12, "regression": 129}
CORRELATION_TOLERANCE = 1e-6
PROTOCOL = "docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_HELDOUT_2026-10-01.md"
SOURCE_FILES = (
    "source/isaaclab_pruning/isaaclab_pruning/perception/depth_appearance.py",
    "source/isaaclab_pruning/isaaclab_pruning/perception/visual_servo.py",
    "source/isaaclab_pruning/isaaclab_pruning/perception/jaw_self_mask.py",
    "source/isaaclab_pruning/isaaclab_pruning/task/simulated_cut.py",
    "tools/replay_depth_appearance.py",
)


# ------------------------------------------------------------------------------------------------ recorded runs
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


def tracker_config(frames):
    config = dict(frames[0]["live_vision"]["tracker_config"])
    config["roi_half_size_px"] = tuple(config["roi_half_size_px"])
    return VisualServoConfig(**config)


def _has_frames(path):
    return (
        (path / "report.json").is_file()
        and (path / "frames.json").is_file()
        and (path / "frames").is_dir()
        and any((path / "frames").glob("wrist_*.png"))
    )


def discover_heldout(base=VISION_ROBUSTNESS):
    """The jaw-shadow counterfactual's run directories (the registered held-out data)."""
    return [path for path in sorted(Path(base).glob(f"{HELDOUT_BATCH_GLOB}/run_*")) if _has_frames(path)]


def discover_regression(base=VISION_ROBUSTNESS):
    """Every earlier recorded run with frames (outside the batches recorded after the protocol)."""
    return [
        path
        for path in sorted(Path(base).glob("*/run_*"))
        if not path.parent.name.startswith(REGRESSION_EXCLUDED_PREFIXES) and _has_frames(path)
    ]


def output_refusal(path):
    """Why the output path is not allowed (inside the repository or artifacts/, or existing), else None."""
    resolved = Path(path).expanduser().resolve()
    for forbidden in (ROOT.resolve(), (ROOT / "artifacts").resolve(), Path.home().resolve()):
        if resolved == forbidden or forbidden in resolved.parents:
            return f"refusing to write inside {forbidden}: {resolved}"
    if resolved.exists():
        return f"refusing to overwrite an existing file: {resolved}"
    if not resolved.parent.is_dir():
        return f"the output directory does not exist: {resolved.parent}"
    return None


def same_measurement(replayed, recorded):
    """Differing fields: state, reason and pixel_xy exactly, patch_correlation within the tolerance."""
    fields = [key for key in ("state", "reason", "pixel_xy") if replayed.get(key) != recorded.get(key)]
    a, b = replayed.get("patch_correlation"), recorded.get("patch_correlation")
    if (a is None) != (b is None) or (a is not None and abs(a - b) > CORRELATION_TOLERANCE):
        fields.append("patch_correlation")
    return fields


def _brief(measurement):
    measurement = measurement or {}
    keys = ("state", "reason", "pixel_xy", "patch_correlation", "feature_count", "confidence", "depth_reason")
    out = {key: measurement.get(key) for key in keys}
    for key in (
        "measured_confidence",
        "confidence_with_gate_value",
        "appearance_gate_value",
        "depth_valid_fraction",
        "world_jump_m",
        "target_position_world_m",
    ):
        if key in measurement:
            out[key] = measurement[key]
    return out


def _rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _event_record(index, event):
    record = {"frame": index, **{key: value for key, value in event.items() if key != "stats"}}
    record.update(event.get("stats") or {})
    return record


def replay_run(path):
    """Replay one recorded run through the base tracker and both arms in lockstep, up to its recorded stop."""
    path = Path(path)
    report = json.loads((path / "report.json").read_text())
    if report.get("stage") != "complete":
        raise ValueError(f"{path}: the recording is not complete (stage {report.get('stage')!r})")
    if any((report.get(arm) or {}).get("enabled") for arm in ("jaw_self_mask", "closure_hold")):
        raise ValueError(f"{path}: recorded with the jaw self-mask or closure hold, which this replay does not model")
    frames = json.loads((path / "frames.json").read_text())["frames"]
    if [frame["index"] for frame in frames] != list(range(len(frames))):
        raise ValueError(f"{path}: frame indexes are not 0..N-1")
    camera_matrix = np.asarray(report["camera"]["wrist_intrinsics"], dtype=float)
    config = tracker_config(frames)
    trackers = {"base": VisualServoTracker(config)}
    trackers.update({arm: DepthAppearanceTracker(config, arm=arm) for arm in ARMS})
    initial_live = report.get("initial_live_vision") or {}
    closing_axis = initial_live.get("closing_axis_tool") or frames[0]["live_vision"]["closing_axis_tool"]
    roll = proxy_roll_rad(closing_axis)
    radius = float(report["blender_scene"]["target"]["radius_m"])
    recorded_stop = next((f["index"] for f in frames if f["live_vision"]["cut"]["phase"] == "stopped"), None)
    detach = next((f["index"] for f in frames if f["live_vision"]["cut"].get("detach_event")), None)
    last = recorded_stop if recorded_stop is not None else len(frames) - 1

    recorded_preview = initial_live.get("measurement") or {}
    preview_rgb = _rgb(path / "preview_wrist.png")
    preview_depth = np.load(path / "preview_depth.npy", allow_pickle=False)
    preview_pose = preview_transform(report)
    recorded_init = (report.get("vision_initialization") or {}).get("tracker") or {}
    rejected = recorded_init.get("state") == "initialization_rejected"
    seed = report["vision_initialization"]["pixel_xy"]
    init = {name: None if rejected else t.initialize(preview_rgb, seed, preview_depth) for name, t in trackers.items()}
    preview_boxes = frame_jaw_boxes(report["initial_tool_pose_wxyz"][0], roll, 0.0, radius)
    preview = {
        name: tracker.update(preview_rgb, preview_depth, camera_matrix, preview_pose)
        if name == "base"
        else tracker.update(preview_rgb, preview_depth, camera_matrix, preview_pose, jaw_boxes=preview_boxes)
        for name, tracker in trackers.items()
    }

    per_arm = {name: {"mismatches": [], "max_corr": 0.0} for name in trackers}
    events = {arm: [] for arm in ARMS}
    for name, tracker in trackers.items():
        if name != "base" and tracker.last_event is not None:
            record = _event_record("preview", tracker.last_event)
            record.update({"recorded": _brief(recorded_preview), "replayed": _brief(preview[name])})
            events[name].append(record)
    at_stop = {}
    for frame in frames[: last + 1]:
        index = frame["index"]
        rgb = _rgb(path / f"frames/wrist_{index:05d}.png")
        depth = np.load(path / f"frames/depth_{index:05d}.npy", allow_pickle=False)
        pose = _transform(frame["wrist_position_w_m"], frame["wrist_rotation_w_ros"])
        boxes = frame_jaw_boxes(frame["tool_pose_wxyz"], roll, frame.get("visual_jaw_closure_progress") or 0.0, radius)
        recorded = frame["live_vision"]["measurement"]
        for name, tracker in trackers.items():
            if name == "base":
                result = tracker.update(rgb, depth, camera_matrix, pose)
            else:
                result = tracker.update(rgb, depth, camera_matrix, pose, jaw_boxes=boxes)
                if tracker.last_event is not None:
                    record = _event_record(index, tracker.last_event)
                    record["after_recorded_detach"] = detach is not None and index > detach
                    record["recorded"] = _brief(recorded)
                    record["recorded_cut_phase"] = frame["live_vision"]["cut"]["phase"]
                    record["replayed"] = _brief(result)
                    events[name].append(record)
            fields = same_measurement(result, recorded)
            a, b = result.get("patch_correlation"), recorded.get("patch_correlation")
            if a is not None and b is not None:
                per_arm[name]["max_corr"] = max(per_arm[name]["max_corr"], abs(a - b))
            if fields:
                per_arm[name]["mismatches"].append(
                    {
                        "frame": index,
                        "fields": fields,
                        "recorded": _brief(recorded),
                        "replayed": _brief(result),
                        "after_recorded_detach": detach is not None and index > detach,
                        "event_frame": name != "base" and tracker.last_event is not None,
                    }
                )
            if index == recorded_stop:
                at_stop[name] = _brief(result)

    divergence = {}
    for name in trackers:
        mismatches = per_arm[name]["mismatches"]
        preview_fields = same_measurement(preview[name], recorded_preview)
        divergence[name] = {
            "preview_mismatch_fields": preview_fields,
            "mismatch_frames": [item["frame"] for item in mismatches],
            "first_divergence_frame": mismatches[0]["frame"] if mismatches else None,
            "divergences": mismatches,
            "max_abs_patch_correlation_difference": per_arm[name]["max_corr"],
            "equals_recording": not mismatches and not preview_fields,
        }
    base_init = init["base"]
    init_check = {
        "recorded": [recorded_init.get("state"), recorded_init.get("reason"), recorded_init.get("feature_count", 0)],
        "replayed": None
        if base_init is None
        else [base_init["state"], base_init["reason"], base_init["feature_count"]],
    }
    init_check["matches"] = rejected or init_check["replayed"] == init_check["recorded"]
    base_exact = init_check["matches"] and divergence["base"]["equals_recording"]
    stop_measurement = frames[recorded_stop]["live_vision"]["measurement"] if recorded_stop is not None else None
    joined = _join_events(events)
    return {
        "run": f"{path.parent.name}/{path.name}",
        "path": str(path),
        "frames": len(frames),
        "source_sha256": {name: _sha256(path / name) for name in ("report.json", "frames.json")},
        "recorded": {
            "task_outcome": report.get("task_outcome"),
            "stop_frame": recorded_stop,
            "stop_reason": None
            if recorded_stop is None
            else frames[recorded_stop]["live_vision"]["cut"]["stopped_reason"],
            "stop_measurement": _brief(stop_measurement) if stop_measurement else None,
            "detach_frame": detach,
            "initialization_state": recorded_init.get("state"),
        },
        "replayed_through_frame": last,
        "initialization": init_check,
        "base_reproduces_recording": base_exact,
        "variant_results_interpretable": base_exact,
        "measurement_at_recorded_stop": at_stop,
        "divergence": divergence,
        "events": events,
        "events_joined": joined,
        "grey_failures": sorted(
            {event["frame"] for arm in ARMS for event in events[arm]}, key=lambda f: -1 if f == "preview" else f
        ),
    }


def _join_events(events):
    """Per frame: the depth test and J (from the strict arm when both arms saw the frame) and both arms' decisions."""
    by_frame = {}
    for arm in ARMS:
        for event in events[arm]:
            by_frame.setdefault(str(event["frame"]), {})[arm] = event
    joined = []
    for frame, arms in by_frame.items():
        source = arms.get("strict") or arms.get("agreement")
        same_input = (
            "strict" in arms
            and "agreement" in arms
            and arms["strict"].get("previous_pixel") == arms["agreement"].get("previous_pixel")
            and arms["strict"].get("propagated_pixel") == arms["agreement"].get("propagated_pixel")
        )
        row = {
            "frame": source["frame"],
            "arms_saw_the_same_update": same_input,
            "ncc": source["ncc"],
            "accept": source["accept"],
            "reasons": source["reasons"],
            "failed_conditions": source.get("failed_conditions"),
            "not_evaluated_conditions": source.get("not_evaluated_conditions"),
            "conditions": source.get("conditions"),
            "jaw": source.get("jaw"),
            "strict_confidence": source.get("strict_confidence"),
            "agreement_confidence": source.get("agreement_confidence"),
            "feature_ratio": source.get("feature_ratio"),
            "depth_valid_fraction": source.get("depth_valid_fraction"),
            "initial_feature_count": source.get("initial_feature_count"),
            "reached_confidence_gate": source.get("reached_confidence_gate"),
            "after_recorded_detach": source.get("after_recorded_detach"),
        }
        for key in (
            "tracked_depth_prev_m",
            "n_same_surface",
            "n_verified",
            "verified_fraction",
            "median_abs_m",
            "median_signed_m",
            "near_count",
            "near_fraction",
            "agreement_fraction",
            "pixel_disagreement_px",
        ):
            row[key] = source.get(key)
        for arm in ARMS:
            event = arms.get(arm)
            row[f"{arm}_decision"] = (
                None
                if event is None
                else {
                    "state": (event.get("replayed") or {}).get("state"),
                    "reason": (event.get("replayed") or {}).get("reason"),
                    "continues": (event.get("replayed") or {}).get("state") == "tracking",
                    "accept": event["accept"],
                    "ended_by": event.get("ended_by"),
                }
            )
        joined.append(row)
    return joined


# ------------------------------------------------------------------------------------------------ output
def _json_safe(value):
    """Strict JSON: non-finite floats become null, numpy values become Python."""
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


def _provenance():
    hashes = {name: _sha256(ROOT / name) for name in SOURCE_FILES}
    try:
        revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", *SOURCE_FILES], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"code_revision": revision, "source_files_status_porcelain": dirty, "source_sha256": hashes}


def _summary(results):
    ok = [r for r in results if "error" not in r]
    summary = {
        "runs": len(results),
        "errors": [r["run"] for r in results if "error" in r],
        "base_reproduces_recording": sum(r["base_reproduces_recording"] for r in ok),
        "base_does_not_reproduce": [r["run"] for r in ok if not r["base_reproduces_recording"]],
        "updates_replayed": sum(r["replayed_through_frame"] + 1 for r in ok),
        "runs_with_grey_failures": {r["run"]: r["grey_failures"] for r in ok if r["grey_failures"]},
    }
    for arm in ARMS:
        summary[arm] = {
            "runs_equal_to_recording": sum(r["divergence"][arm]["equals_recording"] for r in ok),
            "diverging_runs": {
                r["run"]: {
                    "frames": r["divergence"][arm]["mismatch_frames"],
                    "preview_fields": r["divergence"][arm]["preview_mismatch_fields"],
                    "states": [
                        [d["frame"], d["recorded"]["state"], d["replayed"]["state"], d["replayed"]["reason"]]
                        for d in r["divergence"][arm]["divergences"]
                    ],
                }
                for r in ok
                if not r["divergence"][arm]["equals_recording"]
            },
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-dir", type=Path, action="append", help="A recorded run directory (repeatable)")
    selection.add_argument("--heldout", action="store_true", help=f"The {HELDOUT_BATCH_GLOB} run directories")
    selection.add_argument("--regression", action="store_true", help="The 129 earlier runs with frames")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    args = parser.parse_args(argv)
    progress = args.output.with_name(args.output.name + ".progress.jsonl")
    for path in (args.output, progress):
        refusal = output_refusal(path)
        if refusal:
            parser.error(refusal)
    import cv2

    cv2.setNumThreads(1)
    if args.heldout:
        selection_name, runs = "heldout", discover_heldout()
    elif args.regression:
        selection_name, runs = "regression", discover_regression()
    else:
        selection_name, runs = "run_dir", [Path(path) for path in args.run_dir]
    if selection_name in EXPECTED_RUNS and len(runs) != EXPECTED_RUNS[selection_name]:
        parser.error(f"{selection_name}: found {len(runs)} runs, registered {EXPECTED_RUNS[selection_name]}")
    header = {
        "header": True,
        "tool": "tools/replay_depth_appearance.py",
        "protocol": PROTOCOL,
        "protocol_sha256": _sha256(ROOT / PROTOCOL),
        "selection": selection_name,
        "runs": [f"{Path(r).parent.name}/{Path(r).name}" for r in runs],
        **_provenance(),
    }
    with progress.open("a") as stream:
        stream.write(json.dumps(_json_safe(header), allow_nan=False) + "\n")
    results = []
    for number, path in enumerate(runs, 1):
        started = time.time()
        try:
            result = replay_run(path)
        except Exception as error:  # noqa: BLE001 - a failed replay is recorded, never dropped
            result = {"run": f"{Path(path).parent.name}/{Path(path).name}", "path": str(path), "error": repr(error)}
        result["replay_seconds"] = round(time.time() - started, 2)
        results.append(result)
        with progress.open("a") as stream:
            stream.write(json.dumps(_json_safe(result), allow_nan=False) + "\n")
        if "error" in result:
            print(f"[{number}/{len(runs)}] {result['run']}: ERROR {result['error']}", flush=True)
            continue
        print(
            f"[{number}/{len(runs)}] {result['run']}: through={result['replayed_through_frame']} "
            f"base_exact={result['base_reproduces_recording']} grey_failures={result['grey_failures']} "
            + " ".join(f"{arm}_div={result['divergence'][arm]['mismatch_frames']}" for arm in ARMS)
            + f" t={result['replay_seconds']}s",
            flush=True,
        )
    document = {
        "schema_version": 1,
        "tool": "tools/replay_depth_appearance.py",
        "protocol": PROTOCOL,
        "protocol_sha256": _sha256(ROOT / PROTOCOL),
        "selection": selection_name,
        "label": SIMULATOR_DEPTH_LABEL,
        "arms": ARM_LABELS,
        **_provenance(),
        "thresholds": {
            "min_patch_correlation_gate": "the tracker config's min_patch_correlation (0.35)",
            "min_confidence": "the tracker config's min_confidence (0.15)",
            "min_verified_fraction": MIN_VERIFIED_FRACTION,
            "min_same_surface_px": MIN_SAME_SURFACE_PX,
            "near_tolerance_m": NEAR_TOLERANCE_M,
            "max_near_fraction": MAX_NEAR_FRACTION,
            "max_median_abs_m": MAX_MEDIAN_ABS_M,
            "max_pixel_disagreement_px": MAX_PIXEL_DISAGREEMENT_PX,
            "same_surface_m": SAME_SURFACE_M,
            "verified_fraction_denominator": "same-surface elements (within 25 mm of the tracked depth)",
            "near_fraction_denominator": "verified same-surface elements",
            "agreement_fraction": "verified elements with |measured - predicted| <= 10 mm, over the verified elements",
            "depth_sampling": (
                "3 x 3 median rule on both the previous and the current depth, as depthcheck.py median3: the median "
                "of the valid samples, invalid unless at least min_depth_valid_fraction (0.75) of the window is "
                "valid; no spread or centre check"
            ),
            "tracked_depth": "median of the valid samples in the tracker's own depth window at p, previous frame",
            "jaw_mask_margin_px": MASK_MARGIN_PX,
            "jaw_touch_rule": (
                "any mask pixel in the getRectSubPix footprint of the previous patch (previous frame's mask) or the "
                "propagated patch (current mask), or at a rounded reprojected same-surface element (current mask); "
                "fails closed when the silhouette is undefined or no jaw pose is known"
            ),
        },
        "criterion": (
            "Per run, through its recorded stop (else every frame): the replayed measurement against the recorded "
            "live_vision.measurement, state, reason and pixel_xy exactly and patch_correlation within 1e-6; plus the "
            "seed initialization and the preview update. Variant results count only where the base tracker "
            "reproduces the recording."
        ),
        "scope": (
            "Offline replay of recorded images only. A divergence shows that the recorded stop would not have "
            "happened at that frame, not what follows; no sensor noise, pose error or time-sync error is modelled."
        ),
        "summary": _summary(results),
        "runs": results,
    }
    with args.output.open("x") as stream:
        json.dump(_json_safe(document), stream, allow_nan=False)
        stream.write("\n")
    print(json.dumps(_json_safe(document["summary"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
