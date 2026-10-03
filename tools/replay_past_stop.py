#!/usr/bin/env python3
"""Replay the depth-aware appearance check past the recorded stop on the shadowed evening 14944 recordings.

Design study for docs/SCOPE_AGREEMENT_ARM_CLOSED_LOOP_2026-10-02.md ("What would be built", item 1). CPU only.

The replay. The base tracker and the ``strict`` (D_strict + J) and ``agreement`` arms of
``perception/depth_appearance.py`` run in lockstep on the five shadowed evening 14944 recordings, which all stop
at frame 67 (0-based ``frames.json`` index, the first frame whose recorded cut phase is ``stopped``). Every input
is fed exactly as ``tools/replay_depth_appearance.py`` feeds it (imported, never modified): the tracker config
from frames[0], the seed initialization on the preview, the preview update with the camera pose rebuilt from the
initial tool pose and the open-jaw boxes, then every frame with its recorded wrist pose and depth, and jaw boxes
from the recorded tool pose, attachment roll, rendered closure progress and scene radius. The only difference from
``replay_run`` is the last frame: it stops at the recorded stop, this tool goes on to frame 72 with each tracker's
own state. Nothing from the recording is injected after the stop; the base tracker and the strict arm latch their
own loss.

Fidelity, checked before any number past the stop is used:
  * every report.json / frames.json against the sha256 the committed evidence recorded;
  * the seed initialization, the preview update and frames 0..66 of every tracker against the recording, on every
    field: state, reason and pixel_xy exactly, every other number within 1e-6 (largest differences reported);
  * frame 67: the tracker that ran live (the base tracker, or the strict arm in the two depth closed-loop runs)
    against the recorded measurement; the agreement arm's depth event against the strict arm's (same inputs,
    arm-dependent keys left out), against the recorded live event (depth closed loop) or against the committed
    replay evidence (the other three);
  * ``replay_depth_appearance.replay_run`` itself, run on the same inputs: its results through the stop must equal
    this tool's.

The pose comparison. Frames 60-77 of the five shadowed recordings against the two shadow-removed runs
(jaw-shadow-eve-b r1/r2): tool and wrist-camera position (mm) and rotation (deg) differences, the no-shadow r1-r2
floor, cut phase, rendered jaw closure progress, the recorded servo decisions and the frame-to-frame motion. The
recorded wrist rotation matrices are orthonormal only to ~4e-7, which puts a 0.02-0.04 deg noise floor under a
plain arccos angle, so each is replaced by its closest rotation and the angle is taken with atan2.

What frames 68-72 can and cannot show. They were rendered for a robot that had stopped: it held still with the
jaw open, as a continuing run holds for alignment over the same frames, so the replay past the stop approximates a
continuing run only as closely as the pose comparison shows (about 1 mm and 0.03 deg). The controller's decisions
with the agreement arm are not replayed: the cut-gate figures are arithmetic on the replayed targets and the
recorded tool poses against the thresholds of the controller as the runner builds it. Frames 73-77 (closure) are
not observable: no shadowed recording renders a moving jaw. Offline replay of recorded simulator images (simulator
depth: noise-free, exact camera motion); no sensor noise, pose error or time-sync error is modelled. Nothing here
is a live outcome or a grade; only tools/validate_vision_sequence.py grades live runs. Inputs are read only; the
output must be a new file outside the repository (and artifacts/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "source/isaaclab_pruning"))
import replay_depth_appearance as rda  # the committed offline replay, imported and never modified
from replay_jaw_self_mask import build_demo  # the controller as the runner builds it (for its cut thresholds)

from isaaclab_pruning.perception.jaw_self_mask import proxy_roll_rad, rotation_angle_deg
from isaaclab_pruning.perception.visual_servo import VisualServoTracker
from isaaclab_pruning.task.simulated_cut import tool_mouth_geometry

SCHEMA_VERSION = 1
VISION_ROBUSTNESS = ROOT / "artifacts/vision_robustness"
SCOPE_DOCUMENT = "docs/SCOPE_AGREEMENT_ARM_CLOSED_LOOP_2026-10-02.md"
HELDOUT_EVIDENCE = "docs/evidence/depth_heldout_replay_2026-10-01.json"
REGRESSION_EVIDENCE = "docs/evidence/depth_regression_replay_2026-10-01.json"
DEPTH_LOOP_VERDICTS = "docs/evidence/depth_loop_verdicts_2026-10-02.json"
EVIDENCE_FILES = (HELDOUT_EVIDENCE, REGRESSION_EVIDENCE, DEPTH_LOOP_VERDICTS)
#: The five shadowed evening 14944 recordings: which tracker ran live, and the committed replay that covers them.
SHADOWED = {
    "perception_round": {
        "run": "tree1-listed-evening-20260926/run_01_evening_tree1_v14944_baseline",
        "live_tracker": "base",
        "committed_replay": REGRESSION_EVIDENCE,
    },
    "jaw_shadow_a_r1": {
        "run": "jaw-shadow-eve-a-r1-20260930/run_00_evening_tree1_v14944_baseline",
        "live_tracker": "base",
        "committed_replay": HELDOUT_EVIDENCE,
    },
    "jaw_shadow_a_r2": {
        "run": "jaw-shadow-eve-a-r2-20260930/run_00_evening_tree1_v14944_baseline",
        "live_tracker": "base",
        "committed_replay": HELDOUT_EVIDENCE,
    },
    "depth_loop_r1": {
        "run": "depth-loop-eve-r1-20261001/run_00_evening_tree1_v14944_baseline_depth_appearance",
        "live_tracker": "strict",
        "committed_replay": None,
    },
    "depth_loop_r2": {
        "run": "depth-loop-eve-r2-20261001/run_00_evening_tree1_v14944_baseline_depth_appearance",
        "live_tracker": "strict",
        "committed_replay": None,
    },
}
#: The two shadow-removed runs of the same target and light (they pass), and the committed replay that covers them.
NO_SHADOW = {
    "no_shadow_b_r1": {
        "run": "jaw-shadow-eve-b-r1-20260930/run_00_evening_tree1_v14944_jaw_no_shadow",
        "committed_replay": HELDOUT_EVIDENCE,
    },
    "no_shadow_b_r2": {
        "run": "jaw-shadow-eve-b-r2-20260930/run_00_evening_tree1_v14944_jaw_no_shadow",
        "committed_replay": HELDOUT_EVIDENCE,
    },
}
#: The arms replayed (fixed here, so that an arm added to the module later does not change this study).
ARMS = ("strict", "agreement")
TRACKERS = ("base", *ARMS)
EXPECTED_STOP = 67
LAST_FRAME = 72
TABLE_FIRST = 60
POSE_FIRST, POSE_LAST = 60, 77
POSE_KEEP_FROM = 50
FLOAT_TOLERANCE = 1e-6
EXACT_FIELDS = ("state", "reason", "pixel_xy")
#: Event keys that depend on the arm (its rule and its decision); everything else is the shared depth test.
ARM_DEPENDENT_EVENT_KEYS = frozenset(
    {"arm", "decision", "appearance_gate_value", "ended_by", "result_state", "result_reason", "reached_confidence_gate"}
)
#: Keys ``replay_run`` adds to an event record around the event itself.
RECORD_EXTRAS = frozenset({"frame", "after_recorded_detach", "recorded", "recorded_cut_phase", "replayed"})
MAX_LISTED_DIFFERENCES = 25
POSE_METRICS = ("tool_position_mm", "tool_rotation_deg", "wrist_position_mm", "wrist_rotation_deg")
#: The scope document's uncommitted pointer, checked here: NCC 0.82-0.88 at frame 68 and 0.98 or above through 72.
POINTER_NCC_FLOOR = 0.98
SCOPE = (
    "Design study for docs/SCOPE_AGREEMENT_ARM_CLOSED_LOOP_2026-10-02.md (What would be built, item 1). The base "
    "tracker and the strict and agreement arms of the depth-aware appearance check are replayed in lockstep on the "
    "five shadowed evening 14944 recordings from the seed initialization through frame 72, past the recorded stop at "
    "frame 67, each with its own state; nothing from the recording is injected after the stop. The tool and "
    "wrist-camera poses of those recordings are compared over frames 60-77 with the two shadow-removed runs "
    "(jaw-shadow-eve-b r1/r2). Frames 68-72 were rendered for a robot that had stopped and held still with the jaw "
    "open, as a continuing run holds for alignment over the same frames: they approximate a continuing run only as "
    "closely as the pose comparison shows, and the controller's own decisions with the agreement arm are not "
    "replayed (the cut-gate figures are arithmetic on the replayed targets and the recorded tool poses). Frames "
    "73-77 (closure) are not observable: no shadowed recording renders a moving jaw. Offline replay of recorded "
    "simulator images with simulator depth; no sensor noise, pose error or time-sync error is modelled; not a live "
    "outcome or a grade (only tools/validate_vision_sequence.py grades live runs)."
)
FIDELITY_CRITERION = (
    "Inputs equal the sha256 the committed evidence recorded. Seed initialization, preview update and frames 0..66 "
    "of every tracker equal the recording on every field: state, reason and pixel_xy exactly, every other number "
    "within 1e-6. At frame 67 the tracker that ran live equals the recorded measurement; the agreement arm's depth "
    "event equals the strict arm's on the arm-independent keys, and the recorded live event (depth closed loop) or "
    "the committed replay evidence (the other three). replay_depth_appearance.replay_run, run on the same inputs, "
    "equals this tool through the stop."
)


# ------------------------------------------------------------------------------------------------ comparison
def normalize(value):
    """JSON-native copy exactly as the recordings store values (non-finite floats become null)."""
    return json.loads(json.dumps(rda._json_safe(value), allow_nan=False))


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def compare(replayed, recorded, tolerance=FLOAT_TOLERANCE, exact_fields=EXACT_FIELDS, ignore=()):
    """Field-by-field comparison of two JSON-native values.

    Top-level ``exact_fields`` (with everything below them), strings, bools and nulls must be equal; every other
    number within ``tolerance``. A bool never equals a number. Returns whether they are equal, the first
    differences, the keys present on one side only, and the largest absolute numeric difference per top-level key.
    Top-level keys in ``ignore`` are skipped.
    """
    out = {"differences": [], "only_in_replayed": [], "only_in_recorded": [], "max_abs_difference": {}}

    def walk(a, b, path, top, exact):
        if isinstance(a, dict) and isinstance(b, dict):
            for key in sorted(set(a) | set(b), key=str):
                if not path and key in ignore:
                    continue
                sub = f"{path}.{key}" if path else str(key)
                if key not in b:
                    out["only_in_replayed"].append(sub)
                elif key not in a:
                    out["only_in_recorded"].append(sub)
                else:
                    walk(a[key], b[key], sub, top or str(key), exact or (not path and key in exact_fields))
            return
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                out["differences"].append({"path": path, "replayed": f"list[{len(a)}]", "recorded": f"list[{len(b)}]"})
                return
            for i, (x, y) in enumerate(zip(a, b)):
                walk(x, y, f"{path}[{i}]", top, exact)
            return
        if _is_number(a) and _is_number(b):
            difference = abs(float(a) - float(b))
            out["max_abs_difference"][top] = max(out["max_abs_difference"].get(top, 0.0), difference)
            if (exact and difference != 0.0) or difference > tolerance:
                out["differences"].append({"path": path, "replayed": a, "recorded": b, "abs_difference": difference})
            return
        if type(a) is not type(b) or a != b:
            out["differences"].append({"path": path, "replayed": a, "recorded": b})

    walk(replayed, recorded, "", "", False)
    out["equal"] = not (out["differences"] or out["only_in_replayed"] or out["only_in_recorded"])
    out["difference_count"] = len(out["differences"])
    out["differences"] = out["differences"][:MAX_LISTED_DIFFERENCES]
    return out


def event_core(event):
    """The arm-independent part of a raw event or an event record: the depth test, J and both confidences."""
    return {key: value for key, value in event.items() if key not in ARM_DEPENDENT_EVENT_KEYS | RECORD_EXTRAS}


def merge_max(target, source):
    """Fold per-key maxima from ``source`` into ``target``."""
    for key, value in source.items():
        target[key] = max(target.get(key, 0.0), value)
    return target


def target_shift_mm(a, b):
    """Distance in mm between two world points (None if either is missing)."""
    if a is None or b is None:
        return None
    return float(1e3 * np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)))


# ------------------------------------------------------------------------------------------------ rotations
def closest_rotation(matrix):
    """The proper rotation nearest a (nearly orthonormal) 3 x 3 matrix, by SVD."""
    u, _, vt = np.linalg.svd(np.asarray(matrix, float))
    rotation = u @ vt
    if np.linalg.det(rotation) < 0:
        u[:, -1] *= -1
        rotation = u @ vt
    return rotation


def rotation_matrix_angle_deg(a, b):
    """Relative rotation angle of two recorded rotation matrices, well conditioned at small angles.

    Each matrix is replaced by its closest rotation, and the angle is atan2(|vee(M - M^T)| / 2, (trace(M) - 1) / 2)
    of M = Ra^T Rb; a plain arccos((trace - 1) / 2) of matrices orthonormal only to ~4e-7 has a 0.02-0.04 deg floor.
    """
    m = closest_rotation(a).T @ closest_rotation(b)
    sine = np.linalg.norm([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0], m[1, 0] - m[0, 1]]) / 2.0
    return float(np.degrees(np.arctan2(sine, (np.trace(m) - 1.0) / 2.0)))


def quaternion_angle_deg(q0_wxyz, q1_wxyz):
    """Relative rotation angle of two quaternions: 2 atan2(|v|, |w|) of the unit quaternion q0* q1."""
    a = np.asarray(q0_wxyz, float) / np.linalg.norm(q0_wxyz)
    b = np.asarray(q1_wxyz, float) / np.linalg.norm(q1_wxyz)
    w = a[0] * b[0] + a[1:] @ b[1:]
    v = a[0] * b[1:] - b[0] * a[1:] - np.cross(a[1:], b[1:])
    return float(np.degrees(2.0 * np.arctan2(np.linalg.norm(v), abs(w))))


def orthonormality_error(matrix):
    """Largest absolute entry of R^T R - I."""
    matrix = np.asarray(matrix, float)
    return float(np.abs(matrix.T @ matrix - np.eye(3)).max())


def pose_difference(a, b):
    """Tool (xyz + wxyz) and wrist-camera position (mm) and rotation (deg) differences of two recorded frames."""
    pa, pb = np.asarray(a["tool_pose_wxyz"], float), np.asarray(b["tool_pose_wxyz"], float)
    wrist_a, wrist_b = np.asarray(a["wrist_position_w_m"], float), np.asarray(b["wrist_position_w_m"], float)
    return {
        "tool_position_mm": 1e3 * float(np.linalg.norm(pa[:3] - pb[:3])),
        "tool_rotation_deg": rotation_angle_deg(pa[3:], pb[3:]),
        "tool_rotation_deg_quaternion_atan2": quaternion_angle_deg(pa[3:], pb[3:]),
        "wrist_position_mm": 1e3 * float(np.linalg.norm(wrist_a - wrist_b)),
        "wrist_rotation_deg": rotation_matrix_angle_deg(a["wrist_rotation_w_ros"], b["wrist_rotation_w_ros"]),
    }


def largest(pairs):
    """Per pose metric: the largest value over ``{pair_name: pose_difference}`` and the pair that has it."""
    out = {}
    for metric in POSE_METRICS:
        name = max(pairs, key=lambda key: pairs[key][metric])
        out[metric] = {"value": pairs[name][metric], "pair": name}
    return out


# ------------------------------------------------------------------------------------------------ inputs
def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def output_refusal(path):
    """Why the output path is not allowed (inside the repository or artifacts/, existing, no directory), else None."""
    return rda.output_refusal(path)


def frame_files_digest(path, last_frame=LAST_FRAME):
    """sha256 over the per-file sha256 of every image and depth file the replay reads, in reading order."""
    names = ["preview_wrist.png", "preview_depth.npy"]
    for index in range(last_frame + 1):
        names += [f"frames/wrist_{index:05d}.png", f"frames/depth_{index:05d}.npy"]
    digest = hashlib.sha256()
    for name in names:
        digest.update(f"{name}:{sha256(Path(path) / name)}\n".encode())
    return {"files": len(names), "sha256_of_file_hashes": digest.hexdigest()}


def committed_hashes(spec, evidence):
    """The report.json / frames.json sha256 the committed evidence recorded for one run."""
    if spec.get("committed_replay"):
        entry = next(r for r in evidence[spec["committed_replay"]]["runs"] if r["run"] == spec["run"])
        return {
            "source": spec["committed_replay"],
            "report.json": entry["source_sha256"]["report.json"],
            "frames.json": entry["source_sha256"]["frames.json"],
        }
    batch, run_directory = spec["run"].split("/")
    entry = next(
        r for r in evidence[DEPTH_LOOP_VERDICTS]["runs"] if r["batch"] == batch and r["run_directory"] == run_directory
    )
    return {"source": DEPTH_LOOP_VERDICTS, "report.json": entry["report_sha256"], "frames.json": entry["frames_sha256"]}


def _git(*args):
    environment = {**os.environ, "GIT_OPTIONAL_LOCKS": "0"}  # git status must not refresh the repository index
    return subprocess.check_output(["git", "-C", str(ROOT), *args], text=True, env=environment).strip()


def provenance():
    """HEAD, whether tracked files differ from it, and every repository module this process has loaded."""
    root = ROOT.resolve()
    loaded = sorted(
        {
            str(Path(module.__file__).resolve().relative_to(root))
            for module in list(sys.modules.values())
            # A relative __file__ (some extension modules report a bare name) resolves against the working
            # directory and names no repository file; only absolute paths to existing files count.
            if getattr(module, "__file__", None)
            and Path(module.__file__).is_absolute()
            and Path(module.__file__).is_file()
            and root in Path(module.__file__).resolve().parents
        }
    )
    try:
        revision = _git("rev-parse", "HEAD")
        status = _git("status", "--porcelain", "--untracked-files=no")
    except (OSError, subprocess.CalledProcessError):
        revision, status = None, None
    return {
        "code_revision": revision,
        "code_tree_dirty": None if status is None else bool(status),
        "code_tree_status_porcelain": status,
        "loaded_repository_modules_sha256": {name: sha256(root / name) for name in loaded},
    }


# ------------------------------------------------------------------------------------------------ the replay
def _patch_stats(rgb, pixel_xy):
    """Telemetry only: grey-level mean and spread of the 13 x 13 patch at a pixel (never fed back)."""
    import cv2

    if pixel_xy is None:
        return None
    gray = cv2.cvtColor(np.ascontiguousarray(rgb[:, :, :3]), cv2.COLOR_RGB2GRAY).astype(np.float32)
    patch = cv2.getRectSubPix(gray, (13, 13), tuple(float(v) for v in pixel_xy))
    return {"mean": float(patch.mean()), "std": float(patch.std())}


def _frame_geometry(frame, closing_axis, branch_axis):
    """The recorded frame's mouth point and the cut gate's perpendicularity, as the controller computes them."""
    live = frame["live_vision"]
    mouth, closing_axis_w = tool_mouth_geometry(
        frame["tool_pose_wxyz"],
        tuple(live.get("mouth_offset_tool_m") or (0.0, 0.0, 0.07)),
        tuple(live.get("closing_axis_tool") or closing_axis),
    )
    unit = np.asarray(closing_axis_w) / np.linalg.norm(closing_axis_w)
    perpendicularity = math.degrees(math.asin(float(np.clip(abs(np.dot(branch_axis, unit)), 0, 1))))
    return list(mouth), perpendicularity


def replay_through(path, last_frame=LAST_FRAME):
    """``replay_depth_appearance.replay_run``'s loop, run on to ``last_frame`` instead of the recorded stop."""
    path = Path(path)
    report = json.loads((path / "report.json").read_text())
    if report.get("stage") != "complete":
        raise ValueError(f"{path}: the recording is not complete (stage {report.get('stage')!r})")
    if any((report.get(arm) or {}).get("enabled") for arm in ("jaw_self_mask", "closure_hold")):
        raise ValueError(f"{path}: recorded with the jaw self-mask or closure hold, which this replay does not model")
    document = json.loads((path / "frames.json").read_text())
    frames, fps = document["frames"], float(document["fps"])
    del document
    if [frame["index"] for frame in frames] != list(range(len(frames))):
        raise ValueError(f"{path}: frame indexes are not 0..N-1")
    camera_matrix = np.asarray(report["camera"]["wrist_intrinsics"], dtype=float)
    config = rda.tracker_config(frames)
    trackers = {"base": VisualServoTracker(config)}
    trackers.update({arm: rda.DepthAppearanceTracker(config, arm=arm) for arm in ARMS})
    initial_live = report.get("initial_live_vision") or {}
    closing_axis = initial_live.get("closing_axis_tool") or frames[0]["live_vision"]["closing_axis_tool"]
    roll = proxy_roll_rad(closing_axis)
    radius = float(report["blender_scene"]["target"]["radius_m"])
    axis = np.asarray(report["blender_scene"]["target"]["axis_w"], float)
    branch_axis = axis / np.linalg.norm(axis)
    cut_config = asdict(build_demo(report, False, False).cutter.config)  # read, never run
    recorded_stop = next((f["index"] for f in frames if f["live_vision"]["cut"]["phase"] == "stopped"), None)
    detach = next((f["index"] for f in frames if f["live_vision"]["cut"].get("detach_event")), None)
    if recorded_stop is None or recorded_stop >= last_frame:
        raise ValueError(f"{path}: no recorded stop before frame {last_frame}")

    recorded_preview = initial_live.get("measurement") or {}
    preview_rgb = rda._rgb(path / "preview_wrist.png")
    preview_depth = np.load(path / "preview_depth.npy", allow_pickle=False)
    preview_pose = rda.preview_transform(report)
    recorded_init = (report.get("vision_initialization") or {}).get("tracker") or {}
    rejected = recorded_init.get("state") == "initialization_rejected"
    seed = report["vision_initialization"]["pixel_xy"]
    init = {name: None if rejected else t.initialize(preview_rgb, seed, preview_depth) for name, t in trackers.items()}
    preview_boxes = rda.frame_jaw_boxes(report["initial_tool_pose_wxyz"][0], roll, 0.0, radius)
    preview = {
        name: tracker.update(preview_rgb, preview_depth, camera_matrix, preview_pose)
        if name == "base"
        else tracker.update(preview_rgb, preview_depth, camera_matrix, preview_pose, jaw_boxes=preview_boxes)
        for name, tracker in trackers.items()
    }
    preview_events = {
        arm: None if trackers[arm].last_event is None else normalize(trackers[arm].last_event) for arm in ARMS
    }

    results = {name: [] for name in trackers}
    patch_stats = {name: [] for name in trackers}
    raw_events = {arm: {} for arm in ARMS}
    records = {arm: {} for arm in ARMS}
    frame_info = []
    for frame in frames[: last_frame + 1]:
        index = frame["index"]
        rgb = rda._rgb(path / f"frames/wrist_{index:05d}.png")
        depth = np.load(path / f"frames/depth_{index:05d}.npy", allow_pickle=False)
        pose = rda._transform(frame["wrist_position_w_m"], frame["wrist_rotation_w_ros"])
        boxes = rda.frame_jaw_boxes(
            frame["tool_pose_wxyz"], roll, frame.get("visual_jaw_closure_progress") or 0.0, radius
        )
        recorded = frame["live_vision"]["measurement"]
        for name, tracker in trackers.items():
            if name == "base":
                result = tracker.update(rgb, depth, camera_matrix, pose)
            else:
                result = tracker.update(rgb, depth, camera_matrix, pose, jaw_boxes=boxes)
                if tracker.last_event is not None:
                    raw_events[name][index] = normalize(tracker.last_event)
                    record = rda._event_record(index, tracker.last_event)
                    record["after_recorded_detach"] = detach is not None and index > detach
                    record["recorded"] = rda._brief(recorded)
                    record["recorded_cut_phase"] = frame["live_vision"]["cut"]["phase"]
                    record["replayed"] = rda._brief(result)
                    records[name][index] = normalize(record)
            results[name].append(normalize(result))
            patch_stats[name].append(_patch_stats(rgb, result.get("pixel_xy")))
        certificate = (frame["live_vision"].get("cut") or {}).get("certificate") or {}
        mouth, perpendicularity = _frame_geometry(frame, closing_axis, branch_axis)
        frame_info.append(
            {
                "index": index,
                "recorded_measurement": recorded,
                "recorded_cut_phase": frame["live_vision"]["cut"]["phase"],
                "recorded_command_phase": frame.get("phase"),
                "recorded_mouth_distance_m": certificate.get("mouth_distance_m"),
                "recorded_perpendicularity_error_deg": certificate.get("perpendicularity_error_deg"),
                "mouth_position_w_m": mouth,
                "perpendicularity_error_deg": perpendicularity,
                "visual_jaw_closure_progress": frame.get("visual_jaw_closure_progress"),
                "jaw_boxes_known": boxes is not None,
            }
        )
    del frames
    for previous, info in zip(frame_info, frame_info[1:]):
        info["mouth_step_mm"] = target_shift_mm(info["mouth_position_w_m"], previous["mouth_position_w_m"])
    frame_info[0]["mouth_step_mm"] = None
    return {
        "recorded_stop": recorded_stop,
        "detach": detach,
        "recorded_init": recorded_init,
        "recorded_preview": recorded_preview,
        "init": {name: None if value is None else normalize(value) for name, value in init.items()},
        "preview": {name: normalize(value) for name, value in preview.items()},
        "preview_events": preview_events,
        "results": results,
        "patch_stats": patch_stats,
        "raw_events": raw_events,
        "records": records,
        "frame_info": frame_info,
        "cut_config": cut_config,
        "fps": fps,
        "min_patch_correlation": config.min_patch_correlation,
        "min_confidence": config.min_confidence,
    }


# ------------------------------------------------------------------------------------------------ fidelity
def _tracker_fidelity(name, replay, live_tracker):
    stop = replay["recorded_stop"]
    init = replay["init"][name]
    item = {
        "initialization": None if init is None else compare(init, replay["recorded_init"]),
        "preview": compare(replay["preview"][name], replay["recorded_preview"]),
    }
    mismatches, max_abs, committed_criterion = [], {}, []
    for index in range(stop):
        result = replay["results"][name][index]
        recorded = replay["frame_info"][index]["recorded_measurement"]
        check = compare(result, recorded)
        merge_max(max_abs, check["max_abs_difference"])
        if not check["equal"]:
            keys = ("differences", "only_in_replayed", "only_in_recorded", "difference_count")
            mismatches.append({"frame": index, **{key: check[key] for key in keys}})
        fields = rda.same_measurement(result, recorded)
        if fields:
            committed_criterion.append({"frame": index, "fields": fields})
    item["frames_before_stop"] = {
        "frames_compared": stop,
        "all_fields_equal": not mismatches,
        "mismatching_frames": mismatches,
        "max_abs_difference_per_field": max_abs,
        "committed_criterion_mismatches": committed_criterion,
    }
    item["frame_at_stop_vs_recorded"] = compare(
        replay["results"][name][stop], replay["frame_info"][stop]["recorded_measurement"]
    )
    item["frame_at_stop_vs_recorded"]["expected_equal"] = name == live_tracker
    item["reproduces_before_stop"] = (
        (item["initialization"] is None or item["initialization"]["equal"])
        and item["preview"]["equal"]
        and not mismatches
    )
    return item


def _event_fidelity(spec, replay, evidence):
    stop = replay["recorded_stop"]
    agreement_event = replay["raw_events"]["agreement"].get(stop)
    strict_event = replay["raw_events"]["strict"].get(stop)
    events = {
        "agreement_event_at_stop_exists": agreement_event is not None,
        "strict_event_at_stop_exists": strict_event is not None,
    }
    if agreement_event is not None and strict_event is not None:
        events["agreement_vs_strict_replayed_core"] = compare(
            event_core(agreement_event), event_core(strict_event), tolerance=0.0
        )
    if spec["live_tracker"] == "strict":
        live_event = (replay["frame_info"][stop]["recorded_measurement"] or {}).get("depth_appearance")
        events["recorded_live_event_exists"] = live_event is not None
        if live_event is not None and agreement_event is not None and strict_event is not None:
            events["agreement_vs_recorded_live_core"] = compare(event_core(agreement_event), event_core(live_event))
            events["strict_vs_recorded_live_full"] = compare(strict_event, live_event)
    if spec.get("committed_replay"):
        document = evidence[spec["committed_replay"]]
        entry = next(r for r in document["runs"] if r["run"] == spec["run"])
        committed = {"evidence_code_revision": document.get("code_revision")}
        for arm in ARMS:
            evidence_records = {r["frame"]: r for r in entry["events"][arm]}
            mine = replay["records"][arm]
            committed[arm] = {
                "event_frames_evidence": sorted(evidence_records, key=lambda f: -1 if f == "preview" else f),
                "event_frames_replayed_through_stop": sorted(f for f in mine if f <= stop),
                "event_at_stop_vs_evidence": None
                if stop not in evidence_records or stop not in mine
                else compare(mine[stop], evidence_records[stop]),
            }
        committed["measurement_at_stop_vs_evidence"] = {
            name: compare(
                normalize(rda._brief(replay["results"][name][stop])), entry["measurement_at_recorded_stop"][name]
            )
            for name in TRACKERS
        }
        events["vs_committed_evidence"] = committed
    return events


def _cross_check(replay, committed_run):
    """This tool's results through the stop against ``replay_depth_appearance.replay_run`` on the same inputs."""
    if "error" in committed_run:
        return {"error": committed_run["error"], "this_tool_equals_committed_replay_run": False}
    stop = replay["recorded_stop"]
    cross = {
        "replayed_through_frame": committed_run["replayed_through_frame"],
        "base_reproduces_recording": committed_run["base_reproduces_recording"],
        "grey_failures": committed_run["grey_failures"],
        "mismatch_frames": {name: committed_run["divergence"][name]["mismatch_frames"] for name in TRACKERS},
        "measurement_at_stop_equal": {
            name: compare(
                normalize(rda._brief(replay["results"][name][stop])),
                normalize(committed_run["measurement_at_recorded_stop"][name]),
                tolerance=0.0,
            )["equal"]
            for name in TRACKERS
        },
        "events_through_stop_equal": {},
    }
    for arm in ARMS:
        theirs = [r for r in normalize(committed_run["events"][arm]) if r["frame"] != "preview"]
        mine = [replay["records"][arm][f] for f in sorted(replay["records"][arm]) if f <= stop]
        cross["events_through_stop_equal"][arm] = len(mine) == len(theirs) and all(
            compare(a, b, tolerance=0.0)["equal"] for a, b in zip(mine, theirs)
        )
    cross["this_tool_equals_committed_replay_run"] = all(cross["measurement_at_stop_equal"].values()) and all(
        cross["events_through_stop_equal"].values()
    )
    return cross


def fidelity(spec, replay, committed_run, evidence):
    """Every fidelity check of one run (see the module docstring); ``passed`` is their conjunction."""
    stop = replay["recorded_stop"]
    per_tracker = {name: _tracker_fidelity(name, replay, spec["live_tracker"]) for name in TRACKERS}
    out = {
        "recorded_stop_frame": stop,
        "stop_is_expected_frame": stop == EXPECTED_STOP,
        "live_tracker": spec["live_tracker"],
        "per_tracker": per_tracker,
        "live_tracker_reproduces_frame_at_stop": per_tracker[spec["live_tracker"]]["frame_at_stop_vs_recorded"][
            "equal"
        ],
        "event_at_stop": _event_fidelity(spec, replay, evidence),
        "committed_replay_run_cross_check": _cross_check(replay, committed_run),
    }
    out["checks"] = fidelity_checks(out)
    out["passed"] = all(out["checks"].values())
    return out


def fidelity_checks(result):
    """The named pass/fail checks of one run's fidelity record."""
    events = result["event_at_stop"]
    checks = {
        "stop_is_expected_frame": result["stop_is_expected_frame"],
        "all_trackers_reproduce_before_stop": all(
            item["reproduces_before_stop"] for item in result["per_tracker"].values()
        ),
        "live_tracker_reproduces_frame_at_stop": result["live_tracker_reproduces_frame_at_stop"],
        "agreement_event_equals_strict_event": (events.get("agreement_vs_strict_replayed_core") or {}).get("equal")
        is True,
        "equals_committed_replay_run": result["committed_replay_run_cross_check"].get(
            "this_tool_equals_committed_replay_run"
        )
        is True,
    }
    if result["live_tracker"] == "strict":
        checks["agreement_event_equals_recorded_live_event"] = (
            events.get("agreement_vs_recorded_live_core") or {}
        ).get("equal") is True
        checks["strict_event_equals_recorded_live_event"] = (events.get("strict_vs_recorded_live_full") or {}).get(
            "equal"
        ) is True
    if "vs_committed_evidence" in events:
        committed = events["vs_committed_evidence"]
        checks["events_equal_committed_evidence"] = all(
            (committed[arm]["event_at_stop_vs_evidence"] or {}).get("equal") is True for arm in ARMS
        )
        checks["measurements_equal_committed_evidence"] = all(
            item["equal"] for item in committed["measurement_at_stop_vs_evidence"].values()
        )
    return checks


# ------------------------------------------------------------------------------------------------ tables
def event_summary(event):
    """The depth event's decision inputs: acceptance, failed conditions, J, the statistics and both confidences."""
    stats = event.get("stats") or {}
    jaw = event.get("jaw") or {}
    summary = {
        "accept": event.get("accept"),
        "ncc": event.get("ncc"),
        "failed_conditions": event.get("failed_conditions"),
        "not_evaluated_conditions": event.get("not_evaluated_conditions"),
        "reasons": event.get("reasons"),
        "decision": event.get("decision"),
        "appearance_gate_value": event.get("appearance_gate_value"),
        "feature_ratio": event.get("feature_ratio"),
        "depth_valid_fraction": event.get("depth_valid_fraction"),
        "strict_confidence": event.get("strict_confidence"),
        "agreement_confidence": event.get("agreement_confidence"),
        "jaw_touches": jaw.get("touches"),
        "jaw_reason": jaw.get("reason"),
        "jaw_masked_previous_reprojected_propagated": [
            jaw.get("previous_patch_masked"),
            jaw.get("reprojected_elements_masked"),
            jaw.get("propagated_patch_masked"),
        ],
        "jaw_propagated_pixel_signed_distance_px": jaw.get("propagated_pixel_signed_distance_px"),
    }
    for key in (
        "tracked_depth_prev_m",
        "n_same_surface",
        "n_verified",
        "verified_fraction",
        "median_abs_m",
        "near_fraction",
        "agreement_fraction",
        "pixel_disagreement_px",
    ):
        summary[key] = stats.get(key)
    return summary


def grey_check(ncc, gate):
    """``passed`` / ``failed`` against the 0.35 gate, or ``not_reached`` when the update ended before it."""
    if ncc is None:
        return "not_reached"
    return "passed" if ncc >= gate else "failed"


def table(name, replay, first=TABLE_FIRST, last=LAST_FRAME):
    """One tracker's per-frame record, frames ``first..last``."""
    rows = []
    stop = replay["recorded_stop"]
    target66 = replay["results"][name][EXPECTED_STOP - 1].get("target_position_world_m")
    for index in range(first, last + 1):
        result = replay["results"][name][index]
        info = replay["frame_info"][index]
        recorded = info["recorded_measurement"] or {}
        target = result.get("target_position_world_m")
        ncc = result.get("patch_correlation")
        grey = grey_check(ncc, replay["min_patch_correlation"])
        event = None if name == "base" else replay["raw_events"][name].get(index)
        row = {
            "frame": index,
            "after_recorded_stop": index > stop,
            "state": result.get("state"),
            "reason": result.get("reason"),
            "feature_count": result.get("feature_count"),
            "roundtrip_inlier_count": result.get("roundtrip_inlier_count"),
            "replenished_for_next_frame": None
            if "pending_feature_pixels_for_next_frame" not in result
            else len(result["pending_feature_pixels_for_next_frame"]),
            "initial_feature_count": result.get("initial_feature_count"),
            "median_roundtrip_error_px": result.get("median_roundtrip_error_px"),
            "flow_reason": result.get("flow_reason"),
            "patch_correlation": ncc,
            "grey_check": grey,
            "confidence": result.get("confidence"),
            "measured_confidence": result.get("measured_confidence"),
            "appearance_gate_value": result.get("appearance_gate_value"),
            "depth_valid_fraction": result.get("depth_valid_fraction"),
            "depth_m": result.get("depth_m"),
            "depth_reason": result.get("depth_reason"),
            "world_jump_m": result.get("world_jump_m"),
            "pixel_xy": result.get("pixel_xy"),
            "target_position_world_m": target,
            "target_shift_from_frame66_mm": target_shift_mm(target, target66),
            "target_shift_from_previous_frame_mm": target_shift_mm(
                target, replay["results"][name][index - 1].get("target_position_world_m")
            ),
            "mouth_distance_mm": target_shift_mm(target, info["mouth_position_w_m"]),
            "mouth_step_mm": info["mouth_step_mm"],
            "perpendicularity_error_deg": info["perpendicularity_error_deg"],
            "patch_gray_mean_std": replay["patch_stats"][name][index],
            "depth_event": None if event is None else event_summary(event),
            "jaw_boxes_known": info["jaw_boxes_known"],
            "visual_jaw_closure_progress": info["visual_jaw_closure_progress"],
            "recorded": {
                "state": recorded.get("state"),
                "reason": recorded.get("reason"),
                "patch_correlation": recorded.get("patch_correlation"),
                "cut_phase": info["recorded_cut_phase"],
                "command_phase": info["recorded_command_phase"],
            },
        }
        if name == "base":
            row["depth_event_note"] = "base tracker: no depth test"
        elif event is None:
            row["depth_event_note"] = (
                "no grey failure: NCC >= 0.35, the depth test did not run"
                if grey == "passed"
                else "grey check not reached: no depth test"
            )
        rows.append(row)
    return rows


def arithmetic_check(replay):
    """The mouth-distance and perpendicularity arithmetic against the recorded certificate, frames before the stop."""
    worst_distance = worst_perpendicularity = 0.0
    compared = 0
    for index in range(replay["recorded_stop"]):
        info = replay["frame_info"][index]
        recorded = info["recorded_mouth_distance_m"]
        target = (info["recorded_measurement"] or {}).get("target_position_world_m")
        if recorded is None or target is None:
            continue
        mine = float(np.linalg.norm(np.asarray(target, float) - np.asarray(info["mouth_position_w_m"], float)))
        worst_distance = max(worst_distance, abs(mine - recorded))
        if info["recorded_perpendicularity_error_deg"] is not None:
            worst_perpendicularity = max(
                worst_perpendicularity,
                abs(info["perpendicularity_error_deg"] - info["recorded_perpendicularity_error_deg"]),
            )
        compared += 1
    return {
        "frames_compared": compared,
        "max_abs_mouth_distance_difference_m": worst_distance,
        "max_abs_perpendicularity_difference_deg": worst_perpendicularity,
    }


def cut_gate_arithmetic(cut_config, fps, rows, frames):
    """Arithmetic only, not a controller replay: one arm's targets against the cut gate's thresholds.

    Per frame, with the recorded tool pose: tracking, mouth distance within the mouth tolerance, perpendicularity
    within the alignment tolerance, and the stability inputs (mouth speed and frame-to-frame target shift).
    """
    thresholds = {
        "mouth_position_tolerance_mm": 1e3 * cut_config["mouth_position_tolerance_m"],
        "alignment_tolerance_deg": cut_config["alignment_tolerance_deg"],
        "max_stable_speed_mm_s": 1e3 * cut_config["max_stable_speed_m_s"],
        "max_target_shift_mm": 1e3 * cut_config["max_target_shift_m"],
        "stable_frames": cut_config["stable_frames"],
        "closure_duration_s": cut_config["closure_duration_s"],
    }
    out = {}
    for i in frames:
        row = rows[i]
        speed = None if row["mouth_step_mm"] is None else row["mouth_step_mm"] * fps
        shift = row["target_shift_from_previous_frame_mm"]
        distance = row["mouth_distance_mm"]
        out[str(i)] = {
            "tracking": row["state"] == "tracking",
            "mouth_distance_mm": distance,
            "within_mouth_tolerance": distance is not None and distance <= thresholds["mouth_position_tolerance_mm"],
            "perpendicularity_error_deg": row["perpendicularity_error_deg"],
            "within_alignment_tolerance": row["perpendicularity_error_deg"] <= thresholds["alignment_tolerance_deg"],
            "mouth_speed_mm_s": speed,
            "below_stable_speed": speed is not None and speed <= thresholds["max_stable_speed_mm_s"],
            "target_shift_from_previous_frame_mm": shift,
            "within_target_shift": shift is not None and shift <= thresholds["max_target_shift_mm"],
        }
    return {"thresholds": thresholds, "frames": out}


def run_summary(replay, tables):
    """Frames from the stop to 72: the agreement arm's outcome and the strict arm's latch."""
    stop = replay["recorded_stop"]
    agreement = {row["frame"]: row for row in tables["agreement"]}
    strict = {row["frame"]: row for row in tables["strict"]}
    after = range(stop, LAST_FRAME + 1)
    later = range(stop + 1, LAST_FRAME + 1)

    def series(key):
        return {str(i): agreement[i][key] for i in after}

    return {
        "agreement": {
            "states": {str(i): [agreement[i]["state"], agreement[i]["reason"]] for i in after},
            "tracks_every_frame_through_72": all(agreement[i]["state"] == "tracking" for i in after),
            "ncc": series("patch_correlation"),
            "confidence": series("confidence"),
            "feature_count": series("feature_count"),
            "roundtrip_inliers": series("roundtrip_inlier_count"),
            "replenished_for_next_frame": series("replenished_for_next_frame"),
            "grey_failures_after_stop": [i for i in later if agreement[i]["grey_check"] == "failed"],
            "depth_events": {str(i): agreement[i]["depth_event"] for i in after if agreement[i]["depth_event"]},
            "target_shift_from_frame66_mm": series("target_shift_from_frame66_mm"),
            "target_shift_from_previous_frame_mm": series("target_shift_from_previous_frame_mm"),
            "mouth_distance_mm": series("mouth_distance_mm"),
            "cut_gate_arithmetic": cut_gate_arithmetic(replay["cut_config"], replay["fps"], agreement, after),
        },
        "strict": {
            "states": {str(i): [strict[i]["state"], strict[i]["reason"]] for i in after},
            "measured_confidence_at_stop": strict[stop]["measured_confidence"],
            "ncc_at_stop": strict[stop]["patch_correlation"],
            "latched_after_stop": all(
                strict[i]["state"] == "tracking_lost" and strict[i]["reason"] == "explicit_initialization_required"
                for i in later
            ),
        },
    }


# ------------------------------------------------------------------------------------------------ poses
def compact_run(run_dir):
    """The per-frame fields the pose comparison needs (frames 50..77) and whole-run phase markers."""
    frames = json.loads((Path(run_dir) / "frames.json").read_text())["frames"]
    if [frame["index"] for frame in frames] != list(range(len(frames))):
        raise ValueError(f"{run_dir}: frame indexes are not 0..N-1")
    rows = {}
    for frame in frames[POSE_KEEP_FROM : POSE_LAST + 1]:
        live = frame["live_vision"]
        cut = live["cut"] or {}
        certificate = cut.get("certificate") or {}
        decision = frame.get("visual_servo_decision") or {}
        delta = decision.get("delta_world_m")
        measurement = live.get("measurement") or {}
        distance = certificate.get("mouth_distance_m")
        rows[frame["index"]] = {
            "tool_pose_wxyz": frame["tool_pose_wxyz"],
            "wrist_position_w_m": frame["wrist_position_w_m"],
            "wrist_rotation_w_ros": frame["wrist_rotation_w_ros"],
            "command_phase": frame.get("phase"),
            "cut_phase": cut.get("phase"),
            "cut_closure_progress": cut.get("closure_progress"),
            "cut_stable_count": cut.get("stable_count"),
            "cut_stopped_reason": cut.get("stopped_reason"),
            "cut_detach_event": cut.get("detach_event"),
            "visual_jaw_closure_progress": frame.get("visual_jaw_closure_progress"),
            "certificate_ready_to_close": certificate.get("ready_to_close"),
            "certificate_mouth_distance_mm": None if distance is None else 1e3 * distance,
            "decision_state": decision.get("state"),
            "decision_reason": decision.get("reason"),
            "decision_step_mm": None if delta is None else 1e3 * float(np.linalg.norm(delta)),
            "controller_source_frame_index": frame.get("controller_source_frame_index"),
            "external_stop_reason": live.get("external_stop_reason"),
            "measurement_state": measurement.get("state"),
            "measurement_reason": measurement.get("reason"),
        }

    def first(predicate):
        return next((f["index"] for f in frames if predicate(f)), None)

    markers = {
        "frames": len(frames),
        "first_stopped_cut_frame": first(lambda f: f["live_vision"]["cut"]["phase"] == "stopped"),
        "first_align_cut_frame": first(lambda f: f["live_vision"]["cut"]["phase"] == "align"),
        "first_closing_cut_frame": first(lambda f: f["live_vision"]["cut"]["phase"] == "closing"),
        "first_rendered_jaw_motion_frame": first(lambda f: (f.get("visual_jaw_closure_progress") or 0.0) > 0.0),
        "detach_frame": first(lambda f: f["live_vision"]["cut"].get("detach_event")),
        # The approach controller's own steps (decision state "tracking"), up to frame 77 (retreat comes later).
        "last_approach_step_frame_through_77": max(
            (
                f["index"]
                for f in frames[: POSE_LAST + 1]
                if (f.get("visual_servo_decision") or {}).get("state") == "tracking"
            ),
            default=None,
        ),
        "wrist_rotation_orthonormality_error_max": max(
            orthonormality_error(f["wrist_rotation_w_ros"]) for f in frames[POSE_KEEP_FROM : POSE_LAST + 1]
        ),
    }
    positions = [f["tool_pose_wxyz"][:3] for f in frames[: POSE_LAST + 1]]
    return rows, markers, positions


RUN_ROW_KEYS = (
    "command_phase",
    "cut_phase",
    "cut_closure_progress",
    "cut_stable_count",
    "cut_stopped_reason",
    "cut_detach_event",
    "visual_jaw_closure_progress",
    "certificate_ready_to_close",
    "certificate_mouth_distance_mm",
    "decision_state",
    "decision_reason",
    "decision_step_mm",
    "controller_source_frame_index",
    "external_stop_reason",
    "measurement_state",
    "measurement_reason",
)


def pose_frame(data, index):
    """One frame of the pose comparison: largest differences, the r1-r2 floor and every run's recorded fields."""
    cross = {f"{s}|{n}": pose_difference(data[s][index], data[n][index]) for s in SHADOWED for n in NO_SHADOW}
    within = {f"{a}|{b}": pose_difference(data[a][index], data[b][index]) for a, b in combinations(SHADOWED, 2)}
    runs = {}
    for label in data:
        row, previous = data[label][index], data[label][index - 1]
        p, q = np.asarray(row["tool_pose_wxyz"], float), np.asarray(previous["tool_pose_wxyz"], float)
        runs[label] = {key: row[key] for key in RUN_ROW_KEYS}
        runs[label]["tool_step_from_previous_frame_mm"] = 1e3 * float(np.linalg.norm(p[:3] - q[:3]))
        runs[label]["tool_rotation_from_previous_frame_deg"] = rotation_angle_deg(q[3:], p[3:])
    return {
        "frame": index,
        "largest_shadowed_vs_no_shadow": largest(cross),
        "largest_per_shadowed_run_vs_either_no_shadow": {
            s: largest({f"{s}|{n}": cross[f"{s}|{n}"] for n in NO_SHADOW}) for s in SHADOWED
        },
        "no_shadow_r1_vs_r2_floor": pose_difference(data["no_shadow_b_r1"][index], data["no_shadow_b_r2"][index]),
        "largest_within_shadowed": largest(within),
        "tool_rotation_method_max_discrepancy_deg": max(
            abs(item["tool_rotation_deg"] - item["tool_rotation_deg_quaternion_atan2"]) for item in cross.values()
        ),
        "runs": runs,
    }


def window_max(per_frame, first, last, key):
    """Per pose metric: the largest value over frames ``first..last``, with its frame (and pair, where one exists)."""
    rows = [row for row in per_frame if first <= row["frame"] <= last]
    out = {}
    for metric in POSE_METRICS:

        def value(row, metric=metric):
            item = row[key][metric]
            return item["value"] if isinstance(item, dict) else item

        best = max(rows, key=value)
        item = best[key][metric]
        out[metric] = {"value": value(best), "frame": best["frame"]}
        if isinstance(item, dict):
            out[metric]["pair"] = item["pair"]
    return out


def pose_claims(data, markers):
    """The scope document's statements about the approach, the alignment and the jaw, checked on recorded fields."""
    labels = list(data)
    claims = {
        "approach_ends_at_67_all_seven": {
            "definition": (
                "the last frame (<= 77) whose recorded visual_servo_decision is an approach step (state 'tracking', "
                "decided from the previous frame's measurement and applied before this frame's capture) is 67, and "
                "every decision at 68-77 is a hold"
            ),
            "last_approach_step_frame": {
                label: markers[label]["last_approach_step_frame_through_77"] for label in labels
            },
            "decisions_68_77": {
                label: sorted(
                    {f"{data[label][i]['decision_state']}:{data[label][i]['decision_reason']}" for i in range(68, 78)}
                )
                for label in labels
            },
            "step_at_67_mm": {label: data[label][EXPECTED_STOP]["decision_step_mm"] for label in labels},
        }
    }
    claims["approach_ends_at_67_all_seven"]["holds"] = all(
        markers[label]["last_approach_step_frame_through_77"] == EXPECTED_STOP
        and all(data[label][i]["decision_state"] == "hold" for i in range(68, 78))
        for label in labels
    )
    claims["residual_settling_after_67_mm"] = {
        label: {
            str(i): 1e3
            * float(
                np.linalg.norm(
                    np.asarray(data[label][i]["tool_pose_wxyz"][:3])
                    - np.asarray(data[label][i - 1]["tool_pose_wxyz"][:3])
                )
            )
            for i in range(66, 75)
        }
        for label in labels
    }
    per_run = {}
    for label in NO_SHADOW:
        per_run[label] = {
            "align_frames": [i for i in range(POSE_FIRST, POSE_LAST + 1) if data[label][i]["cut_phase"] == "align"],
            "closing_frames": [i for i in range(POSE_FIRST, POSE_LAST + 1) if data[label][i]["cut_phase"] == "closing"],
            "first_closing_cut_frame": markers[label]["first_closing_cut_frame"],
            "first_rendered_jaw_motion_frame": markers[label]["first_rendered_jaw_motion_frame"],
            "detach_frame": markers[label]["detach_frame"],
            "rendered_progress_by_frame": {
                str(i): data[label][i]["visual_jaw_closure_progress"] for i in range(67, 78)
            },
        }
        per_run[label]["holds"] = (
            per_run[label]["align_frames"][:4] == [67, 68, 69, 70]
            and markers[label]["first_align_cut_frame"] == 67
            and markers[label]["first_closing_cut_frame"] == 71
            and markers[label]["first_rendered_jaw_motion_frame"] == 73
        )
    claims["no_shadow_align_67_70_closing_71_jaw_moves_73"] = {
        "per_run": per_run,
        "holds": all(item["holds"] for item in per_run.values()),
    }
    shadowed = {
        label: {
            "first_stopped_cut_frame": markers[label]["first_stopped_cut_frame"],
            "first_rendered_jaw_motion_frame": markers[label]["first_rendered_jaw_motion_frame"],
            "max_rendered_progress_60_77": max(
                float(data[label][i]["visual_jaw_closure_progress"] or 0.0) for i in range(POSE_FIRST, POSE_LAST + 1)
            ),
        }
        for label in SHADOWED
    }
    claims["shadowed_stop_at_67_jaw_never_moves"] = {
        "per_run": shadowed,
        "holds": all(
            item["first_stopped_cut_frame"] == EXPECTED_STOP and item["first_rendered_jaw_motion_frame"] is None
            for item in shadowed.values()
        ),
    }
    return claims


def pose_comparison(base=VISION_ROBUSTNESS):
    """Frames 60-77 of the five shadowed recordings against the two shadow-removed runs (one frames.json at a time)."""
    data, markers, positions = {}, {}, {}
    for label, spec in {**SHADOWED, **NO_SHADOW}.items():
        data[label], markers[label], positions[label] = compact_run(Path(base) / spec["run"])
    positions = {label: np.asarray(value, float) for label, value in positions.items()}
    separation = [
        {
            "frame": index,
            "largest_shadowed_vs_no_shadow_mm": max(
                1e3 * float(np.linalg.norm(positions[s][index] - positions[n][index]))
                for s in SHADOWED
                for n in NO_SHADOW
            ),
            "no_shadow_r1_vs_r2_mm": 1e3
            * float(np.linalg.norm(positions["no_shadow_b_r1"][index] - positions["no_shadow_b_r2"][index])),
        }
        for index in range(POSE_LAST + 1)
    ]
    per_frame = [pose_frame(data, index) for index in range(POSE_FIRST, POSE_LAST + 1)]
    windows = {
        name: {
            key: window_max(per_frame, first, last, key)
            for key in ("largest_shadowed_vs_no_shadow", "no_shadow_r1_vs_r2_floor", "largest_within_shadowed")
        }
        for name, (first, last) in {"60-66": (60, 66), "67-72": (67, 72), "68-72": (68, 72), "60-77": (60, 77)}.items()
    }
    return {
        "frames_compared": [POSE_FIRST, POSE_LAST],
        "rotation_metric": (
            "Tool: jaw_self_mask.rotation_angle_deg on the normalized quaternions (cross-checked by 2 atan2(|v|, |w|) "
            "of q0* q1). Wrist camera: closest rotation (SVD) of each recorded matrix, then atan2(|vee(M - M^T)| / 2, "
            "(trace(M) - 1) / 2) of M = Ra^T Rb (the recorded matrices are orthonormal only to ~4e-7)."
        ),
        "tool_rotation_method_max_discrepancy_deg": max(
            row["tool_rotation_method_max_discrepancy_deg"] for row in per_frame
        ),
        "claims": pose_claims(data, markers),
        "windows": windows,
        "run_markers": markers,
        "tool_path_separation_0_77": separation,
        "per_frame": per_frame,
    }


# ------------------------------------------------------------------------------------------------ headline
def _arm_rows(entry, arm):
    rows = {}
    for row in entry["tables"][arm]:
        if row["frame"] < EXPECTED_STOP:
            continue
        event = row["depth_event"]
        rows[str(row["frame"])] = {
            "state": row["state"],
            "reason": row["reason"],
            "feature_count": row["feature_count"],
            "roundtrip_inliers": row["roundtrip_inlier_count"],
            "replenished_for_next_frame": row["replenished_for_next_frame"],
            "ncc": row["patch_correlation"],
            "grey_check": row["grey_check"],
            "confidence": row["confidence"],
            "measured_confidence": row["measured_confidence"],
            "depth_event": None
            if event is None
            else {
                "accept": event["accept"],
                "failed_conditions": event["failed_conditions"],
                "not_evaluated_conditions": event["not_evaluated_conditions"],
                "agreement_fraction": event["agreement_fraction"],
                "median_abs_mm": None if event["median_abs_m"] is None else 1e3 * event["median_abs_m"],
                "pixel_disagreement_px": event["pixel_disagreement_px"],
                "jaw_touches": event["jaw_touches"],
                "strict_confidence": event["strict_confidence"],
                "agreement_confidence": event["agreement_confidence"],
                "decision": event["decision"],
            },
            "depth_event_note": row.get("depth_event_note"),
            "pixel_xy": row["pixel_xy"],
            "target_position_world_m": row["target_position_world_m"],
            "target_shift_from_frame66_mm": row["target_shift_from_frame66_mm"],
            "target_shift_from_previous_frame_mm": row["target_shift_from_previous_frame_mm"],
            "mouth_distance_mm": row["mouth_distance_mm"],
            "patch_gray_mean_std": row["patch_gray_mean_std"],
        }
    return rows


def pointer_check(agreement):
    """The scope document's uncommitted pointer: NCC 0.82-0.88 at frame 68 and 0.98 or above through 72."""
    at68 = {label: item["68"]["ncc"] for label, item in agreement.items()}
    later = [(item[str(i)]["ncc"], label, i) for label, item in agreement.items() for i in range(69, LAST_FRAME + 1)]
    lowest = min(later)
    return {
        "ncc_68": at68,
        "ncc_68_range": [min(at68.values()), max(at68.values())],
        "lowest_ncc_69_72": {"value": lowest[0], "run": lowest[1], "frame": lowest[2]},
        "frames_69_72_below_0_98": sorted([label, i, ncc] for ncc, label, i in later if ncc < POINTER_NCC_FLOOR),
        "lowest_ncc_70_72": min(ncc for ncc, _, i in later if i >= 70),
    }


def _cut_gate_headline(entry):
    arithmetic = entry["summary"]["agreement"]["cut_gate_arithmetic"]
    frames = arithmetic["frames"]
    return {
        "note": "arithmetic on the agreement arm's targets and the recorded tool poses; not a controller replay",
        "thresholds": arithmetic["thresholds"],
        "tracking_and_within_mouth_and_alignment_tolerance_after_stop": all(
            item["tracking"] and item["within_mouth_tolerance"] and item["within_alignment_tolerance"]
            for item in frames.values()
        ),
        "max_mouth_distance_mm": max(item["mouth_distance_mm"] for item in frames.values()),
        "max_target_shift_from_previous_frame_mm": max(
            item["target_shift_from_previous_frame_mm"] for item in frames.values()
        ),
        "mouth_speed_mm_s": {frame: item["mouth_speed_mm_s"] for frame, item in frames.items()},
        "frames_below_stable_speed": [int(frame) for frame, item in frames.items() if item["below_stable_speed"]],
    }


def headline(runs, pose):
    """The compact answer: fidelity, frames 67-72 of both arms, the pointer, the cut-gate arithmetic, the poses."""
    ok = {label: entry for label, entry in runs.items() if "error" not in entry}
    out = {
        "errors": {label: entry["error"] for label, entry in runs.items() if "error" in entry},
        "fidelity_passed": bool(ok) and len(ok) == len(runs) and all(e["fidelity"]["passed"] for e in ok.values()),
        "inputs_match_committed_hashes": all(entry["inputs_match_committed_hashes"] for entry in runs.values()),
        "fidelity_checks": {label: entry["fidelity"]["checks"] for label, entry in ok.items()},
        "max_abs_float_difference_before_stop": {
            label: {
                name: max(item["frames_before_stop"]["max_abs_difference_per_field"].values(), default=0.0)
                for name, item in entry["fidelity"]["per_tracker"].items()
            }
            for label, entry in ok.items()
        },
        "agreement_67_72": {
            label: {
                "tracks_every_frame_67_72": entry["summary"]["agreement"]["tracks_every_frame_through_72"],
                "grey_failures_68_72": entry["summary"]["agreement"]["grey_failures_after_stop"],
                "frames": _arm_rows(entry, "agreement"),
            }
            for label, entry in ok.items()
        },
        "strict_67_72": {
            label: {
                "latched_lost_68_72": entry["summary"]["strict"]["latched_after_stop"],
                "frames": _arm_rows(entry, "strict"),
            }
            for label, entry in ok.items()
        },
        "cut_gate_arithmetic": {label: _cut_gate_headline(entry) for label, entry in ok.items()},
        "patch_gray_mean_agreement_pixel_60_72": {
            label: {
                str(row["frame"]): None if row["patch_gray_mean_std"] is None else row["patch_gray_mean_std"]["mean"]
                for row in entry["tables"]["agreement"]
            }
            for label, entry in ok.items()
        },
    }
    if len(ok) == len(SHADOWED):
        out["scope_document_pointer_check"] = pointer_check(
            {label: item["frames"] for label, item in out["agreement_67_72"].items()}
        )
    if pose is not None and "error" not in pose:
        out["pose_comparison"] = {
            "claims_hold": {name: item["holds"] for name, item in pose["claims"].items() if "holds" in item},
            "windows": pose["windows"],
            "residual_settling_after_67_mm": pose["claims"]["residual_settling_after_67_mm"],
        }
    return out


# ------------------------------------------------------------------------------------------------ main
def replay_one(label, spec, base, evidence):
    """Hash, replay, cross-check and tabulate one shadowed recording (errors are recorded, never dropped)."""
    path = Path(base) / spec["run"]
    entry = {"run": spec["run"], "live_tracker": spec["live_tracker"]}
    started = time.time()
    try:
        committed = committed_hashes(spec, evidence)
        entry["committed_hashes"] = committed
        entry["frame_files"] = frame_files_digest(path)
        entry["inputs_match_committed_hashes"] = committed["report.json"] == sha256(path / "report.json") and committed[
            "frames.json"
        ] == sha256(path / "frames.json")
        replay = replay_through(path)
        try:
            committed_run = rda.replay_run(path)
        except Exception as error:  # noqa: BLE001 - a failed cross-check is recorded, never dropped
            committed_run = {"error": repr(error)}
        entry["fidelity"] = fidelity(spec, replay, committed_run, evidence)
        first = min(TABLE_FIRST, replay["recorded_stop"])
        entry["tables"] = {name: table(name, replay, first) for name in TRACKERS}
        entry["summary"] = run_summary(replay, entry["tables"])
        entry["arithmetic_check"] = arithmetic_check(replay)
        entry["cut_config_as_built_by_runner"] = replay["cut_config"]
        entry["tracker_thresholds"] = {
            "min_patch_correlation": replay["min_patch_correlation"],
            "min_confidence": replay["min_confidence"],
        }
        entry["raw_events"] = {arm: {str(k): v for k, v in replay["raw_events"][arm].items()} for arm in ARMS}
        entry["preview_events"] = replay["preview_events"]
    except Exception as error:  # noqa: BLE001 - a failed replay is recorded, never dropped
        entry["error"] = repr(error)
        entry["traceback"] = traceback.format_exc()
        entry.setdefault("inputs_match_committed_hashes", False)
    entry["seconds"] = round(time.time() - started, 2)
    return entry


def inputs_sha256():
    """sha256 of every report.json and frames.json read, and of the committed documents checked against."""
    hashes = {}
    for spec in {**SHADOWED, **NO_SHADOW}.values():
        for name in ("report.json", "frames.json"):
            hashes[f"artifacts/vision_robustness/{spec['run']}/{name}"] = sha256(VISION_ROBUSTNESS / spec["run"] / name)
    for name in (*EVIDENCE_FILES, SCOPE_DOCUMENT):
        hashes[name] = sha256(ROOT / name)
    return hashes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    args = parser.parse_args(argv)
    refusal = output_refusal(args.output)
    if refusal:
        parser.error(refusal)
    import cv2

    cv2.setNumThreads(1)
    started = time.time()
    provenance_start = provenance()
    evidence = {name: json.loads((ROOT / name).read_text()) for name in EVIDENCE_FILES}
    hashes = inputs_sha256()
    try:
        pose = pose_comparison(VISION_ROBUSTNESS)
        for label, spec in NO_SHADOW.items():
            pose["run_markers"][label]["inputs_match_committed_hashes"] = committed_hashes(spec, evidence) == {
                "source": spec["committed_replay"],
                "report.json": sha256(VISION_ROBUSTNESS / spec["run"] / "report.json"),
                "frames.json": sha256(VISION_ROBUSTNESS / spec["run"] / "frames.json"),
            }
    except Exception as error:  # noqa: BLE001 - recorded, never dropped
        pose = {"error": repr(error), "traceback": traceback.format_exc()}
    print(f"pose comparison: {'ERROR ' + pose['error'] if 'error' in pose else 'ok'}", flush=True)
    runs = {}
    for label, spec in SHADOWED.items():
        runs[label] = replay_one(label, spec, VISION_ROBUSTNESS, evidence)
        entry = runs[label]
        status = f"ERROR {entry['error']}" if "error" in entry else f"fidelity_passed={entry['fidelity']['passed']}"
        print(f"{label}: {status} ({entry['seconds']} s)", flush=True)
    provenance_end = provenance()
    document = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/replay_past_stop.py",
        "scope": SCOPE,
        "scope_document": SCOPE_DOCUMENT,
        "arm_labels": {arm: rda.ARM_LABELS[arm] for arm in ARMS},
        "depth_label": rda.SIMULATOR_DEPTH_LABEL,
        "code_revision": provenance_start["code_revision"],
        "code_tree_dirty": provenance_start["code_tree_dirty"],
        "code_tree_status_porcelain": provenance_start["code_tree_status_porcelain"],
        "loaded_repository_modules_sha256": provenance_start["loaded_repository_modules_sha256"],
        "code_unchanged_during_run": provenance_start["code_revision"] == provenance_end["code_revision"]
        and provenance_start["code_tree_status_porcelain"] == provenance_end["code_tree_status_porcelain"]
        and all(
            provenance_end["loaded_repository_modules_sha256"].get(name) == digest
            for name, digest in provenance_start["loaded_repository_modules_sha256"].items()
        ),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "opencv": cv2.__version__,
            "platform": platform.platform(),
            "host": socket.gethostname(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        },
        "inputs_sha256": hashes,
        "replayed_frames": [0, LAST_FRAME],
        "recorded_stop_frame": EXPECTED_STOP,
        "float_tolerance": FLOAT_TOLERANCE,
        "exact_fields": list(EXACT_FIELDS),
        "arm_dependent_event_keys": sorted(ARM_DEPENDENT_EVENT_KEYS),
        "fidelity_criterion": FIDELITY_CRITERION,
        "headline": headline(runs, pose),
        "runs": runs,
        "pose_comparison": pose,
        "seconds": round(time.time() - started, 2),
    }
    with args.output.open("x") as stream:
        json.dump(normalize(document), stream, allow_nan=False)
        stream.write("\n")
    print(json.dumps({key: document["headline"][key] for key in ("errors", "fidelity_passed")}), flush=True)
    return 1 if document["headline"]["errors"] or "error" in pose else 0


if __name__ == "__main__":
    raise SystemExit(main())
