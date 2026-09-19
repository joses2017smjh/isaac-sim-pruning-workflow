#!/usr/bin/env python3
"""Paired raw/CLAHE offline RGB-D replay with deterministic input corruptions.

This does not simulate counterfactual robot motion or change recorded task
outcomes. All recorded frames remain in denominators after tracker loss. Source
RGB-D is read only; perturbations affect copies in memory. Scene target metadata
is used once for the recorded seed and subsequently only for error scoring.
The default reference is the current demo configuration, including feature
maintenance. Historical capture settings remain separate provenance; use
--config-policy recorded only for explicitly historical diagnostics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source/isaaclab_pruning"))
from isaaclab_pruning.perception.visual_servo import VisualServoConfig, VisualServoTracker
from isaaclab_pruning.task.simulated_cut import tool_mouth_geometry

SCENARIOS = ("baseline", "exposure_transition", "depth_dropout", "rgb_dropout")
STRESS_PROTOCOL = {
    "period_frames": 30,
    "exposure_transition": "index modulo 30: 0-9 unchanged, 10-19 gain 0.35, 20-29 gain 1.6 plus 20",
    "depth_dropout": "index modulo 30 in 10-12: all optical-Z samples NaN",
    "rgb_dropout": "index modulo 30 equals 10: all RGB samples zero",
    "exposure_space": "display-referred uint8 RGB; not a physical camera, sunlight or sensor noise model",
    "initial_preview": "unperturbed for all variants; one identical seed, no automatic reseeding",
}


def stress_inputs(rgb, depth, index, scenario):
    """Return independent arrays; schedules use capture indices, never truth."""
    if scenario not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {scenario}")
    image, optical_z = rgb.copy(), depth.copy()
    phase = index % 30
    applied = False
    if scenario == "exposure_transition" and phase >= 10:
        gain, offset = (0.35, 0.0) if phase < 20 else (1.6, 20.0)
        image = np.clip(image.astype(np.float32) * gain + offset, 0, 255).astype(np.uint8)
        applied = True
    elif scenario == "depth_dropout" and 10 <= phase <= 12:
        optical_z = np.full(depth.shape, np.nan, dtype=np.float32)
        applied = True
    elif scenario == "rgb_dropout" and phase == 10:
        image.fill(0)
        applied = True
    return image, optical_z, applied


def _initial_transform(report):
    camera = report["camera"]
    pose = report["initial_tool_pose_wxyz"][0]
    rotation = np.column_stack([tool_mouth_geometry(pose, closing_axis_tool=axis)[1] for axis in np.eye(3)])
    transform = np.eye(4)
    transform[:3, :3] = rotation @ np.asarray(camera["wrist_rotation_in_tool_ros"])
    transform[:3, 3] = np.asarray(pose[:3]) + rotation @ camera["wrist_position_in_tool_m"]
    return transform


def _source_config(report):
    """Match recorded feature settings, retaining the established depth gate."""
    if "tracker_config" in report:
        return VisualServoConfig(**report["tracker_config"])
    initial = report["vision_initialization"]["tracker"]
    features = initial.get("initial_feature_config", {})
    measurement = report.get("initial_live_vision", {}).get("measurement", {})
    window = measurement.get("depth_window_size_px", 9)
    radius = int((np.sqrt(window) - 1) / 2)
    if (2 * radius + 1) ** 2 != window:
        raise ValueError("Recorded depth window must be an odd square")
    return VisualServoConfig(
        roi_half_size_px=tuple(features.get("roi_half_size_px", (14, 24))),
        feature_quality_level=features.get("quality_level", 0.02),
        replenish_features=initial.get("feature_maintenance_enabled", False),
        depth_radius_px=radius,
    )


def _summary(rows):
    valid = [row for row in rows if row["state"] == "tracking"]
    errors = [row["target_centroid_error_m"] for row in valid if row["target_centroid_error_m"] is not None]
    active_times = [row["update_ms"] for row in rows if row["tracker_active_before_update"]]
    first_failure = next((row for row in rows if row["state"] != "tracking"), None)
    return {
        "attempted_frames": len(rows),
        "tracking_frames": len(valid),
        "tracking_fraction": len(valid) / len(rows) if rows else None,
        "state_counts": dict(Counter(row["state"] for row in rows)),
        "reason_counts": dict(Counter(row["reason"] for row in rows if row["reason"])),
        "first_nontracking_frame_index": None if first_failure is None else first_failure["index"],
        "first_nontracking_reason": None if first_failure is None else first_failure["reason"],
        "centroid_error_scored_frames": len(errors),
        "centroid_error_median_m": float(np.median(errors)) if errors else None,
        "centroid_error_p95_m": float(np.percentile(errors, 95)) if errors else None,
        "active_update_median_ms": float(np.median(active_times)) if active_times else None,
        "active_update_p95_ms": float(np.percentile(active_times, 95)) if active_times else None,
    }


def _row(frame, measurement, period, perturbation_applied, elapsed_ms, was_active, after_recorded_stop):
    # Metadata enters only AFTER update(). A surface-depth measurement is not
    # expected to equal the selected branch centroid, so name this bias clearly.
    target = frame.get("target_position_m")
    error = None
    if period == "pre_release" and measurement["state"] == "tracking" and target is not None:
        reference = np.asarray(target, dtype=float)
        if reference.shape == (3,) and np.isfinite(reference).all():
            error = float(np.linalg.norm(np.asarray(measurement["target_position_world_m"]) - reference))
    return {
        "index": frame["index"],
        "period": period,
        "recorded_robot_phase": frame.get("phase"),
        "after_recorded_stop": after_recorded_stop,
        "perturbation_applied": perturbation_applied,
        "state": measurement["state"],
        "reason": measurement.get("reason"),
        "flow_reason": measurement.get("flow_reason"),
        "feature_count": measurement["feature_count"],
        "confidence": measurement["confidence"],
        "pixel_xy": measurement["pixel_xy"],
        "target_position_world_m": measurement["target_position_world_m"],
        "target_centroid_error_m": error,
        "tracker_active_before_update": was_active,
        "update_ms": elapsed_ms,
    }


def benchmark_capture(source, scenarios=SCENARIOS, config_policy="current"):
    source = Path(source).resolve()
    digest = hashlib.sha256()

    def recorded_bytes(relative):
        content = (source / relative).read_bytes()
        digest.update(relative.encode())
        digest.update(hashlib.sha256(content).digest())
        return content

    report = json.loads(recorded_bytes("report.json"))
    frames = json.loads(recorded_bytes("frames.json"))["frames"]
    if not frames or len(frames) != report.get("frame_count", len(frames)):
        raise ValueError("No frames or inconsistent recorded frame count; refusing a partial denominator")
    indices = [frame["index"] for frame in frames]
    if indices != list(range(len(frames))):
        raise ValueError("Capture indices must be contiguous and ordered from zero")
    if not scenarios or len(set(scenarios)) != len(scenarios) or any(s not in SCENARIOS for s in scenarios):
        raise ValueError("Require unique, known scenarios")
    if config_policy not in ("current", "recorded"):
        raise ValueError("config_policy must be current or recorded")
    from io import BytesIO

    initial_rgb = np.array(Image.open(BytesIO(recorded_bytes("preview_wrist.png"))).convert("RGB"))
    initial_depth = np.load(BytesIO(recorded_bytes("preview_depth.npy")), allow_pickle=False)
    matrix = report["camera"]["wrist_intrinsics"]
    transform = _initial_transform(report)
    source_config = _source_config(report)
    config = (
        VisualServoConfig(depth_radius_px=1, feature_quality_level=0.005, replenish_features=True)
        if config_policy == "current"
        else source_config
    )
    variants = []
    for scenario in scenarios:
        for mode in ("raw", "clahe"):
            tracker = VisualServoTracker(replace(config, photometric_normalization=mode))
            initialized = tracker.initialize(initial_rgb, report["vision_initialization"]["pixel_xy"], initial_depth)
            warmup = tracker.update(initial_rgb, initial_depth, matrix, transform)
            variants.append(
                {
                    "scenario": scenario,
                    "mode": mode,
                    "tracker": tracker,
                    "config": asdict(tracker.config),
                    "initialization_state": initialized["state"],
                    "preview_measurement_state": warmup["state"],
                    "active": not warmup["requires_reinitialize"],
                    "frames": [],
                }
            )
    released = False
    release_frame = None
    source_stopped = False
    first_stop_reported = None
    for frame in frames:
        index = frame["index"]
        rgb = np.array(Image.open(BytesIO(recorded_bytes(f"frames/wrist_{index:05d}.png"))).convert("RGB"))
        depth = np.load(BytesIO(recorded_bytes(f"frames/depth_{index:05d}.npy")), allow_pickle=False)
        transform = np.eye(4)
        transform[:3, :3] = frame["wrist_rotation_w_ros"]
        transform[:3, 3] = frame["wrist_position_w_m"]
        cut = frame.get("live_vision", {}).get("cut", {})
        # The release request is made AFTER that frame's RGB-D capture.
        if cut.get("detached") and not frame.get("detachment_requested_after_capture", False):
            released = True
        period = "after_release" if released else "pre_release"
        if released and release_frame is None:
            release_frame = index
        for variant in variants:
            image, optical_z, applied = stress_inputs(rgb, depth, index, variant["scenario"])
            started = time.perf_counter()
            measurement = variant["tracker"].update(image, optical_z, matrix, transform)
            elapsed_ms = (time.perf_counter() - started) * 1000
            variant["frames"].append(
                _row(frame, measurement, period, applied, elapsed_ms, variant["active"], source_stopped)
            )
            variant["active"] = not measurement["requires_reinitialize"]
        if frame.get("detachment_requested_after_capture", False):
            released = True
        live = frame.get("live_vision", {})
        if live.get("external_stop_reason") or cut.get("phase") == "stopped" or frame.get("sensor_stop_reason"):
            if first_stop_reported is None:
                first_stop_reported = index
            # The current capture led to the decision; subsequent captures
            # observe the stopped trajectory rather than counterfactual motion.
            source_stopped = True
    for variant in variants:
        del variant["tracker"], variant["active"]
        rows = variant["frames"]
        variant["summary"] = _summary(rows)
        variant["by_release_period"] = {
            period: _summary([row for row in rows if row["period"] == period])
            for period in ("pre_release", "after_release")
        }
        variant["by_recorded_stop"] = {
            "through_first_stop_report": _summary([row for row in rows if not row["after_recorded_stop"]]),
            "after_first_stop_report": _summary([row for row in rows if row["after_recorded_stop"]]),
        }
    return {
        "capture_dir": str(source),
        "source_job_id": report.get("job_id"),
        "recorded_task_outcome": report.get("task_outcome"),
        "recorded_frames": len(frames),
        "first_capture_after_recorded_release": release_frame,
        "first_recorded_stop_report_frame_index": first_stop_reported,
        "source_metadata_and_all_rgb_depth_sha256": digest.hexdigest(),
        "reference_config_policy": config_policy,
        "reference_tracker_config": asdict(config),
        "source_tracker_config": asdict(source_config),
        "source_config_provenance": (
            "Complete recorded tracker_config"
            if "tracker_config" in report
            else "Recorded feature/maintenance/depth-window settings; other current tracker defaults"
        ),
        "comparison_scope": "Paired modes differ only in photometric_normalization; source config is provenance",
        "variants": variants,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scenario", choices=SCENARIOS, action="append")
    parser.add_argument(
        "--config-policy",
        choices=("current", "recorded"),
        default="current",
        help="current: current demo maintenance baseline (default); recorded: historical settings diagnostic",
    )
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing benchmark")
    cv2.setNumThreads(1)
    result = {
        "schema_version": 2,
        "evaluation_type": "offline_paired_saved_rgbd_replay",
        "scope": "Fixed recorded trajectories; replay cannot establish closed-loop success, release or retreat",
        "error_scope": "Pre-release measured surface to branch-centroid distance; no post-release accuracy claim",
        "stress_protocol": STRESS_PROTOCOL,
        "runtime": {"python": platform.python_version(), "opencv": cv2.__version__, "numpy": np.__version__},
        "code_sha256": {
            "benchmark": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "tracker": hashlib.sha256(
                Path(sys.modules[VisualServoTracker.__module__].__file__).read_bytes()
            ).hexdigest(),
        },
        "captures": [
            benchmark_capture(path, args.scenario or SCENARIOS, args.config_policy) for path in args.capture_dir
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            [
                {
                    "job_id": capture["source_job_id"],
                    "mode": variant["mode"],
                    "scenario": variant["scenario"],
                    **variant["summary"],
                }
                for capture in result["captures"]
                for variant in capture["variants"]
            ],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
