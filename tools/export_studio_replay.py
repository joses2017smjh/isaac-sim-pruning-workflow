#!/usr/bin/env python3
"""Export a recorded capture into the compact bundle the replay studio loads.

Every value the studio displays comes from here, and everything here comes from
a recorded capture. There is no browser physics, no interpolation and no policy:
if a number was not recorded, the studio has nothing to show for it.

Two things are deliberately excluded. Tree meshes, bark textures and mock-pruner
CAD-derived geometry are not cleared for redistribution, so no mesh of any kind
is written. The published page therefore shows recorded camera video and recorded
telemetry, and labels its schematic robot as a stand-in.

Video is re-encoded small enough to open on a phone. The manifest records the
byte budget so a regression is visible rather than discovered by a recruiter on
mobile data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Distribution ffmpeg builds often omit libx264; this repository already resolves
# that in one place, so reuse it rather than growing a second copy.
from compose_isaac_workflow import _ffmpeg_executable  # noqa: E402
from validate_vision_sequence import grade_sequence  # noqa: E402

SCHEMA_VERSION = 1

#: Total published payload we are willing to ask a phone to download.
DEFAULT_BUDGET_BYTES = 12 * 1024 * 1024

#: Re-encode width. The wrist camera records at 480x320, so this is a copy of the
#: recorded resolution rather than an upscale.
VIDEO_WIDTH = 480


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _round(value, digits=4):
    if value is None:
        return None
    number = float(value)
    return None if not np.isfinite(number) else round(number, digits)


def _grid(values, mask):
    """An 8x8 grid in millimetres, with invalid zones as null rather than a number."""
    out = []
    for row_values, row_mask in zip(values, mask, strict=True):
        row = []
        for value, valid in zip(row_values, row_mask, strict=True):
            if not valid or value is None or not np.isfinite(float(value)):
                row.append(None)
            else:
                row.append(round(float(value) * 1000.0, 1))
        out.append(row)
    return out


def frame_record(frame):
    """One frame, reduced to what the studio draws. Nothing is invented."""
    measurement = (frame.get("live_vision") or {}).get("measurement") or {}
    decision = frame.get("visual_servo_decision") or {}
    cut = (frame.get("live_vision") or {}).get("cut") or {}

    applied = frame.get("command_pose_root_wxyz") or []
    tool = frame.get("tool_position_m") or []

    return {
        "i": int(frame["index"]),
        "t": _round(frame.get("time_s"), 3),
        "phase": frame.get("phase"),
        # Tracker
        "state": measurement.get("state"),
        "reason": measurement.get("reason"),
        "features": measurement.get("feature_count"),
        "confidence": _round(measurement.get("confidence")),
        "depth_valid_fraction": _round(measurement.get("depth_valid_fraction")),
        "pixel": [_round(v, 1) for v in (measurement.get("pixel_xy") or [])] or None,
        # Gates and decision
        "decision_state": decision.get("state"),
        "decision_reason": decision.get("reason"),
        "remaining_m": _round(decision.get("remaining_distance_m")),
        "cut_phase": cut.get("phase"),
        "stopped_reason": cut.get("stopped_reason") or frame.get("sensor_stop_reason"),
        "released": bool(frame.get("detachment_requested_after_capture")),
        "closure": _round(frame.get("visual_jaw_closure_progress"), 3),
        # Commands. Proposed is what the controller asked for; applied is what the
        # recording shows actually went to the robot.
        "proposed_delta_mm": [_round(v * 1000.0, 2) for v in (decision.get("delta_world_m") or [])] or None,
        "applied_xyz": [_round(v) for v in applied[:3]] or None,
        "tool_xyz": [_round(v) for v in tool[:3]] or None,
        "contact_n": _round(frame.get("contact_force_n"), 3),
        "source_frame": frame.get("controller_source_frame_index"),
    }


def tof_series(frames):
    """Both 8x8 grids per frame, in millimetres, invalid zones null."""
    left, right = [], []
    for frame in frames:
        for key, sink in (("left", left), ("right", right)):
            values = frame.get(f"tof_{key}_m")
            mask = frame.get(f"tof_{key}_valid")
            sink.append(None if values is None or mask is None else _grid(values, mask))
    return {"left": left, "right": right}


def encode_video(source, destination, width=VIDEO_WIDTH, crf=32):
    """Re-encode to a small, widely playable mp4. Returns None if ffmpeg is absent."""
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None:
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-vf",
        f"scale={width}:-2",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    if result.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"ffmpeg failed for {source}: {result.stderr[-400:]}")
    return destination


def encode_frames(pattern_dir, prefix, destination, fps, width=VIDEO_WIDTH, crf=32):
    """Encode a recorded PNG sequence. Captures from the sweep have no composed video."""
    ffmpeg = _ffmpeg_executable()
    if ffmpeg is None or not sorted(Path(pattern_dir).glob(f"{prefix}_*.png")):
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-loglevel",
        "error",
        "-framerate",
        str(fps),
        "-i",
        str(Path(pattern_dir) / f"{prefix}_%05d.png"),
        "-vf",
        f"scale={width}:-2",
        "-c:v",
        "libx264",
        "-preset",
        "slow",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        "-an",
        str(destination),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=900, check=False)
    if result.returncode != 0 or not destination.is_file():
        raise RuntimeError(f"ffmpeg failed for {prefix} frames in {pattern_dir}: {result.stderr[-400:]}")
    return destination


def export_run(capture_dir, output_dir, run_id, label, outcome, *, encode=True):
    capture_dir, output_dir = Path(capture_dir), Path(output_dir)
    report = json.loads((capture_dir / "report.json").read_text(encoding="utf-8"))
    document = json.loads((capture_dir / "frames.json").read_text(encoding="utf-8"))
    frames = document["frames"] if isinstance(document, dict) else document

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Only the independent grader decides pass or fail. A capture with no stored
    # grade is graded here rather than displayed as unknown.
    grade_path = capture_dir / "sequence_grade.json"
    grade = (
        json.loads(grade_path.read_text(encoding="utf-8")) if grade_path.is_file() else grade_sequence(report, frames)
    )

    media = {}
    if encode:
        available = sorted((capture_dir / "media").glob("*.mp4")) if (capture_dir / "media").is_dir() else []
        wrist = [item for item in available if "wrist" in item.name]
        overview = [item for item in available if "wrist" not in item.name]
        fps = document.get("fps", 10) if isinstance(document, dict) else 10
        for name, matches, prefix in (("wrist", wrist, "wrist"), ("overview", overview, "overview")):
            target = run_dir / f"{name}.mp4"
            produced = None
            if matches:
                produced = encode_video(matches[0], target)
            elif (capture_dir / "frames").is_dir():
                produced = encode_frames(capture_dir / "frames", prefix, target, fps)
            if produced is not None:
                media[name] = {"path": f"{run_id}/{name}.mp4", "bytes": target.stat().st_size}

    payload = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "label": label,
        "outcome": outcome,
        "source_job_id": str(report.get("job_id", run_id)),
        "fps": document.get("fps", 10) if isinstance(document, dict) else 10,
        "frame_count": len(frames),
        "target_id": report.get("blender_scene", {}).get("target", {}).get("id"),
        "daylight": report.get("blender_scene", {}).get("daylight", {}).get("preset"),
        "photometric_normalization": report.get("photometric_normalization"),
        "grade": None
        if grade is None
        else {
            "graded_here": not grade_path.is_file(),
            "ok": grade["ok"],
            "checks_passed": sum(1 for value in grade["checks"].values() if value),
            "checks_total": len(grade["checks"]),
            "failed_checks": grade["failed_checks"],
        },
        "frames": [frame_record(frame) for frame in frames],
        "tof_mm": tof_series(frames),
        "media": media,
        "provenance": {
            "capture_dir": str(capture_dir),
            "report_sha256": sha256(capture_dir / "report.json"),
            "frames_sha256": sha256(capture_dir / "frames.json"),
        },
    }
    telemetry = run_dir / "telemetry.json"
    telemetry.write_text(json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n", encoding="utf-8")
    return {
        "run_id": run_id,
        "label": label,
        "outcome": outcome,
        "telemetry": f"{run_id}/telemetry.json",
        "telemetry_bytes": telemetry.stat().st_size,
        "media": media,
        "frame_count": len(frames),
        "grade": payload["grade"],
        "target_id": payload["target_id"],
    }


def directory_bytes(path):
    return sum(item.stat().st_size for item in Path(path).rglob("*") if item.is_file())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="ID=DIR=LABEL=OUTCOME",
        help="Repeatable. Example: success=artifacts/isaac_render/job_21328323=Full sequence=pass",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget-bytes", type=int, default=DEFAULT_BUDGET_BYTES)
    parser.add_argument("--no-video", action="store_true", help="Skip re-encoding; telemetry only")
    args = parser.parse_args(argv)

    args.output.mkdir(parents=True, exist_ok=True)
    runs = []
    for spec in args.run:
        parts = spec.split("=")
        if len(parts) != 4:
            parser.error(f"--run needs ID=DIR=LABEL=OUTCOME, got {spec!r}")
        run_id, directory, label, outcome = parts
        runs.append(export_run(directory, args.output, run_id, label, outcome, encode=not args.no_video))

    total = directory_bytes(args.output)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "scope": (
            "Recorded simulator replays. Every displayed value comes from a capture file. "
            "No browser physics, no interpolation, no policy. The schematic robot is a "
            "labelled stand-in: tree meshes, textures and mock-pruner CAD are not "
            "cleared for redistribution and are not published."
        ),
        "runs": runs,
        "payload_bytes": total,
        "payload_budget_bytes": args.budget_bytes,
        "within_budget": total <= args.budget_bytes,
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in manifest.items() if key != "scope"}, indent=2))
    if not manifest["within_budget"]:
        print(f"Payload {total} exceeds budget {args.budget_bytes}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
