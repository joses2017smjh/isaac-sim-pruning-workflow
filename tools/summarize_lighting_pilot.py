#!/usr/bin/env python3
"""Aggregate one frozen lighting/tracker batch into a single evidence file.

Grades are produced by importing the independent grader and re-running it on
each capture's ``report.json`` and ``frames.json``. A stored ``sequence_grade``
is never trusted on its own: this tool records whether the fresh grade agrees
with it. A planned run whose capture is missing is reported as ``incomplete``
and is never silently dropped from the table.

This summarizes recorded simulator telemetry. It does not re-render, re-measure
images, or establish a population success rate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_vision_sequence import grade_sequence  # noqa: E402

SCHEMA_VERSION = 1

SCOPE = (
    "Aggregated grades and scheduler accounting for one frozen, pre-registered lighting batch. "
    "Every grade is recomputed here by the independent grader from recorded JSON. "
    "This is a paired one-target pilot: it is not a population success rate, not a CLAHE "
    "promotion, and not calibrated outdoor or sensor robustness."
)

LIMITS = [
    "One selected spur on one tree; six trials do not estimate a success rate.",
    "Daylight presets are artistic, not radiometric or astronomical.",
    "Depth is RTX optical-Z ground truth; branch identity comes from mesh metadata.",
    "Tracking loss after a completed release is expected during home-directed retreat.",
    "Feature and confidence margins are tracker-internal diagnostics, not the primary metric.",
    "Scheduler state COMPLETED (0:0) is accounting, never evidence of task success.",
]

SACCT_FIELDS = ("JobID", "JobIDRaw", "State", "ExitCode", "Elapsed", "Start", "End", "NodeList", "AllocTRES")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sacct_rows(array_job_id: str) -> dict:
    """Read scheduler accounting. This never submits, holds or modifies a job."""
    if not array_job_id:
        return {}
    command = ["sacct", "-j", str(array_job_id), "-X", "-P", "-n", "--format=" + ",".join(SACCT_FIELDS)]
    try:
        output = subprocess.check_output(command, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return {}
    rows = {}
    for line in output.splitlines():
        values = line.split("|")
        if len(values) != len(SACCT_FIELDS):
            continue
        row = dict(zip(SACCT_FIELDS, values, strict=True))
        rows[row["JobID"]] = row
    return rows


def tracking_profile(frames: list) -> dict:
    """Summarize tracker state over recorded frames, separating pre- and post-release."""
    release_index = None
    for frame in frames:
        if frame.get("detachment_requested_after_capture"):
            release_index = int(frame["index"])
            break

    states, first_loss, loss_reason = [], None, None
    features, confidences = [], []
    for frame in frames:
        measurement = (frame.get("live_vision") or {}).get("measurement") or {}
        state = measurement.get("state")
        states.append(state)
        if state == "tracking":
            if measurement.get("feature_count") is not None:
                features.append(int(measurement["feature_count"]))
            if measurement.get("confidence") is not None:
                confidences.append(float(measurement["confidence"]))
        elif first_loss is None:
            first_loss = int(frame["index"])
            loss_reason = measurement.get("reason")

    tracking_frames = sum(1 for state in states if state == "tracking")
    fps = None
    if frames:
        fps = frames[0].get("time_s")
    return {
        "recorded_frames": len(frames),
        "release_frame_index": release_index,
        "tracking_frames": tracking_frames,
        "tracking_fraction": round(tracking_frames / len(frames), 6) if frames else None,
        "first_nontracking_frame_index": first_loss,
        "first_nontracking_reason": loss_reason,
        "first_nontracking_is_after_release": (
            None if first_loss is None or release_index is None else first_loss > release_index
        ),
        "tracked_feature_count_min": min(features) if features else None,
        "tracked_feature_count_median": sorted(features)[len(features) // 2] if features else None,
        "tracked_confidence_min": min(confidences) if confidences else None,
        "wrist_rgb_std_first_frame": frames[0].get("wrist_rgb_std") if frames else None,
        "first_frame_time_s": fps,
    }


def summarize_run(batch: Path, row: dict, sacct: dict, array_job_id: str) -> dict:
    index = int(row["index"])
    name = f"run_{index:02d}_{row['daylight']}_{row['photometric_normalization']}"
    directory = batch / name
    entry = {
        "index": index,
        "run_directory": name,
        "daylight": row["daylight"],
        "photometric_normalization": row["photometric_normalization"],
        "array_task": f"{array_job_id}_{index}" if array_job_id else None,
    }
    entry["scheduler"] = sacct.get(entry["array_task"] or "")

    report_path, frames_path = directory / "report.json", directory / "frames.json"
    if not report_path.is_file() or not frames_path.is_file():
        entry["status"] = "incomplete"
        entry["incomplete_reason"] = "Missing report.json or frames.json for a planned run"
        entry["grade"] = None
        return entry

    report, frames_document = read_json(report_path), read_json(frames_path)
    frames = frames_document.get("frames") if isinstance(frames_document, dict) else frames_document
    fresh = grade_sequence(report, frames)

    stored_path = directory / "sequence_grade.json"
    stored = read_json(stored_path) if stored_path.is_file() else None
    entry["status"] = "graded"
    entry["input_sha256"] = {"report.json": sha256(report_path), "frames.json": sha256(frames_path)}
    entry["grade"] = {
        "ok": fresh["ok"],
        "checks_passed": sum(1 for value in fresh["checks"].values() if value),
        "checks_total": len(fresh["checks"]),
        "failed_checks": fresh["failed_checks"],
        "metrics": fresh["metrics"],
    }
    entry["stored_grade_agrees"] = None if stored is None else bool(stored.get("ok") == fresh["ok"])
    entry["tracking"] = tracking_profile(frames)

    experiment_path = directory / "experiment_result.json"
    if experiment_path.is_file():
        experiment = read_json(experiment_path)
        entry["recorded_job_id"] = experiment.get("job_id")
        entry["configuration_matches"] = experiment.get("configuration_matches")
        entry["task_outcome"] = experiment.get("task_outcome")
    return entry


def paired_comparison(runs: list) -> list:
    """Pair raw against CLAHE within each lighting preset, as the protocol fixes."""
    pairs = []
    lightings = sorted({run["daylight"] for run in runs})
    for lighting in lightings:
        modes = {run["photometric_normalization"]: run for run in runs if run["daylight"] == lighting}
        raw, clahe = modes.get("raw"), modes.get("clahe")
        if raw is None or clahe is None:
            pairs.append({"daylight": lighting, "complete_pair": False})
            continue

        def value(run, *keys):
            node = run
            for key in keys:
                if node is None:
                    return None
                node = node.get(key)
            return node

        pairs.append(
            {
                "daylight": lighting,
                "complete_pair": True,
                "raw_ok": value(raw, "grade", "ok"),
                "clahe_ok": value(clahe, "grade", "ok"),
                "task_completion_delta": "none"
                if value(raw, "grade", "ok") == value(clahe, "grade", "ok")
                else "differs",
                "raw_tracking_frames": value(raw, "tracking", "tracking_frames"),
                "clahe_tracking_frames": value(clahe, "tracking", "tracking_frames"),
                "raw_feature_count_min": value(raw, "tracking", "tracked_feature_count_min"),
                "clahe_feature_count_min": value(clahe, "tracking", "tracked_feature_count_min"),
                "raw_confidence_min": value(raw, "tracking", "tracked_confidence_min"),
                "clahe_confidence_min": value(clahe, "tracking", "tracked_confidence_min"),
            }
        )
    return pairs


def build(batch: Path, recorded_on: str) -> dict:
    plan = read_json(batch / "plan.json")
    receipt_path = batch / "submission_receipt.json"
    array_job_id = ""
    if receipt_path.is_file():
        array_job_id = str(read_json(receipt_path).get("stdout", "")).strip().split(";")[0]

    sacct = sacct_rows(array_job_id)
    runs = [summarize_run(batch, row, sacct, array_job_id) for row in plan["runs"]]
    graded = [run for run in runs if run["status"] == "graded"]
    passing = [run for run in graded if run["grade"]["ok"]]

    return {
        "schema_version": SCHEMA_VERSION,
        "recorded_on": recorded_on,
        "scope": SCOPE,
        "limits": LIMITS,
        "batch_dir": str(batch),
        "array_job_id": array_job_id,
        "code_revision": plan.get("code_revision"),
        "plan_created_utc": plan.get("created_utc"),
        "plan_sha256": sha256(batch / "plan.json"),
        "grader": "tools/validate_vision_sequence.py:grade_sequence",
        "grader_sha256": sha256(Path(__file__).resolve().parent / "validate_vision_sequence.py"),
        "summarizer_sha256": sha256(Path(__file__).resolve()),
        "planned_run_count": len(plan["runs"]),
        "graded_run_count": len(graded),
        "incomplete_run_count": len(runs) - len(graded),
        "passing_run_count": len(passing),
        "runs": runs,
        "paired_comparison": paired_comparison(graded),
        "scheduler_accounting_rows": len(sacct),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True, type=Path, help="Frozen batch directory containing plan.json")
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    parser.add_argument("--recorded-on", default=None, help="Override the recorded date, for reproducible output")
    args = parser.parse_args(argv)

    recorded_on = args.recorded_on or datetime.now(timezone.utc).date().isoformat()
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
        document = build(args.batch_dir, recorded_on)
        serialized = json.dumps(document, indent=2, allow_nan=False, sort_keys=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
