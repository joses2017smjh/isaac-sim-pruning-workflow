#!/usr/bin/env python3
"""Aggregate success-rate batches into one evidence file, through DuckDB.

The queries in ``sql/`` are the aggregation path, not a demonstration beside it:
every number this tool reports, including the Wilson interval, is computed by
those queries. Changing a definition means editing the SQL, where it is visible
and reviewable, rather than a Python expression buried in a loop.

Reading a batch follows the same rules as the rest of this repository. Only the
independent grader decides pass or fail, a planned run with no capture is
``incomplete`` rather than absent, and a target rejected before recording still
counts in the denominator.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from validate_vision_sequence import grade_sequence  # noqa: E402

SCHEMA_VERSION = 1
SQL_DIR = Path(__file__).resolve().parents[1] / "sql"

SCOPE = (
    "Task success rate over spur geometry, axis orientation and local occlusion at a "
    "canonicalized approach pose, from pre-registered targets. Not a field success rate, "
    "not a reachability study, and not outdoor or sensor-calibrated robustness."
)

#: A run that aborted before recording, with the guard that refused it.
LAYOUT_REJECTION = "Orchard layout rejected"

#: The renderer refuses a tree1 spur that is not in the export's listed
#: candidates. A target like that can never be presented, so its trial measures
#: the target register, not the controller.
UNPRESENTABLE_TARGET = "Unlisted tree1 components require a new geometry audit"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _stop_reason(grade):
    """First recorded stop reason, taken from the grader's own record."""
    stops = (grade.get("metrics") or {}).get("recorded_stops") or []
    for entry in stops:
        if ":" in str(entry):
            return str(entry).split(":")[-1].strip()
    return None


def _tracking(frames):
    tracking, features, confidences = 0, [], []
    for frame in frames:
        measurement = (frame.get("live_vision") or {}).get("measurement") or {}
        if measurement.get("state") == "tracking":
            tracking += 1
            if measurement.get("feature_count") is not None:
                features.append(int(measurement["feature_count"]))
            if measurement.get("confidence") is not None:
                confidences.append(float(measurement["confidence"]))
    return {
        "tracking_frames": tracking,
        "recorded_frames": len(frames),
        "tracked_feature_count_min": min(features) if features else None,
        "tracked_confidence_min": min(confidences) if confidences else None,
    }


def _first_frame(frames, predicate):
    """Index of the first recorded frame satisfying ``predicate``, or None."""
    for frame in frames:
        if predicate(frame):
            return int(frame.get("index", 0))
    return None


def _classify_missing(batch, index, row):
    """Say why a planned run produced no capture, using the job log if it exists."""
    logs = sorted((batch / "logs").glob(f"*_{index}.out")) if (batch / "logs").is_dir() else []
    for log in logs:
        text = log.read_text(encoding="utf-8", errors="replace")
        if LAYOUT_REJECTION in text:
            return "rejected_layout_startup_contact", LAYOUT_REJECTION
        if UNPRESENTABLE_TARGET in text:
            return "infrastructure", "target_not_presentable_unlisted_tree1_component"
        if "selected_branch_occluded_or_wrong_surface" in text:
            return "rejected_visibility", "selected_branch_occluded_or_wrong_surface"
    del row
    return "incomplete", None


def read_batch(batch: Path, condition: str):
    """One row per planned run. A planned run is never silently dropped."""
    batch = Path(batch)
    plan = read_json(batch / "plan.json")
    rows = []
    for row in plan["runs"]:
        index = int(row["index"])
        name = f"run_{index:02d}_{row['daylight']}"
        if "component_first_vertex" in row:
            name += f"_tree{row['target_tree_index']}_v{row['component_first_vertex']}"
        else:
            name += f"_{row['photometric_normalization']}"
        # A strategy row's capture is named after the strategy (run_vision_experiment.run_label).
        if "strategy" in row:
            name += f"_{row['strategy']['name']}"
        directory = batch / name

        record = {
            "condition": condition,
            "run_directory": name,
            "target_tree_index": row.get("target_tree_index"),
            "component_first_vertex": row.get("component_first_vertex"),
            "daylight": row.get("daylight"),
            "photometric_normalization": row.get("photometric_normalization"),
            "strategy": (row.get("strategy") or {}).get("name", "baseline"),
            "stop_frame": None,
            "final_phase_frame": None,
            "failed_checks": None,
            "status": "incomplete",
            "checks_passed": 0,
            "checks_total": 0,
            "applied_commands": 0,
            "stop_reason": None,
            "tracking_frames": 0,
            "recorded_frames": 0,
            "tracked_feature_count_min": None,
            "tracked_confidence_min": None,
            "centroid_error_m": None,
        }

        report_path, frames_path = directory / "report.json", directory / "frames.json"
        experiment_path = directory / "experiment_result.json"
        experiment = read_json(experiment_path) if experiment_path.is_file() else {}
        # A capture that failed before recording may still leave a stub report
        # behind. Grading that stub would file the run under a vague "failed"
        # bucket; the job log says why it never recorded.
        aborted = experiment.get("capture_ok") is False and experiment.get("task_outcome") == "runtime_failure"
        if aborted or not report_path.is_file() or not frames_path.is_file():
            status, reason = _classify_missing(batch, index, row)
            record["status"], record["stop_reason"] = status, reason
            rows.append(record)
            continue

        report = read_json(report_path)
        document = read_json(frames_path)
        frames = document["frames"] if isinstance(document, dict) else document
        grade = grade_sequence(report, frames)

        record.update(
            status="graded",
            checks_passed=sum(1 for value in grade["checks"].values() if value),
            checks_total=len(grade["checks"]),
            applied_commands=int((grade.get("metrics") or {}).get("applied_vision_command_frames") or 0),
            stop_reason=_stop_reason(grade),
            stop_frame=_first_frame(frames, lambda f: f.get("phase") == "stopped_failure"),
            final_phase_frame=_first_frame(
                frames, lambda f: (f.get("visual_servo_decision") or {}).get("approach_phase") == "final"
            ),
            failed_checks=",".join(sorted(k for k, v in grade["checks"].items() if not v)) or None,
            **_tracking(frames),
        )
        rows.append(record)
    return rows


def aggregate(rows, sql_dir=SQL_DIR):
    """Run the committed queries. Every reported number comes from these."""
    import duckdb

    if not rows:
        raise ValueError("No planned runs were read; refusing to report a rate over nothing")

    connection = duckdb.connect()
    # An explicit schema, so a row missing a field fails here rather than being
    # inferred as NULL and quietly changing a count.
    connection.execute(
        """
        CREATE TABLE eval_input (
            condition                 VARCHAR,
            run_directory             VARCHAR,
            target_tree_index         INTEGER,
            component_first_vertex    INTEGER,
            daylight                  VARCHAR,
            photometric_normalization VARCHAR,
            strategy                  VARCHAR,
            stop_frame                INTEGER,
            final_phase_frame         INTEGER,
            failed_checks             VARCHAR,
            status                    VARCHAR,
            checks_passed             INTEGER,
            checks_total              INTEGER,
            applied_commands          INTEGER,
            stop_reason               VARCHAR,
            tracking_frames           INTEGER,
            recorded_frames           INTEGER,
            tracked_feature_count_min INTEGER,
            tracked_confidence_min    DOUBLE,
            centroid_error_m          DOUBLE
        )
        """
    )
    columns = [description[0] for description in connection.execute("SELECT * FROM eval_input").description]
    placeholders = ", ".join("?" for _ in columns)
    connection.executemany(
        f"INSERT INTO eval_input VALUES ({placeholders})",
        [[row.get(column) for column in columns] for row in rows],
    )

    applied = []
    for path in sorted(sql_dir.glob("*.sql")):
        connection.execute(path.read_text(encoding="utf-8"))
        applied.append({"file": path.name, "sha256": sha256(path)})

    def fetch(view):
        cursor = connection.execute(f"SELECT * FROM {view}")
        names = [description[0] for description in cursor.description]
        return [dict(zip(names, values, strict=True)) for values in cursor.fetchall()]

    rate = fetch("success_rate")
    return {
        "queries": applied,
        "success_rate": rate[0] if rate else None,
        "per_condition": fetch("per_condition"),
        "failure_taxonomy": fetch("failure_taxonomy"),
        "rate_by_exclusion": fetch("rate_by_exclusion"),
        "tracking_coverage": fetch("tracking_coverage"),
        "per_target_by_strategy": fetch("per_target_by_strategy"),
        "runs": fetch("runs"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--batch",
        action="append",
        required=True,
        metavar="CONDITION=DIR",
        help="Repeatable. Example: source=artifacts/vision_robustness/targets-source-20260923",
    )
    parser.add_argument("--targets-file", type=Path, help="The pre-registered target register")
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    parser.add_argument("--recorded-on", default=None)
    args = parser.parse_args(argv)

    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")

    rows, batches = [], []
    for spec in args.batch:
        if "=" not in spec:
            parser.error(f"--batch needs CONDITION=DIR, got {spec!r}")
        condition, directory = spec.split("=", 1)
        rows.extend(read_batch(Path(directory), condition))
        batches.append({"condition": condition, "batch_dir": directory})

    result = aggregate(rows)
    document = {
        "schema_version": SCHEMA_VERSION,
        "recorded_on": args.recorded_on or datetime.now(timezone.utc).date().isoformat(),
        "scope": SCOPE,
        "protocol": "docs/EVAL_PROTOCOL_2026-09-23.md",
        "batches": batches,
        "grader": "tools/validate_vision_sequence.py:grade_sequence",
        "aggregation": "DuckDB over sql/*.sql; every reported number is produced by those queries",
        **result,
    }
    if args.targets_file and Path(args.targets_file).is_file():
        register = read_json(Path(args.targets_file))
        document["target_register"] = {
            "path": str(args.targets_file),
            "sha256": sha256(Path(args.targets_file)),
            "seed": register.get("seed"),
            "target_count": register.get("target_count"),
            "selection_rule": register.get("selection_rule"),
        }

    serialized = json.dumps(document, indent=2, allow_nan=False, default=str) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    summary = {key: value for key, value in document.items() if key not in ("runs", "tracking_coverage")}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
