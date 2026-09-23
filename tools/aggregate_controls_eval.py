#!/usr/bin/env python3
"""Aggregate the generalization-controls scoring into one evidence document.

Row construction is shared with ``aggregate_family_eval.py``; the queries in
``sql/controls/`` produce every reported number. Two things are added here:
the registered baseline of each condition (from ``generalization_controls``)
is inserted as a table so the pairing is fixed before the numbers are seen, and
the registered-versus-scored frame counts from the evaluation plan are carried
beside the cells so a missing render is visible in the document.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generalization_controls as gc  # noqa: E402
from aggregate_family_eval import LIMITS, build_rows, read_json, sha256  # noqa: E402
from aggregate_family_eval import aggregate as run_queries  # noqa: E402

SCHEMA_VERSION = 1
SQL_DIR = Path(__file__).resolve().parents[1] / "sql" / "controls"
VIEWS = ("cells", "paired_effect", "sweep", "gates", "frames")

SCOPE = (
    "Offline metric-depth evaluation of frozen learned models on Blender Cycles renders of registered "
    "Envy and UFO L-Py trees under single-axis controls from the matrix source cell: lighting presets and "
    "brightness controls, test-time photometric normalization of the evening cell, the Isaac wrist camera "
    "model, close-range and upward-pitched rigs, and a distance sweep. Lighting presets are artistic, not "
    "radiometric. No prediction is aligned to ground truth. Nothing here is an Isaac render or closed-loop control."
)

CONTROL_LIMITS = LIMITS + [
    "Every control is a Cycles render; agreement with an Isaac cell is evidence about range and camera, "
    "not about the Isaac renderer, assets or tool geometry.",
    "The repo-local brightness controls scale sun energy and world strength by one fixed factor; Filmic tone "
    "mapping is not linear, so the achieved mean luma is reported from the plan, not assumed.",
    "Six-view DINO is scored only on far-rig training-camera cells; close, pitched, wrist-camera and sweep "
    "cells are DA2-only and say nothing about the refiner.",
    "The distance sweep has one centred pose per distance per tree; occluded targets drop out of the target "
    "columns and are counted in n_frames.",
]


def aggregate(rows, baseline_of=gc.BASELINE_OF):
    tables = {"baseline_of": ([("condition", "VARCHAR"), ("baseline", "VARCHAR")], list(baseline_of.items()))}
    return run_queries(rows, sql_dir=SQL_DIR, views=VIEWS, tables=tables)


def coverage(plan):
    """Registered against scored frames, from the evaluation plan, never from the scores."""
    return {
        "registered_frames": plan.get("registered_frames"),
        "scored_frames": plan.get("scored_frames"),
        "missing": plan.get("missing", []),
        "luma_by_condition": plan.get("luma_by_condition", {}),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True, metavar="MODEL=PATH")
    parser.add_argument("--plan", type=Path, required=True, help="The evaluation plan.json the models scored")
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    parser.add_argument("--recorded-on", default=None)
    parser.add_argument("--no-recompute", action="store_true", help="Do not reload saved predictions")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")

    rows, inputs = [], []
    for spec in args.evaluation:
        if "=" not in spec:
            parser.error(f"--evaluation needs MODEL=PATH, got {spec!r}")
        model, path = spec.split("=", 1)
        evaluation = read_json(path)
        rows.extend(build_rows(model, evaluation, recompute=not args.no_recompute))
        meta = evaluation.get("model") or {}
        inputs.append(
            {
                "model": model,
                "path": str(path),
                "sha256": sha256(path),
                "job_id": evaluation.get("job_id"),
                "frames": len(evaluation.get("rows", [])),
                "skipped_groups": evaluation.get("skipped_groups"),
                "checkpoint": meta.get("checkpoint"),
                "checkpoint_sha256": meta.get("checkpoint_sha256"),
            }
        )
    plan = read_json(args.plan)
    result = aggregate(rows)
    document = {
        "schema_version": SCHEMA_VERSION,
        "recorded_on": args.recorded_on or datetime.now(timezone.utc).date().isoformat(),
        "scope": SCOPE,
        "limits": CONTROL_LIMITS,
        "protocol": plan.get("protocol"),
        "inputs": inputs,
        "plan": {
            "path": str(args.plan),
            "sha256": sha256(args.plan),
            "batch_plan_sha256": plan.get("batch_plan_sha256"),
        },
        "coverage": coverage(plan),
        "baseline_of": gc.BASELINE_OF,
        "aggregation": "DuckDB over sql/controls/*.sql; every reported number is produced by those queries",
        "recomputed_from_saved_predictions": not args.no_recompute,
        **result,
    }
    serialized = json.dumps(document, indent=2, allow_nan=False, default=str) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    summary = {key: value for key, value in document.items() if key != "frames"}
    print(json.dumps(summary, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
