#!/usr/bin/env python3
"""Score the Isaac and Cycles renders of the same tree at the same poses on the same pixels.

The September 23 replay result compared the metric head's affine ceiling over
the Cycles tree mask (26 frames per light) with the Isaac full-frame ceiling
over all 200 Stage A frames, which includes the tool jaws at 0.11 m, posts,
wires and ground. This tool puts both renderers on identical pixel sets:

* ``occ``: pixels where Isaac sees a surface that Cycles does not (the tool and
  robot, absent from the Cycles scene): Isaac depth valid and either Cycles
  depth invalid (sky) or Isaac nearer by more than 5 mm.
* ``agree``: both depths valid and within 5 mm (verified geometry match).
* ``P``: ``occ`` dilated once by a 3x3 square (its anti-aliased edge).
* ``S``: the Cycles tree mask, ``agree`` and not ``P``: the tree pixels both
  renders share. Every like-for-like number is scored on ``S``.

It reads saved predictions only; no model runs. Simulator ground truth on
both sides; the Cycles mask is used offline for scoring, never at run time.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from anchor_depth_analysis import affine_ceiling, correlation, mae, sha256  # noqa: E402

SCHEMA_VERSION = 1
SQL_DIR = Path(__file__).resolve().parents[1] / "sql" / "renderer_gap"
AGREE_M = 0.005
PIXEL_SETS = (
    "isaac_full",
    "isaac_without_tool",
    "isaac_on_shared_tree",
    "cycles_on_shared_tree",
    "cycles_mask",
    "cycles_full",
    "isaac_tool",
)

COLUMNS = [
    ("renderer", "VARCHAR"),
    ("bark", "VARCHAR"),
    ("lighting", "VARCHAR"),
    ("isaac_run", "VARCHAR"),
    ("isaac_frame_index", "INTEGER"),
    ("pixel_set", "VARCHAR"),
    ("pixels", "INTEGER"),
    ("mae_m", "DOUBLE"),
    ("signed_median_m", "DOUBLE"),
    ("median_prediction_m", "DOUBLE"),
    ("median_reference_m", "DOUBLE"),
    ("correlation", "DOUBLE"),
    ("affine_ceiling_m", "DOUBLE"),
    ("constant_floor_m", "DOUBLE"),
]


def valid_depth(depth):
    depth = np.asarray(depth, dtype=float)
    return np.isfinite(depth) & (depth > 0) & (depth < 1e6)


def pixel_sets(isaac_depth, cycles_depth, cycles_mask):
    """The masks defined in the module docstring."""
    from scipy.ndimage import binary_dilation

    valid_i, valid_c = valid_depth(isaac_depth), valid_depth(cycles_depth)
    zi = np.where(valid_i, isaac_depth, np.nan)
    zc = np.where(valid_c, cycles_depth, np.nan)
    with np.errstate(invalid="ignore"):
        agree = valid_i & valid_c & (np.abs(zi - zc) < AGREE_M)
        occ = valid_i & (~valid_c | (zi < zc - AGREE_M))
    edge = binary_dilation(occ, structure=np.ones((3, 3), dtype=bool))
    shared = np.asarray(cycles_mask, dtype=bool) & agree & ~edge
    return {"occ": occ, "edge": edge, "agree": agree, "shared": shared, "valid_i": valid_i, "valid_c": valid_c}


def score(pred, gt, region):
    """Metrics of one prediction on one pixel set; None-filled when the set is empty."""
    pred, gt = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    valid = region & np.isfinite(pred) & (pred > 0) & valid_depth(gt)
    row = {
        "pixels": int(valid.sum()),
        "mae_m": None,
        "signed_median_m": None,
        "median_prediction_m": None,
        "median_reference_m": None,
        "correlation": None,
        "affine_ceiling_m": None,
        "constant_floor_m": None,
    }
    if valid.sum() < 3:
        return row
    p, g = pred[valid], gt[valid]
    row.update(
        mae_m=mae(pred, gt, valid),
        signed_median_m=float(np.median(p - g)),
        median_prediction_m=float(np.median(p)),
        median_reference_m=float(np.median(g)),
        correlation=correlation(pred, gt, valid),
        affine_ceiling_m=affine_ceiling(pred, gt, valid)[0],
        constant_floor_m=float(np.mean(np.abs(g - g.mean()))),
    )
    return row


def build_rows(replay_evaluation, isaac_evaluation):
    """One row per (render, frame, pixel set). Isaac rows are bark-independent and written once per pose."""
    from PIL import Image

    isaac_by_rgb = {str(Path(row["frame"]["rgb"]).resolve()): row for row in isaac_evaluation["rows"]}
    rows, seen = [], set()
    for record in replay_evaluation["rows"]:
        frame = record["frame"]
        isaac = isaac_by_rgb.get(str(Path(frame["isaac_rgb"]).resolve()))
        if isaac is None or not isaac.get("prediction") or not record.get("prediction"):
            raise ValueError(f"No saved Isaac or Cycles prediction for {frame['view_id']}")
        zc = np.load(frame["depth"], allow_pickle=False).astype(float)
        zi = np.load(frame["isaac_depth"], allow_pickle=False).astype(float)
        mask = np.asarray(Image.open(frame["mask"])) > 0
        if mask.ndim == 3:
            mask = mask.any(axis=2)
        if zc.shape != zi.shape or zc.shape != mask.shape:
            raise ValueError(f"Shape mismatch at {frame['view_id']}")
        sets = pixel_sets(zi, zc, mask)
        base = {
            "lighting": frame["lighting"],
            "isaac_run": frame["isaac_run"],
            "isaac_frame_index": int(frame["isaac_frame_index"]),
        }
        pred_c = np.load(record["prediction"], allow_pickle=False)
        for name, region, gt in (
            ("cycles_on_shared_tree", sets["shared"], zc),
            ("cycles_mask", mask, zc),
            ("cycles_full", sets["valid_c"], zc),
        ):
            rows.append(
                {**base, "renderer": "cycles", "bark": frame["bark"], "pixel_set": name, **score(pred_c, gt, region)}
            )
        key = (frame["isaac_run"], int(frame["isaac_frame_index"]))
        if key in seen:
            continue
        seen.add(key)
        pred_i = np.load(isaac["prediction"], allow_pickle=False)
        for name, region in (
            ("isaac_full", sets["valid_i"]),
            ("isaac_without_tool", sets["valid_i"] & ~sets["edge"]),
            ("isaac_on_shared_tree", sets["shared"]),
            ("isaac_tool", sets["occ"]),
        ):
            rows.append(
                {**base, "renderer": "isaac", "bark": "not_applicable", "pixel_set": name, **score(pred_i, zi, region)}
            )
    return rows


def all_frame_rows(isaac_evaluation, runs):
    """The published context: Isaac full frame over every frame of the replayed runs."""
    rows = []
    for record in isaac_evaluation["rows"]:
        frame = record["frame"]
        if frame.get("sequence") not in runs or not record.get("prediction"):
            continue
        gt = np.load(frame["depth"], allow_pickle=False).astype(float)
        pred = np.load(record["prediction"], allow_pickle=False)
        index = int(Path(frame["rgb"]).stem.split("_")[-1])
        rows.append(
            {
                "renderer": "isaac",
                "bark": "not_applicable",
                "lighting": frame["lighting"],
                "isaac_run": frame["sequence"],
                "isaac_frame_index": index,
                "pixel_set": "isaac_full_all_frames",
                **score(pred, gt, valid_depth(gt)),
            }
        )
    return rows


def aggregate(rows, sql_dir=SQL_DIR):
    import duckdb

    if not rows:
        raise ValueError("No rows; refusing to report over nothing")
    connection = duckdb.connect()
    connection.execute(f"CREATE TABLE gap_input ({', '.join(f'{n} {k}' for n, k in COLUMNS)})")
    names = [n for n, _ in COLUMNS]
    connection.executemany(
        f"INSERT INTO gap_input VALUES ({', '.join('?' for _ in names)})", [[r.get(n) for n in names] for r in rows]
    )
    applied = []
    for path in sorted(sql_dir.glob("*.sql")):
        connection.execute(path.read_text(encoding="utf-8"))
        applied.append({"file": path.name, "sha256": sha256(path)})

    def fetch(view):
        cursor = connection.execute(f"SELECT * FROM {view}")
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, values, strict=True)) for values in cursor.fetchall()]

    return {"queries": applied, "cells": fetch("gap_cells"), "paired": fetch("gap_paired")}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replay-evaluation", type=Path, required=True, help="DA2 evaluation.json of the Cycles replay"
    )
    parser.add_argument("--isaac-evaluation", type=Path, required=True, help="DA2 evaluation.json of Stage A")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--recorded-on", default=None)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    replay = json.loads(args.replay_evaluation.read_text())
    isaac = json.loads(args.isaac_evaluation.read_text())
    rows = build_rows(replay, isaac)
    runs = sorted({r["isaac_run"] for r in rows})
    rows += all_frame_rows(isaac, runs)
    document = {
        "schema_version": SCHEMA_VERSION,
        "recorded_on": args.recorded_on or datetime.now(timezone.utc).date().isoformat(),
        "scope": (
            "Offline rescoring of saved DA2 predictions: the Isaac Stage A frames and the Cycles replay of the same "
            "tree at the same recorded wrist poses, on identical pixel sets. Simulator ground truth on both sides; "
            "affine ceilings use ground truth and are never achievable numbers."
        ),
        "definitions": {
            "occ": f"Isaac depth valid and (Cycles depth invalid or Isaac nearer by more than {AGREE_M} m)",
            "shared_tree": f"Cycles tree mask, both depths within {AGREE_M} m, not occ dilated by a 3x3 square",
            "constant_floor_m": "MAE of the best least-squares constant on the same pixels",
        },
        "inputs": [
            {
                "path": str(args.replay_evaluation),
                "sha256": sha256(args.replay_evaluation),
                "job_id": replay.get("job_id"),
            },
            {
                "path": str(args.isaac_evaluation),
                "sha256": sha256(args.isaac_evaluation),
                "job_id": isaac.get("job_id"),
            },
        ],
        "aggregation": "DuckDB over sql/renderer_gap/*.sql",
        **aggregate(rows),
        "rows": rows,
    }
    text = json.dumps(document, indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(text)
    print(json.dumps({k: v for k, v in document.items() if k != "rows"}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
