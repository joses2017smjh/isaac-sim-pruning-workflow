#!/usr/bin/env python3
"""Aggregate frozen depth evaluations into one evidence file, through DuckDB.

Inputs are the ``evaluation.json`` files written by ``depth_generalization.py``
(DA2), ``dino_generalization.py`` (DINO) and the Isaac Stage A run. Every
reported number is produced by the queries in ``sql/family/``, which are hashed
into the output. Nothing here fits, aligns or scales a prediction to ground
truth: the one ground-truth-referenced diagnostic, the per-frame debiased error,
is named as such and never reported as achievable accuracy.

Where the evaluation saved its prediction arrays, the signed bias over tree
pixels and the mask-gated target error are recomputed here from disk, so a
result recorded before the masked target metric existed can still be scored
with it without spending GPU time again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from depth_generalization import target_metrics  # noqa: E402

SCHEMA_VERSION = 1
SQL_DIR = Path(__file__).resolve().parents[1] / "sql" / "family"

SCOPE = (
    "Offline metric-depth evaluation of frozen learned models on rendered frames with simulator "
    "ground-truth depth. Blender Cycles renders of L-Py trees (Envy, UFO) and Isaac RTX renders of the "
    "original orchard tree are scored separately and never pooled. Lighting presets are artistic, not "
    "radiometric. No prediction is aligned to ground truth. Nothing here is closed-loop control."
)

LIMITS = [
    "Every Envy tree was in the DA2 checkpoint's training or validation split; Envy is not an unseen-tree test.",
    "UFO is absent from the task fine-tuning manifests; backbone pretraining exposure is unknown.",
    "Views of one tree are re-renders of one geometry: n_trees, not n_frames, supports a population claim.",
    "mask_mae_debiased_m removes each frame's own median offset using ground truth; it is a diagnostic, not a result.",
    "Render resolution (512x288) differs from training renders (1920x1080); training lighting was not archived.",
    "Isaac Stage A frames sit at 0.16-0.39 m from the target, closer than any known training frame.",
]

COLUMNS = [
    ("model", "VARCHAR"),
    ("family", "VARCHAR"),
    ("tree_id", "VARCHAR"),
    ("lighting", "VARCHAR"),
    ("view_id", "VARCHAR"),
    ("sequence", "VARCHAR"),
    ("target_visible", "BOOLEAN"),
    ("mask_pixels", "INTEGER"),
    ("mask_mae_m", "DOUBLE"),
    ("mask_rmse_m", "DOUBLE"),
    ("mask_abs_relative", "DOUBLE"),
    ("mask_p95_abs_m", "DOUBLE"),
    ("mask_signed_median_m", "DOUBLE"),
    ("mask_signed_mean_m", "DOUBLE"),
    ("mask_mae_debiased_m", "DOUBLE"),
    ("full_mae_m", "DOUBLE"),
    ("full_rmse_m", "DOUBLE"),
    ("full_signed_median_m", "DOUBLE"),
    ("target_valid", "BOOLEAN"),
    ("target_mae_m", "DOUBLE"),
    ("target_p95_abs_m", "DOUBLE"),
    ("target_abs_relative", "DOUBLE"),
    ("target_coverage", "DOUBLE"),
    ("target_masked_valid", "BOOLEAN"),
    ("target_masked_mae_m", "DOUBLE"),
    ("target_masked_p95_abs_m", "DOUBLE"),
    ("target_masked_abs_relative", "DOUBLE"),
    ("target_masked_coverage", "DOUBLE"),
    ("target_masked_tree_pixels", "INTEGER"),
    ("inference_seconds", "DOUBLE"),
]


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _family(frame):
    explicit = frame.get("family")
    if explicit:
        return str(explicit)
    tree = str(frame.get("tree_id") or "")
    for name in ("envy", "ufo"):
        if name in tree:
            return name
    return "original_orchard" if tree else None


def _load_mask(frame):
    if not frame.get("mask"):
        return None
    from PIL import Image

    mask = np.asarray(Image.open(frame["mask"])) > 0
    return mask.any(axis=2) if mask.ndim == 3 else mask


def _signed(pred, gt, mask):
    """Signed error statistics over the pixels selected by ``mask`` (or all valid)."""
    valid = np.isfinite(gt) & (gt > 0) & (gt < 1e6) & np.isfinite(pred) & (pred > 0)
    if mask is not None:
        valid &= mask
    if not valid.any():
        return None, None, None
    signed = pred[valid] - gt[valid]
    median = float(np.median(signed))
    return median, float(signed.mean()), float(np.abs(signed - median).mean())


def _pick(metrics, key):
    return None if not metrics else metrics.get(key)


def _target_columns(prefix, target):
    valid = bool(target and target.get("valid"))
    return {
        f"{prefix}_valid": valid,
        f"{prefix}_mae_m": _pick(target, "mae_m") if valid else None,
        f"{prefix}_p95_abs_m": _pick(target, "p95_abs_m") if valid else None,
        f"{prefix}_abs_relative": _pick(target, "abs_relative") if valid else None,
        f"{prefix}_coverage": _pick(target, "valid_pixel_rate") if valid else None,
    }


def build_rows(model, evaluation, *, recompute=True):
    """One aggregation row per evaluated frame."""
    rows = []
    for record in evaluation.get("rows", []):
        frame = record["frame"]
        mask_metrics = record.get("mask_metrics")
        full = record.get("all_valid_gt") or {}
        row = {
            "model": model,
            "family": _family(frame),
            "tree_id": frame.get("tree_id"),
            "lighting": frame.get("lighting"),
            "view_id": frame.get("view_id"),
            "sequence": frame.get("sequence"),
            "target_visible": frame.get("target_visible"),
            "mask_pixels": _pick(mask_metrics, "gt_pixels"),
            "mask_mae_m": _pick(mask_metrics, "mae_m"),
            "mask_rmse_m": _pick(mask_metrics, "rmse_m"),
            "mask_abs_relative": _pick(mask_metrics, "abs_relative"),
            "mask_p95_abs_m": _pick(mask_metrics, "p95_abs_m"),
            "mask_signed_median_m": None,
            "mask_signed_mean_m": None,
            "mask_mae_debiased_m": None,
            "full_mae_m": full.get("mae_m"),
            "full_rmse_m": full.get("rmse_m"),
            "full_signed_median_m": None,
            "target_masked_tree_pixels": None,
            "inference_seconds": record.get("inference_seconds", record.get("six_view_total_inference_seconds")),
        }
        row.update(_target_columns("target", record.get("target")))
        masked = record.get("target_masked")

        prediction = record.get("prediction")
        if recompute and prediction and Path(prediction).is_file() and Path(frame["depth"]).is_file():
            pred = np.load(prediction, allow_pickle=False).astype(float)
            gt = np.load(frame["depth"], allow_pickle=False).astype(float)
            mask = _load_mask(frame)
            if mask is not None and mask.shape == gt.shape:
                row["mask_signed_median_m"], row["mask_signed_mean_m"], row["mask_mae_debiased_m"] = _signed(
                    pred, gt, mask
                )
                if masked is None and frame.get("target_visible") is not False:
                    masked = target_metrics(pred, gt, frame.get("target_pixel_xy"), mask=mask)
            row["full_signed_median_m"], _, _ = _signed(pred, gt, None)

        row.update(_target_columns("target_masked", masked))
        if masked and masked.get("valid"):
            row["target_masked_tree_pixels"] = masked.get("window_tree_pixels")
        rows.append(row)
    return rows


def aggregate(rows, sql_dir=SQL_DIR):
    """Run the committed queries. Every reported number comes from these."""
    import duckdb

    if not rows:
        raise ValueError("No evaluated frames were read; refusing to report over nothing")
    connection = duckdb.connect()
    schema = ", ".join(f"{name} {kind}" for name, kind in COLUMNS)
    connection.execute(f"CREATE TABLE eval_input ({schema})")
    names = [name for name, _ in COLUMNS]
    placeholders = ", ".join("?" for _ in names)
    connection.executemany(
        f"INSERT INTO eval_input VALUES ({placeholders})", [[row.get(name) for name in names] for row in rows]
    )
    applied = []
    for path in sorted(sql_dir.glob("*.sql")):
        connection.execute(path.read_text(encoding="utf-8"))
        applied.append({"file": path.name, "sha256": sha256(path)})

    def fetch(view):
        cursor = connection.execute(f"SELECT * FROM {view}")
        columns = [description[0] for description in cursor.description]
        return [dict(zip(columns, values, strict=True)) for values in cursor.fetchall()]

    return {
        "queries": applied,
        "cells": fetch("cells"),
        "lighting_effect": fetch("lighting_effect"),
        "family_comparison": fetch("family_comparison"),
        "gates": fetch("gates"),
        "frames": fetch("frames"),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evaluation",
        action="append",
        required=True,
        metavar="MODEL=PATH",
        help="Repeatable. Example: da2=artifacts/.../da2/evaluation.json",
    )
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
                "checkpoint": meta.get("checkpoint"),
                "checkpoint_sha256": meta.get("checkpoint_sha256"),
            }
        )

    result = aggregate(rows)
    document = {
        "schema_version": SCHEMA_VERSION,
        "recorded_on": args.recorded_on or datetime.now(timezone.utc).date().isoformat(),
        "scope": SCOPE,
        "limits": LIMITS,
        "inputs": inputs,
        "aggregation": "DuckDB over sql/family/*.sql; every reported number is produced by those queries",
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
