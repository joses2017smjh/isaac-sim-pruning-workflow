#!/usr/bin/env python3
"""Does one metric range at the target fix the learned depth? Zero-GPU analysis.

Reads saved predictions, ground-truth depth and masks from existing evaluations
and re-scores each frame after anchoring the prediction with a single metric
range at the tracked target: the range the rig itself reads there. Two anchors
are compared, a shift and a scale, because the measured failures differ in kind
(a constant offset on Blender daylight, range compression on Isaac).

The affine ceiling fits scale and shift to the whole ground-truth frame. It is
the best any per-frame linear correction could do and it is reported only to
separate "wrong scale" from "wrong shape". It is never a result.

Protocol and pre-registered predictions: docs/EVAL_PROTOCOL_DEPTH_ANCHORING_2026-09-23.md
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1
SQL_DIR = Path(__file__).resolve().parents[1] / "sql" / "anchoring"

SCOPE = (
    "Zero-GPU re-scoring of saved learned-depth predictions after anchoring with one metric range at the "
    "tracked target. On Isaac frames the anchor is the RTX optical-Z the controller already uses; on Blender "
    "frames it is ground-truth depth at the target pixel standing in for one time-of-flight zone. The affine "
    "ceiling uses the whole ground-truth frame and is a diagnostic, never a result."
)

LIMITS = [
    "The Blender anchor is a ground-truth stand-in for a ToF zone; no ToF noise, multipath or registration modelled.",
    "One anchor fixes one parameter, shift or scale, not both; the affine ceiling shows what both would give.",
    "Frames whose target was not visible are kept in the table unanchored; they count in n_frames, not n_anchored.",
    "Nothing here is closed-loop control, and nothing is measured on a physical sensor.",
]

COLUMNS = [
    ("model", "VARCHAR"),
    ("family", "VARCHAR"),
    ("tree_id", "VARCHAR"),
    ("lighting", "VARCHAR"),
    # The cell. Equal to lighting for the matrix and Stage A; the controls label
    # every changed axis here so camera cells never pool under their light.
    ("condition", "VARCHAR"),
    ("view_id", "VARCHAR"),
    ("sequence", "VARCHAR"),
    ("anchor_source", "VARCHAR"),
    ("anchored", "BOOLEAN"),
    ("anchor_range_m", "DOUBLE"),
    ("anchor_pred_m", "DOUBLE"),
    ("valid_pixels", "INTEGER"),
    ("raw_mae_m", "DOUBLE"),
    ("shift_mae_m", "DOUBLE"),
    ("scale_mae_m", "DOUBLE"),
    ("affine_ceiling_mae_m", "DOUBLE"),
    ("affine_scale", "DOUBLE"),
    ("affine_shift_m", "DOUBLE"),
    ("pred_gt_correlation", "DOUBLE"),
    # Many-zone variant: an 8x8 grid of ranges standing in for the VL53L8CX,
    # co-located with the camera, fitted per frame by scale and shift.
    ("n_zones", "INTEGER"),
    ("zone_mae_m", "DOUBLE"),
    ("zone_scale", "DOUBLE"),
    ("zone_shift_m", "DOUBLE"),
    ("raw_target_abs_m", "DOUBLE"),
    ("zone_target_abs_m", "DOUBLE"),
]

ZONE_GRID = 8
#: VL53L8CX diagonal field of view, the value the simulator's raycaster uses.
ZONE_DIAGONAL_FOV_DEG = 65.0
ZONE_MIN_PIXELS = 4
ZONE_MAX_RANGE_M = 3.4


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


def valid_pixels(pred, gt, mask=None):
    valid = np.isfinite(gt) & (gt > 0) & (gt < 1e6) & np.isfinite(pred) & (pred > 0)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    return valid


def mae(pred, gt, valid):
    return float(np.abs(pred[valid] - gt[valid]).mean()) if valid.any() else None


def anchor_values(pred, gt, valid, pixel, radius=1):
    """Median prediction and range over the valid 3x3 window at the target, or None."""
    if pixel is None:
        return None, None
    x, y = np.asarray(pixel, dtype=float)
    if not np.isfinite([x, y]).all() or x < 0 or y < 0 or x >= gt.shape[1] or y >= gt.shape[0]:
        return None, None
    x, y = int(round(x)), int(round(y))
    window = np.s_[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
    inside = valid[window]
    if not inside.any():
        return None, None
    return float(np.median(pred[window][inside])), float(np.median(gt[window][inside]))


def affine_ceiling(pred, gt, valid):
    """Least-squares scale and shift against ground truth. Diagnostic only."""
    if valid.sum() < 3:
        return None, None, None
    p, g = pred[valid], gt[valid]
    design = np.stack([p, np.ones_like(p)], axis=1)
    (scale, shift), *_ = np.linalg.lstsq(design, g, rcond=None)
    return float(np.abs(scale * p + shift - g).mean()), float(scale), float(shift)


def correlation(pred, gt, valid):
    if valid.sum() < 3:
        return None
    p, g = pred[valid], gt[valid]
    # A constant array can carry a ~1e-16 standard deviation from mean rounding;
    # treat it as constant rather than reporting a meaningless correlation.
    if p.std() < 1e-9 or g.std() < 1e-9:
        return None
    return float(np.corrcoef(p, g)[0, 1])


def zone_grid(K, shape):
    """Zone index (row, col) of every pixel for an 8x8 grid centred on the camera boresight.

    The grid is a square pinhole with a 65 degree diagonal, the simulator's
    VL53L8CX model, aligned with the camera's optical axis. The real sensor sits
    beside the camera; that baseline and its reprojection are not modelled here.
    """
    fx, fy, cx, cy = float(K[0][0]), float(K[1][1]), float(K[0][2]), float(K[1][2])
    half = np.tan(np.radians(ZONE_DIAGONAL_FOV_DEG / 2)) / np.sqrt(2)
    width = 2 * half / ZONE_GRID
    ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
    col = np.floor(((xs - cx) / fx + half) / width).astype(int)
    row = np.floor(((ys - cy) / fy + half) / width).astype(int)
    inside = (col >= 0) & (col < ZONE_GRID) & (row >= 0) & (row < ZONE_GRID)
    return np.where(inside, row * ZONE_GRID + col, -1)


def zone_fit(pred, gt, valid, K):
    """Per-frame scale and shift from zone medians. Returns (n_zones, mae, scale, shift, fitted)."""
    zones = zone_grid(K, gt.shape)
    usable = valid & (zones >= 0) & (gt <= ZONE_MAX_RANGE_M)
    pairs = []
    for zone in np.unique(zones[usable]):
        member = usable & (zones == zone)
        if member.sum() >= ZONE_MIN_PIXELS:
            pairs.append((np.median(pred[member]), np.median(gt[member])))
    if len(pairs) < 3:
        return len(pairs), None, None, None, None
    p, g = np.asarray(pairs).T
    design = np.stack([p, np.ones_like(p)], axis=1)
    (scale, shift), *_ = np.linalg.lstsq(design, g, rcond=None)
    residual = np.abs(scale * p + shift - g)
    keep = residual <= max(3 * np.median(residual), 0.02)  # one robust trim against occluded zones
    if keep.sum() >= 3 and keep.sum() < len(p):
        (scale, shift), *_ = np.linalg.lstsq(design[keep], g[keep], rcond=None)
    fitted = scale * pred + shift
    return len(pairs), mae(fitted, gt, valid), float(scale), float(shift), fitted


def target_abs(values, gt, valid, pixel, radius=1):
    if pixel is None:
        return None
    x, y = np.asarray(pixel, dtype=float)
    if not np.isfinite([x, y]).all() or x < 0 or y < 0 or x >= gt.shape[1] or y >= gt.shape[0]:
        return None
    x, y = int(round(x)), int(round(y))
    window = np.s_[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1]
    inside = valid[window]
    if not inside.any():
        return None
    return float(np.abs(np.median(values[window][inside]) - np.median(gt[window][inside])))


def score_frame(pred, gt, mask, pixel, target_visible, K=None):
    valid = valid_pixels(pred, gt, mask)
    row = {
        "anchored": False,
        "anchor_range_m": None,
        "anchor_pred_m": None,
        "valid_pixels": int(valid.sum()),
        "raw_mae_m": mae(pred, gt, valid),
        "shift_mae_m": None,
        "scale_mae_m": None,
        "pred_gt_correlation": correlation(pred, gt, valid),
        "n_zones": 0,
        "zone_mae_m": None,
        "zone_scale": None,
        "zone_shift_m": None,
        "raw_target_abs_m": None,
        "zone_target_abs_m": None,
    }
    row["affine_ceiling_mae_m"], row["affine_scale"], row["affine_shift_m"] = affine_ceiling(pred, gt, valid)
    if K is not None and valid.any():
        n_zones, zone_mae, zone_scale, zone_shift, fitted = zone_fit(pred, gt, valid, K)
        row.update(n_zones=int(n_zones), zone_mae_m=zone_mae, zone_scale=zone_scale, zone_shift_m=zone_shift)
        if target_visible is not False and fitted is not None:
            row["raw_target_abs_m"] = target_abs(pred, gt, valid, pixel)
            row["zone_target_abs_m"] = target_abs(fitted, gt, valid, pixel)
    if target_visible is False or not valid.any():
        return row
    anchor_pred, anchor_range = anchor_values(pred, gt, valid, pixel)
    if anchor_pred is None or anchor_range is None or anchor_pred <= 0:
        return row
    row.update(anchored=True, anchor_range_m=anchor_range, anchor_pred_m=anchor_pred)
    row["shift_mae_m"] = mae(pred - anchor_pred + anchor_range, gt, valid)
    row["scale_mae_m"] = mae(pred * (anchor_range / anchor_pred), gt, valid)
    return row


def build_rows(model, evaluation):
    rows = []
    for record in evaluation.get("rows", []):
        frame = record["frame"]
        prediction = record.get("prediction")
        if not prediction or not Path(prediction).is_file() or not Path(frame["depth"]).is_file():
            continue
        pred = np.load(prediction, allow_pickle=False).astype(float)
        gt = np.load(frame["depth"], allow_pickle=False).astype(float)
        mask = _load_mask(frame)
        if mask is not None and mask.shape != gt.shape:
            raise ValueError(f"Mask shape differs from depth for {frame.get('rgb')}")
        anchor_source = (
            "gt_at_target_pixel_stand_in_for_tof_zone"
            if mask is not None
            else "rtx_optical_z_at_tracked_pixel_as_used_by_controller"
        )
        K = frame.get("K") or (frame.get("camera") or {}).get("wrist_intrinsics")
        scored = score_frame(pred, gt, mask, frame.get("target_pixel_xy"), frame.get("target_visible"), K=K)
        rows.append(
            {
                "model": model,
                "family": _family(frame),
                "tree_id": frame.get("tree_id"),
                "lighting": frame.get("lighting"),
                "condition": frame.get("condition", frame.get("lighting")),
                "view_id": frame.get("view_id"),
                "sequence": frame.get("sequence"),
                "anchor_source": anchor_source,
                **scored,
            }
        )
    return rows


def aggregate(rows, sql_dir=SQL_DIR):
    import duckdb

    if not rows:
        raise ValueError("No frames with saved predictions were read; refusing to report over nothing")
    connection = duckdb.connect()
    connection.execute("CREATE TABLE anchor_input (" + ", ".join(f"{n} {t}" for n, t in COLUMNS) + ")")
    names = [name for name, _ in COLUMNS]
    connection.executemany(
        f"INSERT INTO anchor_input VALUES ({', '.join('?' for _ in names)})",
        [[row.get(name) for name in names] for row in rows],
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
        "cells": fetch("anchoring_cells"),
        "per_tree": fetch("anchoring_per_tree"),
        "frames": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", action="append", required=True, metavar="MODEL=PATH")
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    parser.add_argument("--recorded-on", default=None)
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")

    rows, inputs = [], []
    for spec in args.evaluation:
        if "=" not in spec:
            parser.error(f"--evaluation needs MODEL=PATH, got {spec!r}")
        model, path = spec.split("=", 1)
        evaluation = read_json(path)
        built = build_rows(model, evaluation)
        rows.extend(built)
        meta = evaluation.get("model") or {}
        inputs.append(
            {
                "model": model,
                "path": str(path),
                "sha256": sha256(path),
                "job_id": evaluation.get("job_id"),
                "frames_scored": len(built),
                "checkpoint_sha256": meta.get("checkpoint_sha256"),
            }
        )

    result = aggregate(rows)
    document = {
        "schema_version": SCHEMA_VERSION,
        "recorded_on": args.recorded_on or datetime.now(timezone.utc).date().isoformat(),
        "protocol": "docs/EVAL_PROTOCOL_DEPTH_ANCHORING_2026-09-23.md",
        "scope": SCOPE,
        "limits": LIMITS,
        "inputs": inputs,
        "aggregation": "DuckDB over sql/anchoring/*.sql; every reported number is produced by those queries",
        "gpu_used": False,
        **result,
    }
    serialized = json.dumps(document, indent=2, allow_nan=False, default=str) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({k: v for k, v in document.items() if k not in ("frames", "per_tree")}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
