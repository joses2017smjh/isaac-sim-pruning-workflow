#!/usr/bin/env python3
"""Frozen metric-depth evaluation. GT is never used to fit/align predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def depth_metrics(pred, gt, mask=None):
    pred, gt = np.asarray(pred, dtype=float), np.asarray(gt, dtype=float)
    if pred.shape != gt.shape or pred.ndim != 2:
        raise ValueError("Depth maps must be matching HxW arrays")
    valid_gt = np.isfinite(gt) & (gt > 0) & (gt < 1e6)
    if mask is not None:
        if np.shape(mask) != gt.shape:
            raise ValueError("Mask shape mismatch")
        valid_gt &= np.asarray(mask, dtype=bool)
    valid = valid_gt & np.isfinite(pred) & (pred > 0)
    total, count = int(valid_gt.sum()), int(valid.sum())
    result = {
        "gt_pixels": total,
        "valid_prediction_pixels": count,
        "valid_pixel_rate": count / total if total else None,
    }
    if not count:
        return dict(
            result,
            rmse_m=None,
            mae_m=None,
            median_abs_m=None,
            p90_abs_m=None,
            p95_abs_m=None,
            abs_relative=None,
            outlier_gt_20mm_fraction=None,
            outlier_gt_10pct_fraction=None,
        )
    error = np.abs(pred[valid] - gt[valid])
    relative = error / gt[valid]
    return dict(
        result,
        rmse_m=float(np.sqrt(np.mean(error**2))),
        mae_m=float(error.mean()),
        median_abs_m=float(np.median(error)),
        p90_abs_m=float(np.quantile(error, 0.90)),
        p95_abs_m=float(np.quantile(error, 0.95)),
        abs_relative=float(relative.mean()),
        outlier_gt_20mm_fraction=float((error > 0.020).mean()),
        outlier_gt_10pct_fraction=float((relative > 0.10).mean()),
    )


def target_metrics(pred, gt, pixel, radius=1, mask=None):
    """Depth error in a small window around the target pixel.

    With ``mask`` given, only pixels on the tree are scored. Without it, a
    (2*radius+1)^2 window on a spur one pixel wide is mostly background, and the
    reported "target" error is then the error on whatever lies behind the branch.
    The unmasked form is kept so earlier results remain comparable; the masked
    form is what a target-level claim should cite.
    """
    if pixel is None:
        return None
    x, y = np.asarray(pixel, dtype=float)
    if not np.isfinite([x, y]).all() or x < 0 or y < 0 or x >= gt.shape[1] or y >= gt.shape[0]:
        return {"valid": False, "reason": "target_outside_frame"}
    x, y = int(round(x)), int(round(y))
    x, y = min(x, gt.shape[1] - 1), min(y, gt.shape[0] - 1)
    sl = np.s_[
        max(0, y - radius) : min(gt.shape[0], y + radius + 1), max(0, x - radius) : min(gt.shape[1], x + radius + 1)
    ]
    p, g = pred[sl], gt[sl]
    window_mask = None
    if mask is not None:
        if np.shape(mask) != gt.shape:
            raise ValueError("Mask shape mismatch")
        window_mask = np.asarray(mask, dtype=bool)[sl]
    m = depth_metrics(p, g, window_mask)
    valid = np.isfinite(g) & (g > 0) & (g < 1e6) & np.isfinite(p) & (p > 0)
    if window_mask is not None:
        valid &= window_mask
    result = dict(
        m,
        valid=bool(valid.any()),
        pixel_xy=[x, y],
        radius_px=radius,
        predicted_m=float(np.median(p[valid])) if valid.any() else None,
        reference_m=float(np.median(g[valid])) if valid.any() else None,
        mask_gated=window_mask is not None,
    )
    if window_mask is not None:
        result["window_tree_pixels"] = int(window_mask.sum())
        if not window_mask.any():
            result["reason"] = "no_tree_pixels_in_target_window"
    return result


def sanity_gates(rows):
    targets = [r["target"] for r in rows if r.get("target") is not None]
    # Frozen before the first inference: local error, valid coverage, and 10 Hz latency.
    checks = {
        "has_target_measurements": bool(targets),
        "all_target_measurements_valid": bool(targets) and all(t.get("valid", False) for t in targets),
        "target_p95_absolute_le_20mm": bool(targets)
        and all(t.get("p95_abs_m") is not None and t["p95_abs_m"] <= 0.020 for t in targets),
        "target_mean_relative_le_10pct": bool(targets)
        and all(t.get("abs_relative") is not None and t["abs_relative"] <= 0.10 for t in targets),
        "target_coverage_ge_99pct": bool(targets)
        and all(t.get("valid_pixel_rate", 0) is not None and t.get("valid_pixel_rate", 0) >= 0.99 for t in targets),
        "p95_inference_le_100ms": bool(rows)
        and float(np.quantile([r["inference_seconds"] for r in rows], 0.95)) <= 0.10,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "threshold_origin": "Pre-registered engineering sanity criteria, not calibrated hardware safety limits",
        "next_stage": "RTX-controlled diagnostic shadow permitted; learned control requires offline and shadow gates",
    }


class FrozenDA2:
    def __init__(self, companion, checkpoint, expected_sha256, device="cuda"):
        import torch

        self.torch = torch
        self.checkpoint = Path(checkpoint)
        self.sha256 = sha256(checkpoint)
        if self.sha256 != expected_sha256:
            raise ValueError("Checkpoint hash differs from frozen plan")
        sys.path.insert(0, str(Path(companion) / "depth-anything-v2/metric_depth"))
        from depth_anything_v2.dpt import DepthAnythingV2

        if device == "cpu":
            from depth_anything_v2.dinov2_layers import attention

            attention.XFORMERS_AVAILABLE = False

        self.device = device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024], max_depth=20.0)
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(ckpt.get("model", ckpt), strict=True)
        self.model.to(device).eval()
        self.metadata = {
            "architecture": "DA2 metric ViT-L",
            "max_depth_m": 20.0,
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": self.sha256,
            "input": "OpenCV BGR, native size, infer_image(input_size=518)",
            "output": "metric metres at input resolution; no scale/shift fit",
            "torch": torch.__version__,
            "device": device,
        }

    def predict(self, bgr):
        with self.torch.inference_mode():
            if self.device == "cuda":
                self.torch.cuda.synchronize()
            start = time.perf_counter()
            pred = self.model.infer_image(bgr, input_size=518)
            if self.device == "cuda":
                self.torch.cuda.synchronize()
        return np.asarray(pred, dtype=np.float32), time.perf_counter() - start


def evaluate(args):
    import cv2
    from PIL import Image

    plan = json.loads(args.manifest.read_text())
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": 1,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "manifest_sha256": sha256(args.manifest),
        "plan": plan,
        "ok": False,
        "rows": [],
        "depth_convention": plan.get("depth_convention"),
    }
    try:
        model = FrozenDA2(args.companion, args.checkpoint, args.checkpoint_sha256, args.device)
        result["model"] = model.metadata
        warm = cv2.imread(plan["frames"][0]["rgb"])
        if warm is None:
            raise ValueError("Warmup RGB missing")
        model.predict(warm)
        for index, frame in enumerate(plan["frames"]):
            image = cv2.imread(frame["rgb"])
            if image is None:
                raise ValueError(f"RGB missing: {frame['rgb']}")
            gt = np.load(frame["depth"], allow_pickle=False)
            if image.shape[:2] != gt.shape:
                raise ValueError("Native RGB and GT dimensions differ")
            pred, latency = model.predict(image)
            mask = np.asarray(Image.open(frame["mask"])) > 0 if frame.get("mask") else None
            if mask is not None and mask.ndim == 3:
                mask = mask.any(axis=2)
            row = {
                "index": index,
                "frame": frame,
                "inference_seconds": latency,
                "all_valid_gt": depth_metrics(pred, gt),
                "mask_metrics": depth_metrics(pred, gt, mask) if mask is not None else None,
                "target": target_metrics(pred, gt, frame.get("target_pixel_xy")),
                # Scored on tree pixels only. Absent when the plan has no mask.
                "target_masked": (
                    target_metrics(pred, gt, frame.get("target_pixel_xy"), mask=mask) if mask is not None else None
                ),
                "rgb_sha256": sha256(frame["rgb"]),
                "gt_sha256": sha256(frame["depth"]),
            }
            if frame.get("target_visible") is False:
                row["target"] = {"valid": False, "reason": "geometric_target_not_visible"}
                if mask is not None:
                    row["target_masked"] = {"valid": False, "reason": "geometric_target_not_visible"}
            if args.save_predictions:
                path = args.output / f"prediction_{index:05d}.npy"
                np.save(path, pred, allow_pickle=False)
                row["prediction"] = str(path)
            result["rows"].append(row)
            print(
                json.dumps(
                    {"index": index, "target": row["target"], "rmse_m": row["all_valid_gt"]["rmse_m"]}, allow_nan=False
                ),
                flush=True,
            )
        result["sanity_gates"] = sanity_gates(result["rows"])
        result["temporal_by_sequence"] = {}
        for sequence in sorted({r["frame"].get("sequence", "dataset") for r in result["rows"]}):
            valid_rows = [
                r
                for r in result["rows"]
                if r["frame"].get("sequence", "dataset") == sequence and r.get("target") and r["target"].get("valid")
            ]
            residual = [r["target"]["predicted_m"] - r["target"]["reference_m"] for r in valid_rows]
            result["temporal_by_sequence"][sequence] = {
                "frames_with_valid_target": len(residual),
                "target_error_std_m": float(np.std(residual)) if residual else None,
                "successive_target_error_change_rms_m": float(np.sqrt(np.mean(np.diff(residual) ** 2)))
                if len(residual) > 1
                else None,
                "scope": "GT-referenced depth residual at tracked target; not stationary-object depth variance",
            }
        if plan.get("temporal_scope", "").startswith("Static"):
            result["temporal_by_sequence"] = {"not_applicable": plan["temporal_scope"]}
        result["ok"] = True
    except Exception:
        import traceback

        result["error"] = traceback.format_exc()
        raise
    finally:
        (args.output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--save-predictions", action="store_true")
    return evaluate(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
