#!/usr/bin/env python3
"""Score the frozen relative DA2 head against ground truth through a disparity-space affine fit.

The metric head was trained from scratch on one camera and one depth band; the
relative head is the public Depth-Anything-V2 ViT-L checkpoint whose backbone
the fine-tune started from. Running both on the same frames separates two
readings of the close-range failure: if a per-frame affine fit of the relative
head's disparity to 1/GT reaches a low error where the metric head's own affine
ceiling is high, the metric head's output range is the failure and a relative
model plus a metric anchor is the architecture; if the relative fit is as bad,
the backbone does not resolve the structure at that camera and range.

Everything here uses ground truth to fit and is therefore a ceiling, never a
result. Disparity predictions are saved so the analysis can be recomputed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from depth_generalization import sha256  # noqa: E402

SCHEMA_VERSION = 1
MIN_PIXELS = 50


def valid_pixels(disparity, gt, mask=None, max_range_m=20.0):
    valid = np.isfinite(gt) & (gt > 0) & (gt <= max_range_m) & np.isfinite(disparity)
    if mask is not None:
        valid &= mask
    return valid


def disparity_affine_ceiling(disparity, gt, valid):
    """Least-squares 1/gt ~ s * disparity + b over ``valid``; error in metres after inversion.

    Returns None-filled numbers when too few pixels or a degenerate fit, so a
    frame with no tree pixels never reports a ceiling.
    """
    empty = {"pixels": int(valid.sum()), "scale": None, "shift": None, "mae_m": None, "correlation": None}
    if valid.sum() < MIN_PIXELS:
        return empty
    d = disparity[valid].astype(np.float64)
    inverse = 1.0 / gt[valid].astype(np.float64)
    if d.std() < 1e-9 or inverse.std() < 1e-9:
        return empty
    scale, shift = np.polyfit(d, inverse, 1)
    fitted = scale * d + shift
    positive = fitted > 1e-6
    if positive.sum() < MIN_PIXELS:
        return dict(empty, scale=float(scale), shift=float(shift))
    mae = float(np.mean(np.abs(1.0 / fitted[positive] - gt[valid][positive])))
    correlation = float(np.corrcoef(d, inverse)[0, 1])
    return {
        "pixels": int(valid.sum()),
        "scale": float(scale),
        "shift": float(shift),
        "mae_m": mae,
        "positive_fraction": float(positive.mean()),
        "correlation": correlation,
    }


def target_window(pixel, shape, radius=1):
    if pixel is None:
        return None
    x, y = int(round(pixel[0])), int(round(pixel[1]))
    if not (0 <= x < shape[1] and 0 <= y < shape[0]):
        return None
    window = np.zeros(shape, dtype=bool)
    window[max(0, y - radius) : y + radius + 1, max(0, x - radius) : x + radius + 1] = True
    return window


def score_frame(disparity, gt, mask, pixel):
    """Frame-level and target-level ceilings from one all-GT disparity fit."""
    valid = valid_pixels(disparity, gt, mask)
    frame = disparity_affine_ceiling(disparity, gt, valid)
    row = {"frame_fit": frame, "target": None}
    window = target_window(pixel, gt.shape)
    if window is not None and frame["scale"] is not None:
        gated = window & valid_pixels(disparity, gt, mask)
        if gated.any():
            fitted = frame["scale"] * disparity[gated] + frame["shift"]
            positive = fitted > 1e-6
            if positive.any():
                pred = 1.0 / fitted[positive]
                row["target"] = {
                    "pixels": int(positive.sum()),
                    "reference_m": float(np.median(gt[gated][positive])),
                    "predicted_m": float(np.median(pred)),
                    "mae_m": float(np.mean(np.abs(pred - gt[gated][positive]))),
                }
    return row


class FrozenRelativeDA2:
    def __init__(self, companion, checkpoint, expected_sha256, device="cuda"):
        import torch

        self.torch = torch
        self.sha256 = sha256(checkpoint)
        if self.sha256 != expected_sha256:
            raise ValueError("Checkpoint hash differs from frozen plan")
        sys.path.insert(0, str(Path(companion) / "depth-anything-v2"))
        from depth_anything_v2.dpt import DepthAnythingV2

        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        self.device = device
        self.model = DepthAnythingV2(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
        state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(state.get("model", state), strict=True)
        self.model.to(device).eval()
        self.metadata = {
            "architecture": "DA2 relative ViT-L (public checkpoint; ReLU disparity head)",
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": self.sha256,
            "input": "OpenCV BGR, native size, infer_image(input_size=518)",
            "output": "affine-invariant disparity at input resolution; metres only through the all-GT fit",
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


def load_frames(spec):
    """``name=path`` where path is a plan.json or an evaluation.json with an embedded plan."""
    name, path = spec.split("=", 1)
    data = json.loads(Path(path).read_text())
    plan = data.get("plan", data)
    return name, path, sha256(path), plan["frames"]


def main(argv=None) -> int:
    import cv2
    from PIL import Image

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--frames", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "schema_version": SCHEMA_VERSION,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "scope": (
            "All-GT affine fit of relative disparity per frame: a ceiling on what any metric anchor could "
            "recover from this head, never an achievable number."
        ),
        "sources": [],
        "ok": False,
        "rows": [],
    }
    try:
        model = FrozenRelativeDA2(args.companion, args.checkpoint, args.checkpoint_sha256, args.device)
        result["model"] = model.metadata
        index = 0
        for spec in args.frames:
            name, path, digest, frames = load_frames(spec)
            result["sources"].append({"name": name, "path": path, "sha256": digest, "frames": len(frames)})
            for frame in frames:
                image = cv2.imread(frame["rgb"])
                if image is None:
                    raise ValueError(f"RGB missing: {frame['rgb']}")
                gt = np.load(frame["depth"], allow_pickle=False).astype(np.float32)
                if image.shape[:2] != gt.shape:
                    raise ValueError("Native RGB and GT dimensions differ")
                mask = None
                if frame.get("mask"):
                    mask = np.asarray(Image.open(frame["mask"])) > 0
                    if mask.ndim == 3:
                        mask = mask.any(axis=2)
                if index == 0:
                    model.predict(image)
                disparity, latency = model.predict(image)
                pixel = frame.get("target_pixel_xy") if frame.get("target_visible") is not False else None
                row = score_frame(disparity, gt, mask, pixel)
                prediction = args.output / f"disparity_{index:05d}.npy"
                np.save(prediction, disparity, allow_pickle=False)
                row.update(
                    {
                        "index": index,
                        "source": name,
                        "frame": frame,
                        "mask_gated": mask is not None,
                        "inference_seconds": latency,
                        "prediction": str(prediction),
                        "rgb_sha256": sha256(frame["rgb"]),
                    }
                )
                result["rows"].append(row)
                print(json.dumps({"index": index, "source": name, "frame_fit_mae_m": row["frame_fit"]["mae_m"]}))
                index += 1
        result["ok"] = True
    except Exception:
        import traceback

        result["error"] = traceback.format_exc()
        raise
    finally:
        (args.output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
