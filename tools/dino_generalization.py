"""Frozen six-view RGB+D comparator with DA2 inputs, never GT depth inputs."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from depth_generalization import depth_metrics, sha256, target_metrics

DINO_SHA = "2750c4d8f5db7c0d5d40138f08ee9d4bc39a1b184063d33f3dfe9386438ac8ca"
DINO_REL = "checkpoints/dino_da2ft_3pair_fusion_nopose_spur_seed1/exp1/seed_01/best_epoch_0023.pt"


class FrozenDINO:
    def __init__(self, companion, device="cuda"):
        import torch

        self.torch, self.device = torch, device
        checkpoint = companion / DINO_REL
        if sha256(checkpoint) != DINO_SHA:
            raise ValueError("DINO checkpoint hash changed")
        sys.path.insert(0, str(companion / "MVP_MODEL"))
        from mvp_stereo_dino_model import MVStereoDINOUNet

        # The full checkpoint contains the backbone. Construct its exact cached
        # architecture without downloading or redundantly loading pretrained weights.
        local_hub = companion.parent / ".cache/torch/hub/facebookresearch_dinov2_main"
        original = torch.hub.load

        def local_load(repo, model, **kwargs):
            if repo != "facebookresearch/dinov2" or model != "dinov2_vitl14":
                raise ValueError("Unexpected backbone request")
            return original(str(local_hub), model, source="local", pretrained=False)

        torch.hub.load = local_load
        try:
            self.model = MVStereoDINOUNet(
                n_views=6, use_pose=False, no_fusion=False, use_plucker=False, pred_mode="absolute"
            )
        finally:
            torch.hub.load = original
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        self.model.load_state_dict(ckpt["model"], strict=True)
        self.model.to(device).eval()
        self.metadata = {
            "checkpoint": str(checkpoint),
            "checkpoint_sha256": DINO_SHA,
            "args": ckpt["args"],
            "torch": torch.__version__,
            "device": device,
            "source_stub_sha256": sha256(companion / "MVP_MODEL/mvp_depth_refine_model.py"),
            "bytecode_sha256": sha256(companion / "MVP_MODEL/_mvp_precompiled.pyc"),
            "limitation": "Encoder source is missing locally; trusted existing Python 3.10 bytecode loaded and hashed",
            "input": (
                "Six synchronized RGB views in rig0 L/R, rig1 L/R, rig2 L/R order; ImageNet normalize;  "
                "DA2 metric depth; no GT scale calibration"
            ),
            "size_hw": [280, 512],
            "pose": "Disabled by checkpoint; matrices unused",
            "latency_scope": "Six-view DINO forward only; acquisition and six DA2 forwards add latency",
        }

    def predict(self, images, depths):
        from PIL import Image

        torch = self.torch
        import torch.nn.functional as F

        if len(images) != 6 or len(depths) != 6:
            raise ValueError("This checkpoint requires six views")
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        rgb = [
            (
                (
                    np.asarray(Image.fromarray(im).resize((512, 280), Image.Resampling.BILINEAR), dtype=np.float32)
                    / 255
                    - mean
                )
                / std
            ).transpose(2, 0, 1)
            for im in images
        ]
        ds = [
            F.interpolate(
                torch.from_numpy(d.astype(np.float32))[None, None],
                size=(280, 512),
                mode="bilinear",
                align_corners=False,
            )[0]
            for d in depths
        ]
        rgb = torch.from_numpy(np.stack(rgb))[None].to(self.device)
        d = torch.stack(ds)[None].to(self.device)
        with torch.inference_mode():
            if self.device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            out = self.model(rgb, d)
            if self.device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            predictions = [
                F.interpolate(out[:, i], size=images[i].shape[:2], mode="bilinear", align_corners=False)[0, 0]
                .cpu()
                .numpy()
                for i in range(6)
            ]
        return predictions, elapsed


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--companion", type=Path, required=True)
    p.add_argument("--da2-result", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    result = {
        "ok": False,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "rows": [],
        "smoke_only": args.smoke,
        "da2_result_sha256": sha256(args.da2_result),
    }
    try:
        from PIL import Image

        model = FrozenDINO(args.companion)
        result["model"] = model.metadata
        source = json.loads(args.da2_result.read_text())
        assert source["ok"]
        groups = {}
        for row in source["rows"]:
            f = row["frame"]
            key = (f.get("tree_id", "smoke"), f.get("lighting", "source"))
            groups.setdefault(key, []).append(row)
        if args.smoke:
            groups = {("smoke", "repeated_single_view"): [source["rows"][0]] * 6}
        for key, rows in groups.items():
            if len(rows) != 6:
                raise ValueError(f"Expected six paired views: {key} has {len(rows)}")
            rows = sorted(rows, key=lambda r: r["frame"].get("view_id", ""))
            images = [np.asarray(Image.open(r["frame"]["rgb"]).convert("RGB")) for r in rows]
            inputs = [np.load(r["prediction"], allow_pickle=False) for r in rows]
            if not result["rows"]:
                model.predict(images, inputs)
            predictions, latency = model.predict(images, inputs)
            for row, pred in zip(rows, predictions):
                index = len(result["rows"])
                f = row["frame"]
                gt = np.load(f["depth"], allow_pickle=False)
                mask = np.asarray(Image.open(f["mask"])) > 0 if f.get("mask") else None
                if mask is not None and mask.ndim == 3:
                    mask = mask.any(axis=2)
                target = target_metrics(pred, gt, f.get("target_pixel_xy"))
                if f.get("target_visible") is False:
                    target = {"valid": False, "reason": "geometric_target_not_visible"}
                path = args.output / f"prediction_{index:05d}.npy"
                np.save(path, pred)
                result["rows"].append(
                    {
                        "frame": f,
                        "all_valid_gt": depth_metrics(pred, gt),
                        "mask_metrics": depth_metrics(pred, gt, mask) if mask is not None else None,
                        "target": target,
                        "prediction": str(path),
                        "group": list(key),
                        "six_view_dino_seconds": latency,
                        "six_view_total_inference_seconds": latency + sum(r["inference_seconds"] for r in rows),
                    }
                )
            print(json.dumps({"group": key, "six_view_seconds": latency}), flush=True)
        result["ok"] = True
    except Exception:
        import traceback

        result["error"] = traceback.format_exc()
        raise
    finally:
        (args.output / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
