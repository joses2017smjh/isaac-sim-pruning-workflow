#!/usr/bin/env python3
"""Repository-local warm-start fine-tune of the DA2 metric head with photometric jitter.

The companion trainer cannot be reused as-is: it has no version control, its
``--pretrained-from`` discards the trained head and its ``--resume`` continues
a finished schedule, and its wandb import no longer loads. This wrapper imports
the companion's dataset and model classes by path, copies the loss, the metric
and the optimizer groups verbatim, loads the full checkpoint with a fresh
schedule, and adds one thing the original run never had: photometric jitter
after the resize and before ImageNet normalization. Everything that changes
against the original recipe is recorded in the result document.

Two arms share every setting except the jitter, so the comparison isolates it:
``jitter`` applies the augmentation, ``control`` does not.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

SCHEMA_VERSION = 1

#: Fixed before any run. Brightness covers the 2.6x darkening of the evening
#: preset in both directions; gamma covers the tone curve; the colour gains
#: cover the warm evening cast; nothing is tuned to a result.
JITTER = {
    "probability": 0.8,
    "brightness_range": (0.35, 1.3),
    "gamma_range": (0.6, 1.6),
    "channel_gain_range": (0.8, 1.2),
    "note": "Applied to the resized float RGB in [0, 1] before ImageNet normalization; clipped to [0, 1]",
}

MODEL_CONFIG = {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def photometric_jitter(image, rng, params=JITTER):
    """Return the jittered image (float32 HxWx3 in [0, 1]) and the parameters drawn."""
    image = np.asarray(image, dtype=np.float32)
    if rng.random() >= params["probability"]:
        return image, {"applied": False}
    gain = float(rng.uniform(*params["brightness_range"]))
    gamma = float(rng.uniform(*params["gamma_range"]))
    channels = [float(rng.uniform(*params["channel_gain_range"])) for _ in range(3)]
    out = np.clip(image, 0.0, 1.0) ** gamma
    out = out * gain * np.asarray(channels, dtype=np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32), {
        "applied": True,
        "brightness": gain,
        "gamma": gamma,
        "channel_gains": channels,
    }


def filter_manifest(source_csv, output_csv):
    """Keep the rows whose three files exist; return counts. Never rewrites the companion's file."""
    kept, dropped = 0, 0
    with Path(source_csv).open(newline="") as src, Path(output_csv).open("w", newline="") as dst:
        reader = csv.DictReader(src)
        writer = csv.DictWriter(dst, fieldnames=reader.fieldnames)
        writer.writeheader()
        for row in reader:
            if all(Path(row[key]).is_file() for key in ("rgb_path", "depth_path", "mask_path")):
                writer.writerow(row)
                kept += 1
            else:
                dropped += 1
    return {"source": str(source_csv), "source_sha256": sha256(source_csv), "kept": kept, "dropped": dropped}


def plan_frames_to_manifest(plan_path, output_csv):
    """List an evaluation plan's frames as a val manifest (logged only, never used for selection)."""
    plan = json.loads(Path(plan_path).read_text())
    with Path(output_csv).open("w", newline="") as dst:
        writer = csv.DictWriter(dst, fieldnames=["rgb_path", "depth_path", "mask_path", "condition"])
        writer.writeheader()
        for frame in plan["frames"]:
            if frame.get("mask"):
                writer.writerow(
                    {
                        "rgb_path": frame["rgb"],
                        "depth_path": frame["depth"],
                        "mask_path": frame["mask"],
                        "condition": frame.get("condition", frame.get("lighting")),
                    }
                )
    return {"source": str(plan_path), "source_sha256": sha256(plan_path)}


class SiLogLoss:
    """Scale-invariant log loss, verbatim from the companion (metric_depth/util/loss.py)."""

    def __init__(self, torch, lambd=0.5):
        self.torch = torch
        self.lambd = lambd

    def __call__(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = self.torch.log(target[valid_mask]) - self.torch.log(pred[valid_mask])
        return self.torch.sqrt(self.torch.pow(diff_log, 2).mean() - self.lambd * self.torch.pow(diff_log.mean(), 2))


def eval_depth(torch, pred, target):
    """Verbatim from the companion (metric_depth/util/metric.py)."""
    thresh = torch.max((target / pred), (pred / target))
    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)
    return {
        "d1": (torch.sum(thresh < 1.25).float() / len(thresh)).item(),
        "abs_rel": torch.mean(torch.abs(diff) / target).item(),
        "rmse": torch.sqrt(torch.mean(torch.pow(diff, 2))).item(),
        "rmse_log": torch.sqrt(torch.mean(torch.pow(diff_log, 2))).item(),
        "silog": torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2)).item(),
    }


def install_jitter(dataset, size, seed):
    """Rebuild the companion's train pipeline with the jitter between resize and normalization."""
    import cv2
    from dataset.trunk_da2 import Crop
    from depth_anything_v2.util.transform import NormalizeImage, PrepareForNet, Resize

    resize = Resize(
        width=size[1],
        height=size[0],
        keep_aspect_ratio=True,
        ensure_multiple_of=14,
        resize_method="lower_bound",
        image_interpolation_method=cv2.INTER_CUBIC,
    )
    normalize = NormalizeImage(mean=dataset.MEAN, std=dataset.STD)
    prepare = PrepareForNet()
    crop = Crop(size)
    state = {"rng": np.random.default_rng(seed), "applied": 0, "seen": 0}

    def transform(sample):
        sample = resize(sample)
        info = os.environ.get("FINETUNE_WORKER_SEED")
        rng = state["rng"] if info is None else np.random.default_rng(int(info) + state["seen"])
        sample["image"], drawn = photometric_jitter(sample["image"], rng)
        state["seen"] += 1
        state["applied"] += int(drawn["applied"])
        return crop(prepare(normalize(sample)))

    dataset.transform = transform
    return state


def worker_init(worker_id):
    import torch

    seed = (torch.initial_seed() + worker_id) % (2**32)
    os.environ["FINETUNE_WORKER_SEED"] = str(seed)
    np.random.seed(seed)
    random.seed(seed)


def validate(torch, model, loader, device, min_depth, max_depth):
    model.eval()
    totals, count = {}, 0
    with torch.no_grad():
        for sample in loader:
            image, depth, valid = (
                sample["image"].to(device),
                sample["depth"].to(device)[0],
                sample["valid_mask"].to(device)[0],
            )
            pred = model(image)
            pred = torch.nn.functional.interpolate(
                pred[:, None], depth.shape[-2:], mode="bilinear", align_corners=True
            )[0, 0]
            mask = (valid == 1) & (depth >= min_depth) & (depth <= max_depth)
            if mask.sum() < 10:
                continue
            metrics = eval_depth(torch, pred[mask], depth[mask])
            for key, value in metrics.items():
                totals[key] = totals.get(key, 0.0) + value
            count += 1
    model.train()
    return {key: value / count for key, value in totals.items()} | {"frames": count} if count else {"frames": 0}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Full companion checkpoint (model + optimizer)")
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--val-manifest", type=Path, required=True)
    parser.add_argument("--extra-val-manifest", type=Path, action="append", default=[], help="Logged only")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--arm", choices=("jitter", "control"), required=True)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--max-runtime-seconds", type=int, default=0)
    parser.add_argument("--min-depth", type=float, default=0.001)
    parser.add_argument("--max-depth", type=float, default=20.0)
    parser.add_argument("--smoke-iterations", type=int, default=0, help="Stop each epoch after N iterations")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)

    import torch
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(args.companion))
    sys.path.insert(0, str(args.companion / "depth-anything-v2/metric_depth"))
    from dataset.trunk_da2 import TrunkDA2
    from depth_anything_v2.dpt import DepthAnythingV2

    started = time.time()
    if sha256(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint hash differs from frozen plan")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")

    size = (518, 518)
    trainset = TrunkDA2(str(args.train_manifest), mode="train", size=size)
    valset = TrunkDA2(str(args.val_manifest), mode="val", size=size)
    extras = [(path, TrunkDA2(str(path), mode="val", size=size)) for path in args.extra_val_manifest]
    jitter_state = install_jitter(trainset, size, args.seed) if args.arm == "jitter" else None
    train_loader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        worker_init_fn=worker_init,
    )
    val_loader = DataLoader(valset, batch_size=1, num_workers=args.num_workers, pin_memory=True)
    extra_loaders = [(path, DataLoader(ds, batch_size=1, num_workers=args.num_workers)) for path, ds in extras]

    model = DepthAnythingV2(**MODEL_CONFIG, max_depth=args.max_depth)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True, mmap=True)
    model.load_state_dict(state.get("model", state), strict=True)
    del state
    model.to(device).train()
    criterion = SiLogLoss(torch)
    optimizer = torch.optim.AdamW(
        [
            {"params": [p for n, p in model.named_parameters() if "pretrained" in n], "lr": args.lr},
            {"params": [p for n, p in model.named_parameters() if "pretrained" not in n], "lr": args.lr * 10.0},
        ],
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )
    total_iters = args.epochs * len(train_loader)
    result = {
        "schema_version": SCHEMA_VERSION,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "arm": args.arm,
        "jitter": JITTER if args.arm == "jitter" else None,
        "warm_start": {
            "checkpoint": str(args.checkpoint),
            "checkpoint_sha256": args.checkpoint_sha256,
            "loaded": "model weights only, strict; fresh AdamW and polynomial schedule from iteration 0",
        },
        "recipe_kept_from_the_original": [
            "SiLog lambda 0.5 on the trunk mask within [min_depth, max_depth]",
            "AdamW betas (0.9, 0.999), weight decay 0.01, head learning rate 10x backbone",
            "polynomial decay power 0.9, FP32, batch 2, 518x518 random crop, 50% horizontal flip",
        ],
        "recipe_changed": [
            "warm start from the fine-tuned checkpoint instead of the relative backbone",
            f"{args.epochs} epochs instead of 14",
            "6,270 surviving frames (box and box_cam1-4, bark_brown_02) instead of 68,400",
            "photometric jitter (jitter arm only)",
        ],
        "train_rows": len(trainset),
        "val_rows": len(valset),
        "extra_val": [str(p) for p, _ in extras],
        "epochs": [],
        "ok": False,
    }
    best = None
    iteration = 0
    stop = False
    try:
        for epoch in range(args.epochs):
            epoch_loss, seen = 0.0, 0
            for sample in train_loader:
                image, depth, valid = (sample[k].to(device) for k in ("image", "depth", "valid_mask"))
                if random.random() < 0.5:
                    image, depth, valid = image.flip(-1), depth.flip(-1), valid.flip(-1)
                pred = model(image)
                mask = (valid == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
                if mask.sum() == 0:
                    continue
                loss = criterion(pred, depth, mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iteration += 1
                lr = args.lr * (1 - iteration / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * 10.0
                epoch_loss += float(loss.item())
                seen += 1
                if seen % 200 == 0:
                    print(json.dumps({"epoch": epoch, "iteration": iteration, "loss": epoch_loss / seen}), flush=True)
                if args.smoke_iterations and seen >= args.smoke_iterations:
                    break
                if args.max_runtime_seconds and time.time() - started > args.max_runtime_seconds:
                    stop = True
                    break
            record = {
                "epoch": epoch,
                "iterations": seen,
                "train_loss": epoch_loss / max(seen, 1),
                "val": validate(torch, model, val_loader, device, args.min_depth, args.max_depth),
                "extra_val": {
                    str(path): validate(torch, model, loader, device, args.min_depth, args.max_depth)
                    for path, loader in extra_loaders
                },
                "jitter_applied_fraction": (
                    jitter_state["applied"] / max(jitter_state["seen"], 1) if jitter_state else None
                ),
                "elapsed_s": time.time() - started,
            }
            result["epochs"].append(record)
            print(json.dumps({k: v for k, v in record.items() if k != "extra_val"}), flush=True)
            torch.save({"model": model.state_dict(), "epoch": epoch}, args.output / "last.pth")
            rmse = record["val"].get("rmse")
            if rmse is not None and (best is None or rmse < best):
                best = rmse
                torch.save({"model": model.state_dict(), "epoch": epoch}, args.output / "best.pth")
                result["best"] = {"epoch": epoch, "val_rmse": rmse}
            if stop:
                result["stopped_by_runtime_limit"] = True
                break
        result["ok"] = True
    except Exception:
        import traceback

        result["error"] = traceback.format_exc()
        raise
    finally:
        for name in ("best.pth", "last.pth"):
            path = args.output / name
            if path.is_file():
                result[f"{name}_sha256"] = sha256(path)
        result["elapsed_s"] = time.time() - started
        (args.output / "training.json").write_text(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
