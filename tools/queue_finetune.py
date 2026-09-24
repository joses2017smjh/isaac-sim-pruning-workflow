#!/usr/bin/env python3
"""Plan or submit one frozen fine-tune arm (jitter or control) and its evaluation.

Same discipline as the other launchers. What is pinned before submission: the
warm-start checkpoint hash, the companion trainer files the wrapper copies
from (by hash), the filtered data manifests (written into the batch with row
counts and hashes), the epochs, and the three evaluation plans the result is
scored on. The fine-tuned checkpoint's own hash is unknown until it exists; the
evaluation job records it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from finetune_da2 import JITTER, filter_manifest, plan_frames_to_manifest, sha256  # noqa: E402
from queue_family_matrix import COMPANION, MODELS, _sbatch, storage_preflight  # noqa: E402
from queue_vision_robustness import _git, dependency_option, freeze_source, submission_environment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("jitter", "control")
EPOCHS = 6
TRAIN_HOURS_RESERVED = 8
EVAL_MINUTES_RESERVED = 60
MAX_RUNTIME_SECONDS = 7 * 3600  # leaves time for the last validation and save inside the reservation
ESTIMATED_OUTPUT_BYTES = 4_500 * 1024 * 1024  # two model-only checkpoints, 1,480 saved predictions
COMPANION_FILES = (
    "train_depth_da2.py",
    "dataset/trunk_da2.py",
    "depth-anything-v2/metric_depth/depth_anything_v2/dpt.py",
)
MANIFEST_DIR = COMPANION / "manifests/full_spur_2tex_all_3view_seed1"
EVALUATION_PLANS = {
    "family_matrix": ROOT / "artifacts/generalization/family-matrix-20260923/eval/plan.json",
    "controls": ROOT / "artifacts/generalization/generalization-controls-20260923/eval/plan.json",
    "stage_a": ROOT / "artifacts/generalization/phase-20260920/stage_a_job_21370005/evaluation.json",
}


def finetune_plan(arm, *, hash_assets=True, companion=COMPANION, evaluation_plans=EVALUATION_PLANS):
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    plan = {
        "schema_version": 1,
        "scope": (
            "Warm-start fine-tune of the DA2 metric head on the 6,270 surviving Envy training frames, with "
            "(jitter arm) or without (control arm) photometric jitter, scored afterwards on the frozen "
            "matrix, controls and Isaac Stage A plans. Not a new dataset, not a new architecture."
        ),
        "protocol": "docs/EVAL_PROTOCOL_FINETUNE_2026-09-23.md",
        "arm": arm,
        "jitter": JITTER if arm == "jitter" else None,
        "epochs": EPOCHS,
        "max_runtime_seconds": MAX_RUNTIME_SECONDS,
        "max_concurrent_gpus": 1,
        "maximum_gpu_minutes": TRAIN_HOURS_RESERVED * 60 + EVAL_MINUTES_RESERVED,
        "warm_start": dict(MODELS["da2"]),
        "evaluation_plans": {name: {"path": str(path)} for name, path in evaluation_plans.items()},
        "selection": (
            "best.pth by the companion validation split's RMSE; the family-matrix frames are logged, never selected on"
        ),
    }
    if hash_assets:
        for name in COMPANION_FILES:
            path = companion / name
            if not path.is_file():
                raise FileNotFoundError(f"Companion file missing: {path}")
            plan.setdefault("companion_sha256", {})[name] = sha256(path)
        for name, entry in plan["evaluation_plans"].items():
            if not Path(entry["path"]).is_file():
                raise FileNotFoundError(f"Evaluation plan missing: {entry['path']}")
            entry["sha256"] = sha256(entry["path"])
    return plan


def write_manifests(batch, plan, manifest_dir=MANIFEST_DIR, family_plan=EVALUATION_PLANS["family_matrix"]):
    out = batch / "manifests"
    out.mkdir()
    manifests = {
        "train": filter_manifest(manifest_dir / "spur_train.csv", out / "train.csv"),
        "val": filter_manifest(manifest_dir / "spur_val.csv", out / "val.csv"),
        "family_matrix": plan_frames_to_manifest(family_plan, out / "family_matrix.csv"),
    }
    for name, entry in manifests.items():
        entry["path"] = str(out / f"{name}.csv")
        entry["sha256"] = sha256(entry["path"])
    if manifests["train"]["kept"] == 0 or manifests["val"]["kept"] == 0:
        raise ValueError("The filtered manifests are empty; the surviving frames were not found")
    plan["manifests"] = manifests
    return manifests


def queue_batch(root, batch_id, arm, *, share_used_bytes=None, share_du_file=None, after=None):
    root = Path(root).resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", batch_id):
        raise ValueError("batch-id must be a short letters/digits/underscore/dash identifier")
    if _git(root, "diff", "HEAD", "--name-only").strip():
        raise ValueError("Commit tracked changes before freezing experiments; untracked work is preserved")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    plan = finetune_plan(arm)
    plan["code_revision"] = revision
    plan["created_utc"] = datetime.now(timezone.utc).isoformat()
    env = submission_environment(os.environ)
    queue_before = subprocess.check_output(
        ["squeue", "--user", os.environ["USER"], "--noheader", "--format=%i|%j|%T|%R"], env=env, timeout=30
    ).decode()
    batch = root / "artifacts/generalization" / batch_id
    batch.mkdir(parents=True, exist_ok=False)
    (batch / "logs").mkdir()
    hashes = freeze_source(root, batch / "code", revision)
    for name in (
        "hpc/slurm/finetune_da2_frozen.sbatch",
        "hpc/slurm/finetune_eval_frozen.sbatch",
        "tools/finetune_da2.py",
        "tools/depth_generalization.py",
    ):
        if name not in hashes:
            raise ValueError(f"Required runner is not committed: {name}")
    plan["source_sha256"] = hashes
    write_manifests(batch, plan)
    (batch / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (batch / "queue_before.txt").write_text(queue_before, encoding="utf-8")
    storage_preflight(batch, share_used_bytes, share_du_file, estimated_output_bytes=ESTIMATED_OUTPUT_BYTES)
    train_command = [
        "sbatch",
        "--parsable",
        *dependency_option(after),
        "--chdir",
        str(batch / "code"),
        "--output",
        str(batch / "logs/%x-%j.out"),
        "--error",
        str(batch / "logs/%x-%j.out"),
        str(batch / "code/hpc/slurm/finetune_da2_frozen.sbatch"),
        str(batch),
    ]
    train_id = _sbatch(train_command, env, batch, "train")
    eval_command = [
        "sbatch",
        "--parsable",
        *dependency_option(train_id),
        "--chdir",
        str(batch / "code"),
        "--output",
        str(batch / "logs/%x-%j.out"),
        "--error",
        str(batch / "logs/%x-%j.out"),
        str(batch / "code/hpc/slurm/finetune_eval_frozen.sbatch"),
        str(batch),
    ]
    eval_id = _sbatch(eval_command, env, batch, "eval")
    return {"train_job_id": train_id, "eval_job_id": eval_id, "batch_dir": str(batch), **plan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--batch-id", default=None, help="Default: finetune-<arm>-20260923")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--share-used-bytes", type=int)
    parser.add_argument("--share-du-file", type=Path)
    parser.add_argument("--after", help="Queue the training job behind this Slurm job id; it is never modified")
    args = parser.parse_args()
    batch_id = args.batch_id or f"finetune-{args.arm}-20260923"
    if args.submit:
        result = queue_batch(
            args.root,
            batch_id,
            args.arm,
            share_used_bytes=args.share_used_bytes,
            share_du_file=args.share_du_file,
            after=args.after,
        )
    else:
        result = finetune_plan(args.arm)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
