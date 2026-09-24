#!/usr/bin/env python3
"""Plan or submit a frozen rescoring job: named checkpoints on named evaluation plans.

Completes an approved evaluation whose step failed, or scores a new checkpoint
on frames another job already scored, on identical bytes. Every checkpoint and
plan is pinned by hash in the plan before the only scheduler mutation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from queue_family_matrix import _sbatch, sha256, storage_preflight
from queue_vision_robustness import _git, dependency_option, freeze_source, submission_environment

ROOT = Path(__file__).resolve().parents[1]
MINUTES_RESERVED = 20
ESTIMATED_OUTPUT_BYTES = 800 * 1024 * 1024


def rescore_plan(checkpoints, plans, *, hash_assets=True):
    """``checkpoints`` and ``plans`` map names to paths; every pair is scored."""
    if not checkpoints or not plans:
        raise ValueError("A rescoring needs at least one checkpoint and one plan")
    plan = {
        "schema_version": 1,
        "scope": "Frozen DA2 metric scoring of named checkpoints on named evaluation plans; no training, no rendering.",
        "max_concurrent_gpus": 1,
        "maximum_gpu_minutes": MINUTES_RESERVED,
        "checkpoints": {name: {"path": str(path)} for name, path in checkpoints.items()},
        "plans": {name: {"path": str(path)} for name, path in plans.items()},
    }
    if hash_assets:
        for table in ("checkpoints", "plans"):
            for name, entry in plan[table].items():
                if not Path(entry["path"]).is_file():
                    raise FileNotFoundError(f"{table[:-1]} {name} is missing: {entry['path']}")
                entry["sha256"] = sha256(entry["path"])
    return plan


def queue_batch(root, batch_id, checkpoints, plans, *, share_used_bytes=None, share_du_file=None, after=None):
    root = Path(root).resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", batch_id):
        raise ValueError("batch-id must be a short letters/digits/underscore/dash identifier")
    if _git(root, "diff", "HEAD", "--name-only").strip():
        raise ValueError("Commit tracked changes before freezing experiments; untracked work is preserved")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    plan = rescore_plan(checkpoints, plans, hash_assets=after is None)
    plan["code_revision"] = revision
    plan["created_utc"] = datetime.now(timezone.utc).isoformat()
    if after is not None:
        plan["pinned_at_run_time"] = "checkpoints produced by the job this waits on are hashed by the job itself"
    env = submission_environment(os.environ)
    queue_before = subprocess.check_output(
        ["squeue", "--user", os.environ["USER"], "--noheader", "--format=%i|%j|%T|%R"], env=env, timeout=30
    ).decode()
    batch = root / "artifacts/generalization" / batch_id
    batch.mkdir(parents=True, exist_ok=False)
    (batch / "logs").mkdir()
    hashes = freeze_source(root, batch / "code", revision)
    for name in ("hpc/slurm/rescore_frozen.sbatch", "tools/depth_generalization.py"):
        if name not in hashes:
            raise ValueError(f"Required runner is not committed: {name}")
    plan["source_sha256"] = hashes
    (batch / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (batch / "queue_before.txt").write_text(queue_before, encoding="utf-8")
    storage_preflight(batch, share_used_bytes, share_du_file, estimated_output_bytes=ESTIMATED_OUTPUT_BYTES)
    command = [
        "sbatch",
        "--parsable",
        *dependency_option(after),
        "--chdir",
        str(batch / "code"),
        "--output",
        str(batch / "logs/%x-%j.out"),
        "--error",
        str(batch / "logs/%x-%j.out"),
        str(batch / "code/hpc/slurm/rescore_frozen.sbatch"),
        str(batch),
    ]
    job_id = _sbatch(command, env, batch, "rescore")
    return {"job_id": job_id, "batch_dir": str(batch), **plan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--checkpoint", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--plan", action="append", required=True, metavar="NAME=PATH")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--share-used-bytes", type=int)
    parser.add_argument("--share-du-file", type=Path)
    parser.add_argument("--after", help="Queue behind this Slurm job id; it is never modified")
    args = parser.parse_args()
    checkpoints = dict(spec.split("=", 1) for spec in args.checkpoint)
    plans = dict(spec.split("=", 1) for spec in args.plan)
    if args.submit:
        result = queue_batch(
            args.root,
            args.batch_id,
            checkpoints,
            plans,
            share_used_bytes=args.share_used_bytes,
            share_du_file=args.share_du_file,
            after=args.after,
        )
    else:
        result = rescore_plan(checkpoints, plans, hash_assets=False)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
