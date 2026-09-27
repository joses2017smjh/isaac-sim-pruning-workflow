#!/usr/bin/env python3
"""Plan or submit the frozen rendered-lighting training render: a one-tree pilot or the full train split.

Same discipline as the other launchers: committed source only, a clean tracked
tree, every tree and frame registered in the plan before the only scheduler
mutation, intent and receipt files, and a storage preflight against the share's
real quota. The full render reuses the pilot's tree when the pilot's manifest is
passed, and splits the array by frames per tree so each task's time limit fits
its own tree. Frames per task are rendered one at a time on one GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import training_lighting as tl  # noqa: E402
from queue_family_matrix import COMPANION, _sbatch, sha256, storage_preflight  # noqa: E402
from queue_vision_robustness import _git, dependency_option, freeze_source, submission_environment  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TRAIN_MANIFEST = COMPANION / "manifests/full_spur_2tex_all_3view_seed1/spur_train.csv"
PILOT_TREE = "lpy_envy_00001"
#: Measured 21.55 s per 1920x1080 frame on a V100 for the August render; the A40 and RTX 8000 rates are
#: unmeasured, so the pilot reserves 50 s per frame and the full array uses the pilot's measured mean x 2.
DEFAULT_SECONDS_PER_FRAME = 50.0
SETUP_SECONDS = 300
PILOT_SOURCE_CHECKS = 2
#: Mean 1920x1080 RGB PNG of the surviving frames, 2.663 MB, plus manifest headroom.
BYTES_PER_FRAME = 3 * 1024 * 1024


def task_minutes(frames, seconds_per_frame, extra_renders=0):
    """Slurm minutes for one tree: every render at the reserved rate plus setup, rounded up to 10 minutes."""
    seconds = (frames + extra_renders) * seconds_per_frame + SETUP_SECONDS
    return int(math.ceil(seconds / 600.0) * 10)


def pilot_rate(pilot_manifest):
    """Mean measured seconds per frame in a finished pilot render, or None."""
    data = json.loads(Path(pilot_manifest).read_text())
    seconds = [f["render_seconds"] for f in data.get("frames", []) if f.get("render_seconds")]
    return (sum(seconds) / len(seconds)) if seconds and not data.get("smoke") else None


def training_plan(mode, *, manifest=TRAIN_MANIFEST, pilot_manifest=None, companion=COMPANION, hash_assets=True):
    if mode not in ("pilot", "full"):
        raise ValueError("mode must be pilot or full")
    grouped = tl.frames_by_tree(tl.surviving_rows(manifest))
    if mode == "pilot":
        if PILOT_TREE not in grouped:
            raise ValueError(f"Pilot tree {PILOT_TREE} has no surviving train frames")
        grouped = {PILOT_TREE: grouped[PILOT_TREE]}
    seconds_per_frame = DEFAULT_SECONDS_PER_FRAME
    trees = {}
    for tree_id, frames in sorted(grouped.items()):
        entry = {"frames": frames, "status": "to_render"}
        if mode == "full" and pilot_manifest is not None and tree_id == PILOT_TREE:
            # The pilot job renders this tree; its manifest may not exist yet when the full array is chained
            # behind the pilot. It is hashed now if present and by the manifest builder otherwise.
            entry = {"frames": frames, "status": "rendered_pilot", "render_manifest": str(pilot_manifest)}
            if Path(pilot_manifest).is_file():
                if hash_assets:
                    entry["render_manifest_sha256"] = sha256(pilot_manifest)
            else:
                entry["render_manifest_pending"] = True
        trees[tree_id] = entry
    if mode == "full" and pilot_manifest is not None and Path(pilot_manifest).is_file():
        measured = pilot_rate(pilot_manifest)
        if measured is not None:
            seconds_per_frame = max(DEFAULT_SECONDS_PER_FRAME / 2, 2.0 * measured)
    extra = PILOT_SOURCE_CHECKS if mode == "pilot" else 0
    arrays = {}
    for tree_id, entry in trees.items():
        if entry["status"] == "to_render":
            arrays.setdefault(len(entry["frames"]), []).append(tree_id)
    arrays = [
        {"frames_per_tree": n, "trees": ids, "minutes_per_task": task_minutes(n, seconds_per_frame, extra)}
        for n, ids in sorted(arrays.items())
    ]
    to_render = sum(len(trees[t]["frames"]) for a in arrays for t in a["trees"])
    plan = {
        "schema_version": 1,
        "scope": (
            "Synthetic training data: the companion's surviving Envy train-split frames re-rendered at their exact "
            "recorded poses under one seeded lighting draw per frame (rendered-lighting-v1), RGB only; depth and "
            "masks are reused from the surviving files after a per-frame geometry agreement check. No matrix, "
            "validation or UFO tree is rendered."
        ),
        "protocol": "docs/EVAL_PROTOCOL_LIGHTING_TRAINING_2026-09-27.md",
        "mode": mode,
        "sampler": {
            "version": tl.SAMPLER_VERSION,
            "regimes": tl.REGIMES,
            "azimuth_deg": tl.AZIMUTH_DEG,
            "sun_energy": tl.SUN_ENERGY,
            "sun_angle_deg": tl.SUN_ANGLE_DEG,
            "holdout_radius_deg": tl.HOLDOUT_RADIUS_DEG,
            "preset_suns": tl.PRESET_SUNS,
        },
        "agreement": {
            "tolerance_m": tl.AGREEMENT_TOLERANCE_M,
            "min_fraction": tl.AGREEMENT_MIN_FRACTION,
            "min_iou": tl.AGREEMENT_MIN_IOU,
        },
        "train_manifest": str(manifest),
        "seconds_per_frame_reserved": seconds_per_frame,
        "source_checks": extra,
        "trees": trees,
        "arrays": arrays,
        "registered_frames": sum(len(e["frames"]) for e in trees.values()),
        "frames_to_render": to_render,
        "max_concurrent_gpus": 1,
        "maximum_gpu_minutes": sum(a["minutes_per_task"] * len(a["trees"]) for a in arrays),
        "estimated_output_bytes": to_render * BYTES_PER_FRAME,
    }
    if hash_assets:
        plan["train_manifest_sha256"] = sha256(manifest)
        for name in (
            "orchard_template.blend",
            "Dataloader/generate_tree2.py",
            "Dataloader/daylight_presets.py",
            "Dataloader/move_camera.py",
        ):
            path = companion / name
            if path.is_file():
                plan.setdefault("external_assets_sha256", {})[name] = sha256(path)
    return plan


def queue_batch(root, batch_id, mode, *, pilot_manifest=None, share_used_bytes=None, share_du_file=None, after=None):
    root = Path(root).resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", batch_id):
        raise ValueError("batch-id must be a short letters/digits/underscore/dash identifier")
    if _git(root, "diff", "HEAD", "--name-only").strip():
        raise ValueError("Commit tracked changes before freezing experiments; untracked work is preserved")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    plan = training_plan(mode, pilot_manifest=pilot_manifest)
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
        "hpc/slurm/training_lighting_render_frozen.sbatch",
        "tools/render_training_lighting.py",
        "tools/training_lighting.py",
    ):
        if name not in hashes:
            raise ValueError(f"Required runner is not committed: {name}")
    plan["source_sha256"] = hashes
    (batch / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (batch / "queue_before.txt").write_text(queue_before, encoding="utf-8")
    storage_preflight(batch, share_used_bytes, share_du_file, estimated_output_bytes=plan["estimated_output_bytes"])
    job_ids, previous = [], after
    for index, array in enumerate(plan["arrays"]):
        command = [
            "sbatch",
            "--parsable",
            "--array",
            f"0-{len(array['trees']) - 1}%1",
            "--time",
            f"{array['minutes_per_task'] // 60:02d}:{array['minutes_per_task'] % 60:02d}:00",
            *dependency_option(previous),
            "--chdir",
            str(batch / "code"),
            "--output",
            str(batch / "logs/%x-%A_%a.out"),
            "--error",
            str(batch / "logs/%x-%A_%a.out"),
            str(batch / "code/hpc/slurm/training_lighting_render_frozen.sbatch"),
            str(batch),
            str(index),
        ]
        previous = _sbatch(command, env, batch, f"render_{index}")
        job_ids.append(previous)
    return {"array_job_ids": job_ids, "batch_dir": str(batch), **{k: v for k, v in plan.items() if k != "trees"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--mode", choices=("pilot", "full"), required=True)
    parser.add_argument("--batch-id", default=None, help="Default: training-lighting-<mode>-20260927")
    parser.add_argument("--pilot-manifest", type=Path, help="Finished pilot render_manifest.json to reuse in full mode")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--share-used-bytes", type=int)
    parser.add_argument("--share-du-file", type=Path)
    parser.add_argument("--after", help="Queue behind this Slurm job id; it is never modified")
    args = parser.parse_args()
    batch_id = args.batch_id or f"training-lighting-{args.mode}-20260927"
    if args.submit:
        result = queue_batch(
            args.root,
            batch_id,
            args.mode,
            pilot_manifest=args.pilot_manifest,
            share_used_bytes=args.share_used_bytes,
            share_du_file=args.share_du_file,
            after=args.after,
        )
    else:
        plan = training_plan(args.mode, pilot_manifest=args.pilot_manifest)
        result = {k: v for k, v in plan.items() if k != "trees"}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
