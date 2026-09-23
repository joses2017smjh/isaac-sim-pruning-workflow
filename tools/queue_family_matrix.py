#!/usr/bin/env python3
"""Plan or submit the frozen Envy/UFO lighting matrix without modifying existing jobs.

Same discipline as ``queue_vision_robustness.py``, whose freezer this reuses:
committed source only, a clean tracked tree, a plan written before the only
scheduler mutation, intent recorded before the call and the receipt after it.
The render array is sized from the plan, and the evaluation job waits on it
with ``afterany`` so a failed render leaves a visible gap rather than a job
pending forever.

Two things this deliberately does not do. It never touches ``venv-isaac60`` or
the companion Computer_Vision tree, which it only reads and hashes. And it does
not choose trees: the register is fixed in the plan from the September 20
provenance audit, and the pilot trees already rendered are reused rather than
re-rendered.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from queue_vision_robustness import _git, dependency_option, freeze_source, submission_environment
from research_storage import assess

ROOT = Path(__file__).resolve().parents[1]
COMPANION = Path(f"/nfs/hpc/share/{os.environ.get('USER', 'sanchej7')}/Computer_Vision")
GENERALIZATION = ROOT / "artifacts/generalization/phase-20260920"

#: Registered on September 20 (docs/evidence/checkpoint_provenance_2026-09-20.json):
#: chosen_envy and chosen_ufo. The two 00000 trees were rendered by pilot jobs
#: 21370027 and 21370028 and are reused; the other six are rendered here.
PILOT_MANIFESTS = {
    "lpy_envy_00000": GENERALIZATION / "family_code_v2/pilot_lpy_envy_00000_job_21370027/render_manifest.json",
    "lpy_ufo_00000": GENERALIZATION / "family_code_v2/pilot_lpy_ufo_00000_job_21370028/render_manifest.json",
}
MATRIX_TREES = ("lpy_envy_00003", "lpy_envy_00008", "lpy_envy_00012", "lpy_ufo_00001", "lpy_ufo_00002", "lpy_ufo_00003")

MODELS = {
    "da2": {
        "checkpoint": str(COMPANION / "checkpoints/full_spur_2tex_all_3view_seed1/best.pth"),
        "checkpoint_sha256": "5ecc5182a2717e14f65dbb9d2d400aac6927435f6e92e87b7e891fb8f7a82834",
        "note": "DA2 metric ViT-L, fine-tuned on Envy; every Envy tree is in its train or validation split",
    },
    "dino": {
        "checkpoint": str(
            COMPANION / "checkpoints/dino_da2ft_3pair_fusion_nopose_spur_seed1/exp1/seed_01/best_epoch_0023.pt"
        ),
        "checkpoint_sha256": "2750c4d8f5db7c0d5d40138f08ee9d4bc39a1b184063d33f3dfe9386438ac8ca",
        "note": "Six-view DINO refiner over DA2 inputs; loaded by dino_generalization.py and checked by the aggregator",
    },
}

# Measured on the pilots: 63 s and 55 s per tree render, 60 s to score 48 frames.
RENDER_MINUTES_RESERVED = 15
EVAL_MINUTES_RESERVED = 30
ESTIMATED_OUTPUT_BYTES = 400 * 1024 * 1024  # renders ~50 MB, predictions ~230 MB, headroom


def sha256(path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def family_of(tree_id: str) -> str:
    for name in ("envy", "ufo"):
        if f"lpy_{name}_" in tree_id:
            return name
    raise ValueError(f"Unknown tree family: {tree_id}")


def matrix_plan(trees=MATRIX_TREES, pilot_manifests=PILOT_MANIFESTS, companion=COMPANION, *, hash_assets=True):
    """The frozen plan. Every tree is registered before any render exists."""
    registered = []
    for tree_id, manifest in sorted(pilot_manifests.items()):
        manifest = Path(manifest)
        entry = {"tree_id": tree_id, "family": family_of(tree_id), "status": "rendered_pilot"}
        entry["render_manifest"] = str(manifest)
        if hash_assets:
            if not manifest.is_file():
                raise FileNotFoundError(f"Pilot render manifest is missing: {manifest}")
            entry["render_manifest_sha256"] = sha256(manifest)
        registered.append(entry)
    for tree_id in trees:
        entry = {"tree_id": tree_id, "family": family_of(tree_id), "status": "to_render"}
        metadata = companion / "trees/metadata" / f"{tree_id}_metadata.json"
        entry["metadata"] = str(metadata)
        if hash_assets:
            if not metadata.is_file():
                raise FileNotFoundError(f"Tree metadata is missing: {metadata}")
            entry["metadata_sha256"] = sha256(metadata)
        registered.append(entry)

    ids = [entry["tree_id"] for entry in registered]
    if len(set(ids)) != len(ids):
        raise ValueError("Registered trees must be unique; a repeated tree would weight the family mean")
    families = {entry["family"] for entry in registered}
    if families != {"envy", "ufo"}:
        raise ValueError("The matrix must register both families")

    to_render = sum(1 for entry in registered if entry["status"] == "to_render")
    plan = {
        "schema_version": 1,
        "scope": (
            "Offline frozen-model depth evaluation on Blender Cycles renders of registered Envy and UFO L-Py "
            "trees under four lighting presets. Not closed-loop control, not an unseen-tree test for Envy, "
            "not a radiometric lighting study."
        ),
        "protocol": "docs/EVAL_PROTOCOL_FAMILY_LIGHTING_2026-09-23.md",
        "max_concurrent_gpus": 1,
        "maximum_gpu_minutes": RENDER_MINUTES_RESERVED * to_render + EVAL_MINUTES_RESERVED,
        "lights": ["source", "morning", "noon", "evening"],
        "views_per_light": 6,
        "width": 512,
        "samples": 16,
        "seed": 1729,
        "renderer": "Blender 4.2.19 Cycles, Filmic view transform, orchard_template.blend, bark_brown_02",
        "trees": registered,
        "models": MODELS,
        "primary_metric": (
            "Tree-mask MAE in metres per model x family x lighting cell, with n_trees as the population "
            "count; the mask-gated target error is the target-level metric. Both are produced by "
            "sql/family/*.sql through tools/aggregate_family_eval.py."
        ),
        "gates": {
            "origin": "Pre-registered September 20 engineering sanity criteria, unchanged",
            "target_p95_absolute_m": 0.02,
            "target_mean_relative": 0.10,
            "target_valid_rate": 0.99,
            "p95_inference_seconds": 0.10,
        },
    }
    if hash_assets:
        blend = companion / "orchard_template.blend"
        if blend.is_file():
            plan["external_assets_sha256"] = {"orchard_template.blend": sha256(blend)}
    return plan


def storage_preflight(
    batch: Path,
    share_used_bytes: int | None,
    share_du_file: Path | None,
    estimated_output_bytes: int = ESTIMATED_OUTPUT_BYTES,
):
    """Refuse the batch when the projection crosses the share warning."""
    if share_used_bytes is None:
        if share_du_file is None or not Path(share_du_file).is_file():
            raise ValueError("A share measurement is required: pass --share-used-bytes or --share-du-file")
        text = Path(share_du_file).read_text(encoding="utf-8").strip()
        if not text:
            raise ValueError(f"Share measurement file is empty (du still running?): {share_du_file}")
        share_used_bytes = int(text.split()[0])
    report = {
        "schema_version": 1,
        "measured_at": datetime.now(timezone.utc).isoformat(),
        "share_used_bytes": int(share_used_bytes),
        "estimated_output_bytes": int(estimated_output_bytes),
        "projection": assess(int(share_used_bytes), int(estimated_output_bytes)),
        "policy": "No heavy home writes; reserve 20 GB; refuse at warning even below hard limit.",
    }
    report["ok"] = bool(report["projection"]["ok"])
    (batch / "storage_preflight.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["ok"]:
        raise RuntimeError("Storage preflight did not pass; nothing was submitted")
    return report


def _sbatch(command, env, batch, name):
    (batch / f"submission_intent_{name}.json").write_text(json.dumps(command, indent=2) + "\n", encoding="utf-8")
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60, check=False)
    receipt = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    (batch / f"submission_receipt_{name}.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"sbatch failed for {name}; inspect {batch}/submission_receipt_{name}.json before any retry")
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unrecognized sbatch receipt for {name}; inspect {batch}; do not resubmit blindly")
    return job_id


def queue_batch(root, batch_id, *, share_used_bytes=None, share_du_file=None, after=None):
    root = Path(root).resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", batch_id):
        raise ValueError("batch-id must be a short letters/digits/underscore/dash identifier")
    if _git(root, "diff", "HEAD", "--name-only").strip():
        raise ValueError("Commit tracked changes before freezing experiments; untracked work is preserved")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    plan = matrix_plan()
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
        "hpc/slurm/family_matrix_frozen.sbatch",
        "hpc/slurm/family_eval_frozen.sbatch",
        "tools/render_family_lighting.py",
        "tools/prepare_family_evaluation.py",
        "tools/depth_generalization.py",
        "tools/dino_generalization.py",
    ):
        if name not in hashes:
            raise ValueError(f"Required runner is not committed: {name}")
    plan["source_sha256"] = hashes
    (batch / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (batch / "queue_before.txt").write_text(queue_before, encoding="utf-8")
    storage_preflight(batch, share_used_bytes, share_du_file)

    to_render = sum(1 for entry in plan["trees"] if entry["status"] == "to_render")
    render_command = [
        "sbatch",
        "--parsable",
        "--array",
        f"0-{to_render - 1}%1",
        *dependency_option(after),
        "--chdir",
        str(batch / "code"),
        "--output",
        str(batch / "logs/%x-%A_%a.out"),
        "--error",
        str(batch / "logs/%x-%A_%a.out"),
        str(batch / "code/hpc/slurm/family_matrix_frozen.sbatch"),
        str(batch),
    ]
    render_id = _sbatch(render_command, env, batch, "render")
    eval_command = [
        "sbatch",
        "--parsable",
        *dependency_option(render_id),
        "--chdir",
        str(batch / "code"),
        "--output",
        str(batch / "logs/%x-%j.out"),
        "--error",
        str(batch / "logs/%x-%j.out"),
        str(batch / "code/hpc/slurm/family_eval_frozen.sbatch"),
        str(batch),
    ]
    eval_id = _sbatch(eval_command, env, batch, "eval")
    return {"render_array_job_id": render_id, "eval_job_id": eval_id, "batch_dir": str(batch), **plan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-id", default="family-matrix-20260923")
    parser.add_argument("--submit", action="store_true", help="Submit once; the default only prints the plan")
    parser.add_argument("--share-used-bytes", type=int, help="Measured du of the user's share, in bytes")
    parser.add_argument("--share-du-file", type=Path, help="File holding 'du -sx -B1' output for the share")
    parser.add_argument("--after", help="Queue the render array behind this Slurm job id; it is never modified")
    args = parser.parse_args()
    if args.submit:
        result = queue_batch(
            args.root,
            args.batch_id,
            share_used_bytes=args.share_used_bytes,
            share_du_file=args.share_du_file,
            after=args.after,
        )
    else:
        result = matrix_plan()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
