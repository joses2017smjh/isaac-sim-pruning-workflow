#!/usr/bin/env python3
"""Plan or submit the frozen generalization-controls batch without touching existing jobs.

Same discipline as ``queue_family_matrix.py``: committed source only, a clean
tracked tree, the plan written before the only scheduler mutation, intent
recorded before the call and the receipt after it. The render array is sized
from the plan (one task per registered tree) and the evaluation job waits on
it with ``afterany`` so a failed render leaves a visible gap, not a job pending
forever.

The register is the matrix register: all eight trees are re-rendered here,
because every condition is a controlled change from the matrix ``source`` cell
and the comparison must be within one batch. The re-rendered ``source`` and
``evening`` cells double as a determinism check against the matrix (metrics,
not bytes; OptiX is not bit-stable).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import generalization_controls as gc
from queue_family_matrix import (
    COMPANION,
    MATRIX_TREES,
    MODELS,
    PILOT_MANIFESTS,
    _sbatch,
    family_of,
    sha256,
    storage_preflight,
)
from queue_vision_robustness import _git, dependency_option, freeze_source, submission_environment

ROOT = Path(__file__).resolve().parents[1]
MATRIX_BATCH = ROOT / "artifacts/generalization/family-matrix-20260923"

# Measured on the matrix: 53-62 s per 24-frame tree on a Quadro RTX 8000, 2.1 s
# per frame; 62 frames plus 38 geometry passes per tree here. Scoring: 60 s per
# 48 DA2 frames on the matrix; 688 metric frames here, 1,480 relative-head
# frames (controls, matrix and Stage A) and DINO on 72 six-view groups.
RENDER_MINUTES_RESERVED = 15
EVAL_MINUTES_RESERVED = 60
# 688 metric, 1,480 disparity and 432 DINO predictions at about 0.6 MB each,
# renders and shared geometry about 260 MB, photometric copies 20 MB, headroom.
ESTIMATED_OUTPUT_BYTES = 2_500 * 1024 * 1024

#: The public relative checkpoint whose backbone the metric fine-tune started
#: from, and the frame sets the relative head is scored on beside the controls.
RELATIVE_HEAD = {
    "checkpoint": str(COMPANION / "da2_weights/depth_anything_v2_vitl.pth"),
    "checkpoint_sha256": "a7ea19fa0ed99244e67b624c72b8580b7e9553043245905be58796a608eb9345",
    "note": "DA2 relative ViT-L, ReLU disparity head; scored only through an all-GT affine fit (a ceiling)",
    "extra_frames": {
        "stage_a_isaac": {
            "path": str(ROOT / "artifacts/generalization/phase-20260920/stage_a_job_21370005/evaluation.json"),
            "sha256": "fcb1c4cfabc2e75d00156f8d361c0ab5725741ce4b7eee4585a12a741d185362",
            "frames": 600,
            "note": "Isaac RTX wrist frames with simulator depth, no tree mask; the close-range failure case",
        },
        "family_matrix": {
            "path": str(MATRIX_BATCH / "eval/plan.json"),
            "sha256": "8bef6a1582d653d74197b531b60bfb1ec8b074a1201e04e02c06fb93cf1c366b",
            "frames": 192,
            "note": "The registered matrix renders, so the two heads are compared on identical bytes",
        },
    },
}

REGISTER = tuple(sorted(PILOT_MANIFESTS)) + MATRIX_TREES


def controls_plan(trees=REGISTER, companion=COMPANION, *, hash_assets=True, matrix_batch=MATRIX_BATCH):
    """The frozen plan. Every tree and condition is registered before any render exists."""
    registered = []
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
    if {entry["family"] for entry in registered} != {"envy", "ufo"}:
        raise ValueError("The controls must register both families")
    condition_ids = [c["id"] for c in gc.CONDITIONS]
    if len(set(condition_ids)) != len(condition_ids):
        raise ValueError("Condition ids must be unique")
    for cond in gc.CONDITIONS:
        if cond["camera_model"] not in gc.CAMERA_MODELS or cond["pose_set"] not in gc.POSE_SETS:
            raise ValueError(f"Condition {cond['id']} names an unregistered camera model or pose set")
    per_tree = gc.frames_per_tree()
    photometric_per_tree = len(gc.PHOTOMETRIC_VARIANTS) * len(gc.pose_ids(gc.POSE_SETS["far"]))

    plan = {
        "schema_version": gc.SCHEMA_VERSION,
        "scope": (
            "Offline frozen-model depth evaluation on Blender Cycles renders of the registered Envy and UFO "
            "L-Py trees under controlled single-axis changes from the matrix source cell: lighting presets "
            "and repo-local brightness controls, test-time photometric normalization, the Isaac wrist camera "
            "model, close-range and upward-pitched rigs, and a distance sweep. Not closed-loop control, not "
            "an unseen-tree test for Envy, not a radiometric lighting study, not an Isaac render."
        ),
        "protocol": "docs/EVAL_PROTOCOL_GENERALIZATION_CONTROLS_2026-09-23.md",
        "max_concurrent_gpus": 1,
        "maximum_gpu_minutes": RENDER_MINUTES_RESERVED * len(registered) + EVAL_MINUTES_RESERVED,
        "conditions": gc.CONDITIONS,
        "baseline_of": gc.BASELINE_OF,
        "camera_models": gc.CAMERA_MODELS,
        "pose_sets": gc.POSE_SETS,
        "lighting_controls": gc.LIGHTING_CONTROLS,
        "luma_factor": gc.LUMA_FACTOR,
        "isaac_pitch_deg": gc.ISAAC_PITCH_DEG,
        "photometric_variants": list(gc.PHOTOMETRIC_VARIANTS),
        "photometric_parameters": pn_variants(),
        "frames_per_tree": per_tree,
        "photometric_frames_per_tree": photometric_per_tree,
        "registered_frames": (per_tree + photometric_per_tree) * len(registered),
        "samples": 16,
        "seed": 1729,
        "renderer": "Blender 4.2.19 Cycles, Filmic view transform, orchard_template.blend, bark_brown_02",
        "trees": registered,
        "models": MODELS,
        "relative_head": RELATIVE_HEAD,
        "dino_scope": "Six-view DINO is scored on far-rig training-camera cells only; other rigs are DA2-only",
        "primary_metric": (
            "Tree-mask MAE in metres per model x family x condition, paired within tree against the "
            "condition's registered baseline; the mask-gated target error is the target-level metric. "
            "Produced by sql/controls/*.sql through tools/aggregate_controls_eval.py."
        ),
        "gates": {
            "origin": "Pre-registered September 20 engineering sanity criteria, unchanged",
            "target_p95_absolute_m": 0.02,
            "target_mean_relative": 0.10,
            "target_valid_rate": 0.99,
            "p95_inference_seconds": 0.10,
        },
        "matrix_reference": {
            "batch": str(matrix_batch),
            "note": "Metrics of the re-rendered source and evening cells are compared with the matrix cells",
        },
    }
    if hash_assets:
        for name in ("orchard_template.blend", "Dataloader/generate_tree2.py", "Dataloader/daylight_presets.py"):
            path = companion / name
            if path.is_file():
                plan.setdefault("external_assets_sha256", {})[name] = sha256(path)
        matrix_plan = matrix_batch / "plan.json"
        if matrix_plan.is_file():
            plan["matrix_reference"]["plan_sha256"] = sha256(matrix_plan)
    return plan


def pn_variants():
    import photometric_normalization as pn

    return {name: dict(spec) for name, spec in pn.VARIANTS.items()}


def queue_batch(root, batch_id, *, share_used_bytes=None, share_du_file=None, after=None):
    root = Path(root).resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", batch_id):
        raise ValueError("batch-id must be a short letters/digits/underscore/dash identifier")
    if _git(root, "diff", "HEAD", "--name-only").strip():
        raise ValueError("Commit tracked changes before freezing experiments; untracked work is preserved")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    plan = controls_plan()
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
        "hpc/slurm/controls_render_frozen.sbatch",
        "hpc/slurm/controls_eval_frozen.sbatch",
        "tools/generalization_controls.py",
        "tools/render_generalization_controls.py",
        "tools/render_family_lighting.py",
        "tools/prepare_controls_evaluation.py",
        "tools/photometric_normalization.py",
        "tools/depth_generalization.py",
        "tools/relative_depth_ceiling.py",
        "tools/dino_generalization.py",
    ):
        if name not in hashes:
            raise ValueError(f"Required runner is not committed: {name}")
    for name, source in plan["relative_head"]["extra_frames"].items():
        if not Path(source["path"]).is_file():
            raise FileNotFoundError(f"Relative-head frame source {name} is missing: {source['path']}")
        if sha256(source["path"]) != source["sha256"]:
            raise ValueError(f"Relative-head frame source {name} changed since it was pinned")
    plan["source_sha256"] = hashes
    (batch / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (batch / "queue_before.txt").write_text(queue_before, encoding="utf-8")
    storage_preflight(batch, share_used_bytes, share_du_file, estimated_output_bytes=ESTIMATED_OUTPUT_BYTES)

    render_command = [
        "sbatch",
        "--parsable",
        "--array",
        f"0-{len(plan['trees']) - 1}%1",
        *dependency_option(after),
        "--chdir",
        str(batch / "code"),
        "--output",
        str(batch / "logs/%x-%A_%a.out"),
        "--error",
        str(batch / "logs/%x-%A_%a.out"),
        str(batch / "code/hpc/slurm/controls_render_frozen.sbatch"),
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
        str(batch / "code/hpc/slurm/controls_eval_frozen.sbatch"),
        str(batch),
    ]
    eval_id = _sbatch(eval_command, env, batch, "eval")
    return {"render_array_job_id": render_id, "eval_job_id": eval_id, "batch_dir": str(batch), **plan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-id", default="generalization-controls-20260923")
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
        result = controls_plan()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
