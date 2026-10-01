#!/usr/bin/env python3
"""Plan or submit the frozen render-gap check: the eight matrix trees re-rendered with the training renderer.

The rendered-lighting fine-tunes improved every Blender matrix cell, source light included, while the
companion's own source-lit validation RMSE did not move. The matrix frames were rendered at 512x288 with 16
fixed samples (seed 1729, no world fix); the training frames at 1920x1080 with the template's adaptive
4,096-sample sampler after the world fix. This batch re-renders the eight matrix trees at the same camera poses
and lights with the training renderer's settings (``render_family_lighting.py --renderer training``) and scores
the five registered checkpoints on them in two arms: NATIVE (the new frames against their own depth and mask)
and MATCHED (the new frames downsampled to 512x288 with INTER_AREA against the published matrix depth and mask,
after a per-view pose-identity check; see ``prepare_render_gap_evaluation.py``).

Same discipline as ``queue_family_matrix.py``, whose helpers this reuses: committed source only, a clean tracked
tree, every tree, checkpoint and matrix ground-truth manifest registered and hashed in the plan before the only
scheduler mutation, intent recorded before each call and the receipt after it, a storage preflight against the
share's quota, a render array at concurrency 1 and one evaluation job chained ``afterany``, so a failed render
leaves a recorded gap rather than a job pending forever. The batch cannot be submitted before its protocol
document is committed.

What it does not do: it never re-renders, edits or moves the published matrix, never writes the companion
(read and hashed only), never chooses trees or checkpoints, and it cannot separate lighting from the render gap
in what the fine-tunes learned. Renders derive from the companion's licensed meshes and textures and stay under
``artifacts/``.
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
import render_family_lighting as rfl  # noqa: E402
import training_lighting as tl  # noqa: E402
from queue_family_matrix import (  # noqa: E402
    COMPANION,
    MATRIX_TREES,
    MODELS,
    PILOT_MANIFESTS,
    _sbatch,
    family_of,
    sha256,
    storage_preflight,
)
from queue_vision_robustness import (  # noqa: E402
    _git,
    dependency_option,
    freeze_source,
    placement_options,
    submission_environment,
)

ROOT = Path(__file__).resolve().parents[1]
GENERALIZATION = ROOT / "artifacts/generalization"
MATRIX_BATCH = GENERALIZATION / "family-matrix-20260923"
MATRIX_EVALUATION_PLAN = MATRIX_BATCH / "eval/plan.json"
#: Written separately; the batch refuses to submit until this file is committed.
PROTOCOL = "docs/EVAL_PROTOCOL_RENDER_GAP_2026-09-30.md"
DEFAULT_BATCH_ID = "render-gap-20260930"
TREES = (
    "lpy_envy_00000",
    "lpy_envy_00003",
    "lpy_envy_00008",
    "lpy_envy_00012",
    "lpy_ufo_00000",
    "lpy_ufo_00001",
    "lpy_ufo_00002",
    "lpy_ufo_00003",
)
LIGHTS = ("source", "morning", "noon", "evening")
VIEWS_PER_LIGHT = 6
NATIVE_RESOLUTION = (1920, 1080)
MATRIX_RESOLUTION = (512, 288)
#: The four fine-tuned arms of docs/EVAL_PROTOCOL_LIGHTING_TRAINING_2026-09-27.md (Result), by batch.
FINETUNES = {
    "J": ("finetune-jitter-20260923", "Published jitter arm: surviving source-lit frames, photometric jitter"),
    "C": ("finetune-control-20260923", "Published control arm: surviving source-lit frames, no jitter"),
    "A": ("finetune-rendered-jitter-20260927", "Rendered-lighting arm A: surviving plus rendered frames, jitter"),
    "B": ("finetune-rendered-control-20260927", "Rendered-lighting arm B: surviving plus rendered frames, no jitter"),
}
PLACEMENT = "gpu,ampere"
#: The training renderer's 4,890 frames in training-lighting-full-20260927 took 15.4 s on average (p99 36.5 s,
#: slowest 243 s): about 6 minutes for a tree's 24 frames, plus six short 512x288 geometry passes. The tail is
#: real: in that batch's array 21442470, one 60-frame task took 31.6 minutes against a median near 16 and one
#: timed out at 60. The evaluation scores 192 frames per arm for five checkpoints.
RENDER_MINUTES_RESERVED = 15
EVAL_MINUTES_RESERVED = 60
RENDER_TIME_RISK = (
    "A tree runs about 6 minutes at the measured mean of 15.4 s per frame. Of the 74 tasks of "
    "training-lighting-full-20260927, one ran about twice as slow as the median (31.6 minutes for 60 frames; 24 "
    "frames at that rate take about 13 minutes and fit) and one stalled until its 60-minute limit (lpy_envy_00014, "
    "whose recorded frames total 27 minutes), which no reservation here would absorb. A tree stopped by its limit "
    "keeps a manifest with ok false and the frames it rendered; it is recorded and excluded from both arms, never "
    "re-rendered by this batch."
)
#: Storage per frame or view: a training-renderer RGB PNG averages 2.7 MB (3 MiB reserved); the depth and every
#: saved prediction are float32 arrays at their resolution; masks and MATCHED PNGs are small.
RGB_BYTES = 3 * 1024 * 1024
MASK_BYTES = 64 * 1024
MATCHED_RGB_BYTES = 512 * 1024
JSON_BYTES = 64 * 1024 * 1024
HEADROOM = 1.25
REQUIRED_SOURCES = (
    "hpc/slurm/render_gap_render_frozen.sbatch",
    "hpc/slurm/render_gap_eval_frozen.sbatch",
    "tools/render_family_lighting.py",
    "tools/prepare_render_gap_evaluation.py",
    "tools/prepare_family_evaluation.py",
    "tools/depth_generalization.py",
    "tools/training_lighting.py",
    PROTOCOL,
)


def matrix_manifests(matrix_batch=MATRIX_BATCH, pilot_manifests=PILOT_MANIFESTS, matrix_trees=MATRIX_TREES):
    """The render manifests the published matrix scored: the two pilots and the six matrix renders, by tree."""
    manifests = {tree_id: Path(path) for tree_id, path in pilot_manifests.items()}
    for index, tree_id in enumerate(matrix_trees):
        manifests[tree_id] = Path(matrix_batch) / f"matrix_{index}_{tree_id}" / "render_manifest.json"
    return manifests


def checkpoint_register(generalization=GENERALIZATION, models=MODELS):
    """The five checkpoints, F first: the frozen published model and the four fine-tuned arms."""
    register = {
        "F": {
            "checkpoint": models["da2"]["checkpoint"],
            "expected_sha256": models["da2"]["checkpoint_sha256"],
            "source": "queue_family_matrix.MODELS['da2']",
            "note": "Frozen published DA2 metric ViT-L checkpoint",
        }
    }
    for label, (batch_id, note) in FINETUNES.items():
        batch = Path(generalization) / batch_id
        register[label] = {
            "checkpoint": str(batch / "train/best.pth"),
            "recorded_status": str(batch / "eval/status.json"),
            "source": batch_id,
            "note": note,
        }
    return register


def estimated_output_bytes(trees, models):
    """Renders, MATCHED inputs, every saved prediction in both arms and the evaluation documents, with headroom."""
    frames = trees * len(LIGHTS) * VIEWS_PER_LIGHT
    native = NATIVE_RESOLUTION[0] * NATIVE_RESOLUTION[1] * 4
    matrix = MATRIX_RESOLUTION[0] * MATRIX_RESOLUTION[1] * 4
    # Per view: the 1920x1080 depth and mask, and the matrix-grid depth and mask that gate MATCHED.
    renders = frames * RGB_BYTES + trees * VIEWS_PER_LIGHT * (native + matrix + 2 * MASK_BYTES)
    predictions = models * frames * (native + matrix)
    return int(HEADROOM * (renders + frames * MATCHED_RGB_BYTES + predictions + JSON_BYTES))


def _register_tree(tree_id, manifest, companion, published, hash_assets):
    metadata = Path(companion) / "trees/metadata" / f"{tree_id}_metadata.json"
    entry = {
        "tree_id": tree_id,
        "family": family_of(tree_id),
        "status": "to_render",
        "metadata": str(metadata),
        "matrix_render_manifest": str(manifest),
    }
    if not hash_assets:
        return entry
    if not metadata.is_file():
        raise FileNotFoundError(f"Tree metadata is missing: {metadata}")
    entry["metadata_sha256"] = sha256(metadata)
    if not manifest.is_file():
        raise FileNotFoundError(f"Matrix GT render manifest for {tree_id} is missing: {manifest}")
    data = json.loads(manifest.read_text())
    frames = len(LIGHTS) * VIEWS_PER_LIGHT
    if data.get("tree_id") != tree_id or not data.get("ok") or len(data.get("frames", [])) != frames:
        raise ValueError(f"{manifest} is not a finished {frames}-frame matrix render of {tree_id}")
    entry["matrix_render_manifest_sha256"] = sha256(manifest)
    if published is not None and published.get(str(manifest)) != entry["matrix_render_manifest_sha256"]:
        raise ValueError(f"{manifest} is not the manifest the published matrix evaluation scored")
    entry["matrix_renderer"] = data.get("renderer")
    entry["matrix_geometry_sha256"] = data.get("geometry_sha256")
    return entry


def _register_checkpoint(label, spec, hash_assets):
    if not re.fullmatch(r"[A-Za-z0-9_-]+", label):
        raise ValueError(f"Checkpoint labels name output directories; {label!r} is not a plain token")
    entry = {k: str(v) for k, v in spec.items() if k != "expected_sha256"}
    path = Path(spec["checkpoint"])
    if not hash_assets:
        if spec.get("expected_sha256"):
            entry["checkpoint_sha256"] = spec["expected_sha256"]
        return entry
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint {label} is missing: {path}")
    digest = sha256(path)
    if spec.get("expected_sha256") and digest != spec["expected_sha256"]:
        raise ValueError(f"Checkpoint {label} differs from its registered hash: {path}")
    recorded = spec.get("recorded_status")
    if recorded and Path(recorded).is_file():
        prior = json.loads(Path(recorded).read_text()).get("checkpoint_sha256")
        if prior and prior != digest:
            raise ValueError(f"Checkpoint {label} differs from the hash its own evaluation recorded: {recorded}")
        entry["recorded_status_sha256_matches"] = prior == digest if prior else None
    entry["checkpoint_sha256"] = digest
    return entry


def render_gap_plan(
    trees=TREES,
    manifests=None,
    checkpoints=None,
    companion=COMPANION,
    matrix_evaluation_plan=MATRIX_EVALUATION_PLAN,
    *,
    hash_assets=True,
):
    """The frozen plan. Every tree, ground-truth source and checkpoint is registered before any render exists."""
    manifests = matrix_manifests() if manifests is None else manifests
    checkpoints = checkpoint_register() if checkpoints is None else checkpoints
    ids = list(trees)
    if len(set(ids)) != len(ids):
        raise ValueError("Registered trees must be unique; a repeated tree would weight the family mean")
    if {family_of(tree_id) for tree_id in ids} != {"envy", "ufo"}:
        raise ValueError("The check must register both families")
    missing = [tree_id for tree_id in ids if tree_id not in manifests]
    if missing:
        raise ValueError(f"No matrix GT render manifest is registered for {missing}")
    published = None
    if hash_assets and matrix_evaluation_plan is not None:
        if not Path(matrix_evaluation_plan).is_file():
            raise FileNotFoundError(f"Published matrix evaluation plan is missing: {matrix_evaluation_plan}")
        published = {
            s["path"]: s["sha256"] for s in json.loads(Path(matrix_evaluation_plan).read_text())["source_renders"]
        }
    registered = [_register_tree(t, Path(manifests[t]), companion, published, hash_assets) for t in ids]
    models = {label: _register_checkpoint(label, spec, hash_assets) for label, spec in checkpoints.items()}
    plan = {
        "schema_version": 1,
        "scope": (
            "Offline frozen-model depth evaluation of five registered checkpoints on Blender Cycles re-renders of "
            "the eight matrix trees at the matrix camera poses and lighting presets with the training renderer's "
            "settings, in two arms (NATIVE, MATCHED). A render-gap check, not a lighting study, not closed-loop "
            "control, not an unseen-tree test for Envy."
        ),
        "protocol": PROTOCOL,
        "renderer": "training",
        "renderer_command": "tools/render_family_lighting.py --renderer training",
        "template_sampler": dict(
            rfl.TEMPLATE_SAMPLER, view_transform=rfl.TEMPLATE_VIEW_TRANSFORM, resolution=[*NATIVE_RESOLUTION, 100]
        ),
        "lights": list(LIGHTS),
        "views_per_light": VIEWS_PER_LIGHT,
        "lighting_seed": 1729,
        "native_resolution": list(NATIVE_RESOLUTION),
        "matrix_resolution": list(MATRIX_RESOLUTION),
        "trees": registered,
        "registered_frames": len(registered) * len(LIGHTS) * VIEWS_PER_LIGHT,
        "models": models,
        "arms": {
            "native": "New 1920x1080 RGB against the same render's own depth and tree mask.",
            "matched": (
                "New RGB downsampled to 512x288 with cv2.INTER_AREA against the published matrix depth and tree "
                "mask of the same tree, light and view; only trees whose every view passes pose identity."
            ),
        },
        "pose_identity": {
            "tolerance_m": tl.AGREEMENT_TOLERANCE_M,
            "min_fraction": tl.AGREEMENT_MIN_FRACTION,
            "min_iou": tl.AGREEMENT_MIN_IOU,
            "function": "training_lighting.agreement(matrix-grid pass, published matrix depth and mask)",
            "new_geometry": (
                "Each view's depth and tree mask rendered once more on the 512x288 matrix grid with the matrix "
                f"renderer's sampler ({rfl.MATRIX_SAMPLES} samples, seed {rfl.MATRIX_SEED}, no adaptive sampling), "
                "under the first light, in the same Blender session as its 1920x1080 frames"
            ),
            "rule": "Every view of a tree must pass; a failing tree is excluded from MATCHED and recorded.",
            "cross_grid_measurement": (
                "The 1920x1080 depth and mask point-sampled at the 1920x1080 pixel containing each 512x288 pixel "
                "centre, compared with the same thresholds; recorded, never gates. Cycles takes depth and object "
                "index at pixel centres and 1920/512 = 3.75, so those samples sit 1/30 or 1/10 of a matrix pixel "
                "from the matrix ones."
            ),
        },
        "matrix_evaluation_plan": str(matrix_evaluation_plan) if matrix_evaluation_plan is not None else None,
        "placement": {"partition": PLACEMENT},
        "render_minutes_per_task": RENDER_MINUTES_RESERVED,
        "render_time_risk": RENDER_TIME_RISK,
        "eval_minutes": EVAL_MINUTES_RESERVED,
        "max_concurrent_gpus": 1,
        "maximum_gpu_minutes": RENDER_MINUTES_RESERVED * len(registered) + EVAL_MINUTES_RESERVED,
        "estimated_output_bytes": estimated_output_bytes(len(registered), len(models)),
        "licensing": "Renders derive from the companion's licensed meshes and textures; they stay under artifacts/.",
    }
    if hash_assets:
        if matrix_evaluation_plan is not None:
            plan["matrix_evaluation_plan_sha256"] = sha256(matrix_evaluation_plan)
        for name in ("orchard_template.blend", "Dataloader/generate_tree2.py", "Dataloader/daylight_presets.py"):
            path = Path(companion) / name
            if path.is_file():
                plan.setdefault("external_assets_sha256", {})[name] = sha256(path)
    return plan


def queue_batch(root, batch_id, *, share_used_bytes=None, share_du_file=None, after=None):
    root = Path(root).resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", batch_id):
        raise ValueError("batch-id must be a short letters/digits/underscore/dash identifier")
    if _git(root, "diff", "HEAD", "--name-only").strip():
        raise ValueError("Commit tracked changes before freezing experiments; untracked work is preserved")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    plan = render_gap_plan()
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
    for name in REQUIRED_SOURCES:
        if name not in hashes:
            raise ValueError(f"Required runner or protocol is not committed: {name}")
    plan["source_sha256"] = hashes
    (batch / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (batch / "queue_before.txt").write_text(queue_before, encoding="utf-8")
    storage_preflight(batch, share_used_bytes, share_du_file, estimated_output_bytes=plan["estimated_output_bytes"])

    to_render = sum(1 for entry in plan["trees"] if entry["status"] == "to_render")
    render_command = [
        "sbatch",
        "--parsable",
        "--array",
        f"0-{to_render - 1}%1",
        *dependency_option(after),
        *placement_options({"partition": PLACEMENT, "minutes_per_task": RENDER_MINUTES_RESERVED}),
        "--chdir",
        str(batch / "code"),
        "--output",
        str(batch / "logs/%x-%A_%a.out"),
        "--error",
        str(batch / "logs/%x-%A_%a.out"),
        str(batch / "code/hpc/slurm/render_gap_render_frozen.sbatch"),
        str(batch),
    ]
    render_id = _sbatch(render_command, env, batch, "render")
    eval_command = [
        "sbatch",
        "--parsable",
        *dependency_option(render_id),
        *placement_options({"partition": PLACEMENT, "minutes_per_task": EVAL_MINUTES_RESERVED}),
        "--chdir",
        str(batch / "code"),
        "--output",
        str(batch / "logs/%x-%j.out"),
        "--error",
        str(batch / "logs/%x-%j.out"),
        str(batch / "code/hpc/slurm/render_gap_eval_frozen.sbatch"),
        str(batch),
    ]
    eval_id = _sbatch(eval_command, env, batch, "eval")
    return {"render_array_job_id": render_id, "eval_job_id": eval_id, "batch_dir": str(batch), **plan}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-id", default=DEFAULT_BATCH_ID)
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
        result = render_gap_plan()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
