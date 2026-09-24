#!/usr/bin/env python3
"""Plan or submit the frozen tree0 renderer-control batch without touching existing jobs.

Same discipline as the other launchers: committed source only, a clean tree,
the plan written before the only scheduler mutation, intent and receipt files,
and the evaluation queued with ``afterany`` on the render. The recorded Stage A
runs the render replays are pinned by the hash of their reports.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from queue_family_matrix import COMPANION, MODELS, _sbatch, sha256, storage_preflight
from queue_generalization_controls import RELATIVE_HEAD
from queue_vision_robustness import _git, dependency_option, freeze_source, submission_environment

ROOT = Path(__file__).resolve().parents[1]
STAGE_A = ROOT / "artifacts/vision_robustness/lighting-20260919"
RUNS = ("run_00_source_raw", "run_02_morning_raw", "run_04_evening_raw")
FRAME_STRIDE = 3
MAX_FRAME = 77  # detachment is requested after frame 77 in every Stage A run
RENDER_MINUTES_RESERVED = 20
EVAL_MINUTES_RESERVED = 30
ESTIMATED_OUTPUT_BYTES = 400 * 1024 * 1024


def frames_per_run(stride=FRAME_STRIDE, max_frame=MAX_FRAME):
    return len(range(0, max_frame + 1, stride))


def replay_plan(runs=RUNS, *, hash_assets=True, stage_a=STAGE_A, companion=COMPANION):
    registered = []
    for name in runs:
        entry = {"name": name, "path": str(stage_a / name)}
        if hash_assets:
            report = stage_a / name / "report.json"
            if not report.is_file():
                raise FileNotFoundError(f"Stage A run is missing: {report}")
            entry["report_sha256"] = sha256(report)
            entry["daylight"] = json.loads(report.read_text())["blender_scene"]["daylight"]["preset"]
        registered.append(entry)
    if len({r["name"] for r in registered}) != len(registered):
        raise ValueError("Runs must be unique")
    per_run = frames_per_run()
    plan = {
        "schema_version": 1,
        "scope": (
            "Renderer control for the Isaac Stage A depth failure: the original orchard tree0 rendered by "
            "Blender Cycles at the recorded Isaac wrist poses with the Isaac camera model, under the palm "
            "bark Isaac carried and the bark_brown_02 of the training renders. Not an Isaac render."
        ),
        "protocol": "docs/EVAL_PROTOCOL_TREE0_REPLAY_2026-09-23.md",
        "max_concurrent_gpus": 1,
        "maximum_gpu_minutes": RENDER_MINUTES_RESERVED + EVAL_MINUTES_RESERVED,
        "runs": registered,
        "frame_stride": FRAME_STRIDE,
        "max_frame": MAX_FRAME,
        "frames_per_run": per_run,
        "barks": ["palm", "bark_brown_02"],
        "registered_frames": per_run * len(registered) * 2,
        "samples": 16,
        "seed": 1729,
        "models": {"da2": MODELS["da2"]},
        "relative_head": {k: v for k, v in RELATIVE_HEAD.items() if k != "extra_frames"},
        "comparison": (
            "Cells are compared with the Stage A Isaac cells of the same runs and frames in "
            "docs/evidence/stage_a_depth_2026-09-23.json and the anchoring ceilings"
        ),
    }
    if hash_assets:
        for name in ("orchard_template.blend", "Dataloader/daylight_presets.py"):
            path = companion / name
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
    plan = replay_plan()
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
        "hpc/slurm/tree0_replay_render_frozen.sbatch",
        "hpc/slurm/tree0_replay_eval_frozen.sbatch",
        "tools/render_tree0_replay.py",
        "tools/generalization_controls.py",
        "tools/render_family_lighting.py",
        "tools/prepare_replay_evaluation.py",
        "tools/depth_generalization.py",
        "tools/relative_depth_ceiling.py",
    ):
        if name not in hashes:
            raise ValueError(f"Required runner is not committed: {name}")
    plan["source_sha256"] = hashes
    (batch / "plan.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
    (batch / "queue_before.txt").write_text(queue_before, encoding="utf-8")
    storage_preflight(batch, share_used_bytes, share_du_file, estimated_output_bytes=ESTIMATED_OUTPUT_BYTES)
    render_command = [
        "sbatch",
        "--parsable",
        *dependency_option(after),
        "--chdir",
        str(batch / "code"),
        "--output",
        str(batch / "logs/%x-%j.out"),
        "--error",
        str(batch / "logs/%x-%j.out"),
        str(batch / "code/hpc/slurm/tree0_replay_render_frozen.sbatch"),
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
        str(batch / "code/hpc/slurm/tree0_replay_eval_frozen.sbatch"),
        str(batch),
    ]
    eval_id = _sbatch(eval_command, env, batch, "eval")
    return {"render_job_id": render_id, "eval_job_id": eval_id, "batch_dir": str(batch), **plan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--batch-id", default="tree0-replay-20260923")
    parser.add_argument("--submit", action="store_true", help="Submit once; the default only prints the plan")
    parser.add_argument("--share-used-bytes", type=int)
    parser.add_argument("--share-du-file", type=Path)
    parser.add_argument("--after", help="Queue the render behind this Slurm job id; it is never modified")
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
        result = replay_plan()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
