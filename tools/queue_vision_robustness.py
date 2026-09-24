#!/usr/bin/env python3
"""Plan or submit six frozen, serial GPU trials without modifying existing jobs."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from vision_experiment_assets import inventory_assets

MINUTES_PER_TRIAL = 17  # Measured 16.5 on array 21360571; rounded up for the declared budget.
TRIAL_TIME_LIMIT_MINUTES = 25  # The sbatch directive for a 200-frame episode; scaled with the plan's frames.


def trial_time_limit(plan):
    """Slurm time limit per task: the 200-frame reservation scaled by the episode length."""
    return f"00:{TRIAL_TIME_LIMIT_MINUTES * max(1, round(plan['frames'] / 200)):02d}:00"


#: Labelled approach strategies (docs/EVAL_PROTOCOL_STRATEGIES_2026-09-23.md).
#: None changes a gate, a threshold or the grader. ``frames`` is the episode
#: length the strategy needs: the fine step halves the speed of both approach
#: and retreat, so it gets twice the baseline's 200 frames.
STRATEGIES = {
    "baseline": {"name": "baseline", "mode": "straight", "standoff_m": 0.0, "max_step_m": 0.004, "frames": 200},
    "tool_axis_standoff": {
        "name": "tool_axis_standoff",
        "mode": "tool_axis_standoff",
        "standoff_m": 0.06,
        "max_step_m": 0.004,
        "frames": 200,
    },
    "horizontal_standoff": {
        "name": "horizontal_standoff",
        "mode": "horizontal_standoff",
        "standoff_m": 0.08,
        "max_step_m": 0.004,
        "frames": 200,
    },
    "fine_step": {"name": "fine_step", "mode": "straight", "standoff_m": 0.0, "max_step_m": 0.002, "frames": 400},
}


def experiment_plan(targets=None, daylight="source", photometric_normalization="raw", strategy=None):
    """Build the frozen plan. Without targets this is the original lighting pilot."""
    base = {
        "schema_version": 1,
        "max_concurrent_gpus": 1,
        "frames": 200,
        "fps": 10,
        "render_samples": 64,
        "overview_width": 1280,
    }
    if strategy is not None:
        if targets is None:
            raise ValueError("A strategy sweep needs a registered target file")
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}; registered: {sorted(STRATEGIES)}")
        base["frames"] = STRATEGIES[strategy]["frames"]
    if targets is None:
        return {
            **base,
            "scope": "Paired one-target lighting pilot; not a population success-rate estimate.",
            "maximum_gpu_minutes": 150,
            "runs": [
                {"index": index, "daylight": light, "photometric_normalization": mode}
                for index, (light, mode) in enumerate(
                    (light, mode) for light in ("source", "morning", "evening") for mode in ("raw", "clahe")
                )
            ],
        }

    runs = [
        {
            "index": index,
            "daylight": daylight,
            "photometric_normalization": photometric_normalization,
            "target_tree_index": int(target["target_tree_index"]),
            "component_first_vertex": int(target["component_first_vertex"]),
            **(
                {"strategy": {k: v for k, v in STRATEGIES[strategy].items() if k != "frames"}}
                if strategy is not None
                else {}
            ),
        }
        for index, target in enumerate(targets)
    ]
    if not runs:
        raise ValueError("A target sweep requires at least one registered target")
    seen = {(run["target_tree_index"], run["component_first_vertex"]) for run in runs}
    if len(seen) != len(runs):
        raise ValueError("Registered targets must be unique; a repeated spur would bias the rate")
    return {
        **base,
        "scope": (
            "Pre-registered target sweep for a task success rate over spur geometry, axis "
            "orientation and local occlusion at a canonicalized approach pose. Not a field "
            "success rate and not a reachability study."
        ),
        "protocol": "docs/EVAL_PROTOCOL_2026-09-23.md"
        if strategy is None
        else "docs/EVAL_PROTOCOL_STRATEGIES_2026-09-23.md",
        "strategy": None if strategy is None else STRATEGIES[strategy],
        "maximum_gpu_minutes": MINUTES_PER_TRIAL * len(runs) * (base["frames"] // 200),
        "runs": runs,
    }


def unpresentable_targets(targets, manifest):
    """Targets the renderer would refuse before recording anything.

    blender_demo_scene.spawn_blender_demo_scene only accepts a tree1 spur that is
    among the export's listed candidates; any other tree1 component raises
    "Unlisted tree1 components require a new geometry audit" inside the GPU job.
    Checking here, on CPU, means such a register is refused before it can spend
    an allocation discovering the same thing.
    """
    listed = {
        int(item["component_first_vertex"])
        for item in manifest["branch_geometry_candidates"]["tree1_SPUR"]["candidates"]
    }
    return [
        target
        for target in targets
        if int(target["target_tree_index"]) == 1 and int(target["component_first_vertex"]) not in listed
    ]


DEFAULT_EXPORT_MANIFEST = (
    Path(__file__).resolve().parents[1] / "artifacts/blender_scene/orchard_two_trees_v1/manifest.json"
)


def load_targets(path, manifest_path=DEFAULT_EXPORT_MANIFEST):
    """Read a pre-registered target register and keep its provenance in the plan."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    targets = document["targets"]
    if len(targets) != int(document["target_count"]):
        raise ValueError("Target register disagrees with its own target_count")
    if manifest_path is not None and Path(manifest_path).is_file():
        refused = unpresentable_targets(targets, json.loads(Path(manifest_path).read_text(encoding="utf-8")))
        if refused:
            vertexes = ", ".join(str(item["component_first_vertex"]) for item in refused)
            raise ValueError(
                f"{len(refused)} registered tree1 targets are not listed export candidates and would be "
                f"refused by the renderer before recording: {vertexes}"
            )
    return targets, {
        "targets_file": str(path),
        "targets_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        "selection_rule": document.get("selection_rule"),
        "seed": document.get("seed"),
    }


def submission_environment(environment):
    # Interactive allocation, old experiment, and sbatch option variables must
    # not become implicit job dependencies, resource requests or tracker modes.
    return {
        key: value
        for key, value in environment.items()
        if not key.startswith(("SLURM_", "SLURMD_", "SBATCH_", "PRUNING_"))
    }


def _git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args])


def freeze_source(root, destination, revision):
    """Archive committed files only, retaining modes and recording each digest."""
    archive = _git(root, "archive", "--format=tar", revision)
    destination.mkdir()
    hashes = {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
        for item in bundle:
            relative = Path(item.name)
            if relative.is_absolute() or ".." in relative.parts or not (item.isfile() or item.isdir()):
                raise ValueError(f"Unsupported source archive member: {item.name}")
            path = destination / relative
            if item.isdir():
                path.mkdir(parents=True, exist_ok=True)
                continue
            data = bundle.extractfile(item).read()
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("xb") as stream:
                stream.write(data)
            path.chmod(item.mode & 0o777)
            hashes[item.name] = hashlib.sha256(data).hexdigest()
    # Assets remain external and are independently checked by the existing
    # pinned-stack/robot preflight and scene exporter provenance.
    for relative in ("artifacts", "third_party/src"):
        source = root / relative
        if source.exists():
            (destination / relative).symlink_to(source.resolve(), target_is_directory=True)
    return hashes


def dependency_option(after):
    """Queue behind an earlier batch so the declared one-GPU-at-a-time limit holds.

    This constrains only the job being submitted. It never edits, holds, releases
    or requeues the job it waits on. ``afterany`` is deliberate: a failed earlier
    batch must still release this one, so its failures stay visible instead of
    leaving a batch stuck pending forever.
    """
    if after is None:
        return []
    if not re.fullmatch(r"[0-9]+", str(after)):
        raise ValueError("--after takes a numeric Slurm job id")
    return ["--dependency", f"afterany:{after}"]


def queue_batch(
    root,
    batch_id,
    targets=None,
    target_provenance=None,
    daylight="source",
    normalization="raw",
    after=None,
    strategy=None,
):
    root = root.resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", batch_id):
        raise ValueError("batch-id must be a short letters/digits/underscore/dash identifier")
    if _git(root, "diff", "HEAD", "--name-only").strip():
        raise ValueError("Commit tracked changes before freezing experiments; untracked work is preserved")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    plan = experiment_plan(targets, daylight, normalization, strategy)
    if target_provenance:
        plan["target_register"] = target_provenance
    plan["code_revision"] = revision
    plan["created_utc"] = datetime.now(timezone.utc).isoformat()
    env = submission_environment(os.environ)
    # Abort before submission if the queue cannot be observed. This never
    # changes priorities, dependencies, running allocations or pending jobs.
    queue_before = subprocess.check_output(
        ["squeue", "--user", os.environ["USER"], "--noheader", "--format=%i|%j|%T|%R"],
        env=env,
        timeout=30,
    ).decode()
    batch = root / "artifacts/vision_robustness" / batch_id
    batch.mkdir(parents=True, exist_ok=False)
    (batch / "logs").mkdir()
    hashes = freeze_source(root, batch / "code", revision)
    for name in ("hpc/slurm/vision_robustness.sbatch", "tools/run_vision_experiment.py"):
        if name not in hashes:
            raise ValueError(f"Required runner is not committed: {name}")
    plan["source_sha256"] = hashes
    plan["external_assets_sha256"] = inventory_assets(batch / "code")
    (batch / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (batch / "queue_before.txt").write_text(queue_before)
    # The array size follows the frozen plan rather than the sbatch directive, so a
    # sweep can never launch more or fewer tasks than it registered. Concurrency
    # stays at one GPU.
    command = [
        "sbatch",
        "--parsable",
        "--array",
        f"0-{len(plan['runs']) - 1}%1",
        "--time",
        trial_time_limit(plan),
        *dependency_option(after),
        "--chdir",
        str(batch / "code"),
        "--output",
        str(batch / "logs/%x-%A_%a.out"),
        "--error",
        str(batch / "logs/%x-%A_%a.out"),
        str(batch / "code/hpc/slurm/vision_robustness.sbatch"),
        str(batch),
    ]
    # Record intent before the only scheduler mutation. Never automatically
    # retry a submission: a client error can occur after Slurm accepts a job.
    (batch / "submission_intent.json").write_text(json.dumps(command, indent=2) + "\n")
    result = subprocess.run(command, env=env, text=True, capture_output=True, timeout=60, check=False)
    receipt = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    (batch / "submission_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    if result.returncode:
        raise RuntimeError(f"sbatch failed; inspect {batch}/submission_receipt.json before any retry")
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unrecognized sbatch receipt; inspect {batch}; do not resubmit blindly")
    # Return the plan that was actually frozen and submitted, not a fresh one.
    return {"array_job_id": job_id, "batch_dir": str(batch), "code_revision": revision, **plan}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-id", default="lighting-20260919")
    parser.add_argument("--submit", action="store_true", help="Submit once; the default only prints the plan")
    parser.add_argument("--targets-file", type=Path, help="Pre-registered target register; omit for the lighting pilot")
    parser.add_argument("--daylight", default="source", help="Daylight preset for a target sweep")
    parser.add_argument("--photometric-normalization", default="raw", help="Tracker preprocessing for a target sweep")
    parser.add_argument("--after", help="Queue behind this Slurm job id; that job is never modified")
    parser.add_argument("--strategy", choices=sorted(STRATEGIES), help="Labelled approach strategy for a target sweep")
    args = parser.parse_args()

    targets, provenance = (None, None)
    if args.targets_file is not None:
        targets, provenance = load_targets(args.targets_file)

    if args.submit:
        result = queue_batch(
            args.root,
            args.batch_id,
            targets,
            provenance,
            args.daylight,
            args.photometric_normalization,
            args.after,
            args.strategy,
        )
    else:
        result = experiment_plan(targets, args.daylight, args.photometric_normalization, args.strategy)
        if provenance:
            result["target_register"] = provenance
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
