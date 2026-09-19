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


def experiment_plan():
    return {
        "schema_version": 1,
        "scope": "Paired one-target lighting pilot; not a population success-rate estimate.",
        "max_concurrent_gpus": 1,
        "maximum_gpu_minutes": 150,
        "frames": 200,
        "fps": 10,
        "render_samples": 64,
        "overview_width": 1280,
        "runs": [
            {"index": index, "daylight": light, "photometric_normalization": mode}
            for index, (light, mode) in enumerate(
                (light, mode) for light in ("source", "morning", "evening") for mode in ("raw", "clahe")
            )
        ],
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


def queue_batch(root, batch_id):
    root = root.resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", batch_id):
        raise ValueError("batch-id must be a short letters/digits/underscore/dash identifier")
    if _git(root, "diff", "HEAD", "--name-only").strip():
        raise ValueError("Commit tracked changes before freezing experiments; untracked work is preserved")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    plan = experiment_plan()
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
    command = [
        "sbatch",
        "--parsable",
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
    return {"array_job_id": job_id, "batch_dir": str(batch), "code_revision": revision, **experiment_plan()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-id", default="lighting-20260919")
    parser.add_argument("--submit", action="store_true", help="Submit once; the default only prints the plan")
    args = parser.parse_args()
    result = queue_batch(args.root, args.batch_id) if args.submit else experiment_plan()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
