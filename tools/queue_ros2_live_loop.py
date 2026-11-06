#!/usr/bin/env python3
"""Freeze and submit ROS 2 live-loop trials: a CPU workspace build, then one GPU task per run.

Each run is the known target, ``tree1_SPUR_component_14944`` under the source
daylight with the unchanged baseline controller (straight approach, 4 mm steps,
no labelled arms), driven through ROS 2: Isaac Sim publishes sensors, the
controller and the C++ safety gate run as ROS 2 nodes, and the arm moves only on
the gated command. A row's ``delay_steps`` is the modelled message delay, in
1/60 s control steps of simulated time (0, 2, 4, 6 = 0, 33, 67, 100 ms). A
``panel`` row also runs the Qt/QML operator panel, which clicks its own Hold
button after ``operator_hold_after_cycles`` gate decisions.

Without ``--submit`` this prints the plan and changes nothing. Submission
follows tools/queue_vision_robustness.py: committed files only, a storage
preflight, the intent written before the one scheduler call, no retries.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from queue_vision_robustness import (  # noqa: E402
    _git,
    freeze_source,
    submission_environment,
    vision_storage_preflight,
)
from vision_experiment_assets import inventory_assets  # noqa: E402

#: The known target: seven of seven in-process passes (17/17) under exactly this condition.
KNOWN_TARGET = {
    "daylight": "source",
    "photometric_normalization": "raw",
    "target_tree_index": 1,
    "component_first_vertex": 14944,
    "strategy": {"name": "baseline", "mode": "straight", "standoff_m": 0.0, "max_step_m": 0.004},
}
CONTROL_STEP_S = 1.0 / 60.0
DESIGNS = {
    # Exploratory: does one live ROS 2 run work end to end? Not part of the registered result.
    "smoke": [(0, None)],
    # Registered (docs/EVAL_PROTOCOL_ROS2_LIVE_LOOP_2026-11-05.md): three repeats of four delays,
    # interleaved so no delay runs on a different day or queue position than the others, then the panel.
    "delay_sweep": [(delay, None) for _ in range(3) for delay in (0, 2, 4, 6)] + [(0, 60)],
}
PROTOCOL = "docs/EVAL_PROTOCOL_ROS2_LIVE_LOOP_2026-11-05.md"
REQUIRED = (
    "hpc/slurm/ros2_live_loop.sbatch",
    "hpc/slurm/ros2_ws_build.sbatch",
    "hpc/inner/ros2_loop_bridge.py",
    "tools/run_vision_experiment.py",
    "ros2/pruning_sil/pruning_sil/live_bringup.py",
    "ros2/pruning_sil/pruning_sil/live_controller.py",
    "ros2/pruning_sil_cpp/src/safety_gate_node.cpp",
)
DEFAULT_IMAGE = "/nfs/hpc/share/sanchej7/containers/pruning_sil_humble_qt.sif"
MINUTES_PER_TASK = 35
PROJECT_ID = "30762"


def experiment_plan(design):
    if design not in DESIGNS:
        raise ValueError(f"Unknown design {design!r}; registered: {sorted(DESIGNS)}")
    runs = []
    for index, (delay, hold_after) in enumerate(DESIGNS[design]):
        runs.append(
            {
                "index": index,
                **json.loads(json.dumps(KNOWN_TARGET)),
                "ros2_loop": {
                    "delay_steps": delay,
                    "delay_ms": round(delay * CONTROL_STEP_S * 1000.0, 1),
                    "operator_hold_after_cycles": hold_after,
                },
            }
        )
    return {
        "schema_version": 1,
        "design": design,
        "max_concurrent_gpus": 1,
        "frames": 200,
        "fps": 10,
        "render_samples": 64,
        "overview_width": 1280,
        "scope": (
            "The known target driven through a live ROS 2 loop in Isaac Sim, with modelled message delay. "
            "Simulation only; one target and one light; not a hardware or latency-guarantee result."
        ),
        "protocol": PROTOCOL if design != "smoke" else "exploratory smoke run; not part of the registered result",
        "maximum_gpu_minutes": MINUTES_PER_TASK * len(runs),
        "runs": runs,
    }


def share_used_bytes():
    """Project usage from the Lustre project quota (the share's real accounting)."""
    output = subprocess.check_output(["lfs", "quota", "-p", PROJECT_ID, "/nfs/hpc/share"], text=True, timeout=60)
    for line in output.splitlines():
        fields = line.split()
        if fields and fields[0] == "/nfs/hpc/share":
            return int(fields[1].rstrip("*")) * 1024
    raise RuntimeError(f"Could not read project usage:\n{output}")


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def queue(root, batch_id, design, image, after=None, submit=False):
    root = root.resolve()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", batch_id):
        raise ValueError("batch-id must be a short letters/digits/underscore/dash identifier")
    plan = experiment_plan(design)
    plan["sil_image"] = str(image)
    plan["placement"] = {"partition": "gpu,ampere", "constraint": "a40|rtx8000", "minutes_per_task": MINUTES_PER_TASK}
    if not submit:
        print(json.dumps(plan, indent=2))
        return None
    if _git(root, "diff", "HEAD", "--name-only").strip():
        raise ValueError("Commit tracked changes before freezing experiments; untracked work is preserved")
    revision = _git(root, "rev-parse", "HEAD").decode().strip()
    plan["code_revision"] = revision
    plan["created_utc"] = datetime.now(timezone.utc).isoformat()
    plan["sil_image_sha256"] = sha256(image)
    env = submission_environment(os.environ)
    queue_before = subprocess.check_output(
        ["squeue", "--user", os.environ["USER"], "--noheader", "--format=%i|%j|%T|%R"], env=env, timeout=30
    ).decode()
    batch = root / "artifacts/vision_robustness" / batch_id
    batch.mkdir(parents=True, exist_ok=False)
    (batch / "logs").mkdir()
    hashes = freeze_source(root, batch / "code", revision)
    for name in REQUIRED:
        if name not in hashes:
            raise ValueError(f"Required file is not committed: {name}")
    plan["source_sha256"] = hashes
    plan["external_assets_sha256"] = inventory_assets(batch / "code")
    (batch / "plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    (batch / "queue_before.txt").write_text(queue_before)
    vision_storage_preflight(batch, plan, share_used_bytes())

    build = [
        "sbatch", "--parsable", "--chdir", str(batch / "code"),
        "--output", str(batch / "logs/%x-%j.out"), "--error", str(batch / "logs/%x-%j.out"),
        *(["--dependency", f"afterany:{after}"] if after else []),
        str(batch / "code/hpc/slurm/ros2_ws_build.sbatch"), str(batch),
    ]  # fmt: skip
    intent = {"build": build, "trials": "submitted after the build is accepted, with --dependency afterok:<build>"}
    (batch / "submission_intent.json").write_text(json.dumps(intent, indent=2) + "\n")
    result = subprocess.run(build, env=env, text=True, capture_output=True, timeout=60, check=False)
    receipt = {"build": {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}}
    (batch / "submission_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    build_id = result.stdout.strip().split(";")[0]
    if result.returncode or not build_id.isdigit():
        raise RuntimeError(f"Build submission failed; inspect {batch}/submission_receipt.json before any retry")
    trials = [
        "sbatch", "--parsable", "--array", f"0-{len(plan['runs']) - 1}%1",
        "--time", f"00:{MINUTES_PER_TASK:02d}:00", "--partition", "gpu,ampere", "--constraint", "a40|rtx8000",
        "--dependency", f"afterok:{build_id}", "--chdir", str(batch / "code"),
        "--output", str(batch / "logs/%x-%A_%a.out"), "--error", str(batch / "logs/%x-%A_%a.out"),
        str(batch / "code/hpc/slurm/ros2_live_loop.sbatch"), str(batch),
    ]  # fmt: skip
    intent["trials"] = trials
    (batch / "submission_intent.json").write_text(json.dumps(intent, indent=2) + "\n")
    result = subprocess.run(trials, env=env, text=True, capture_output=True, timeout=60, check=False)
    receipt["trials"] = {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}
    (batch / "submission_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    array_id = result.stdout.strip().split(";")[0]
    if result.returncode or not array_id.isdigit():
        raise RuntimeError(f"Trial submission failed; inspect {batch}/submission_receipt.json before any retry")
    return {"batch_dir": str(batch), "build_job_id": build_id, "array_job_id": array_id, "code_revision": revision}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--design", choices=sorted(DESIGNS), required=True)
    parser.add_argument("--image", type=Path, default=Path(DEFAULT_IMAGE))
    parser.add_argument("--after", help="Start the build only after this job ends (afterany); it is never modified")
    parser.add_argument("--submit", action="store_true", help="Freeze and submit; the default only prints the plan")
    args = parser.parse_args()
    result = queue(args.root, args.batch_id, args.design, args.image, args.after, args.submit)
    if result:
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
