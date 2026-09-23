#!/usr/bin/env python3
"""Execute one frozen pilot condition and preserve independent failed grades."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

from validate_vision_sequence import grade_sequence
from vision_experiment_assets import verify_assets


def run_environment(plan, row, batch, output, inherited):
    env = {key: value for key, value in inherited.items() if not key.startswith("PRUNING_")}
    env.update(
        PRUNING_ROOT=str(batch / "code"),
        PRUNING_RENDER_DIR=str(output),
        PRUNING_RENDER_MODE="blender_vision",
        PRUNING_RENDER_QUALITY="pathtraced",
        PRUNING_RENDER_SAMPLES=str(plan["render_samples"]),
        PRUNING_OVERVIEW_WIDTH=str(plan["overview_width"]),
        PRUNING_RENDER_FRAMES=str(plan["frames"]),
        PRUNING_DAYLIGHT=row["daylight"],
        PRUNING_PHOTOMETRIC_NORMALIZATION=row["photometric_normalization"],
        PRUNING_BLENDER_SCENE_DIR=str(batch / "code/artifacts/blender_scene/orchard_two_trees_v1"),
        PRUNING_RUN_ENV_SMOKE="0",
        BENCH_OUT=str(output / "smoke.json"),
    )
    # A target-sweep plan names its spur explicitly. A lighting plan omits both
    # keys and keeps the renderer's own defaults, so older frozen plans are
    # unaffected. The pair is always set together or not at all.
    if "component_first_vertex" in row:
        env.update(
            PRUNING_TARGET_TREE=str(row["target_tree_index"]),
            PRUNING_COMPONENT_VERTEX=str(row["component_first_vertex"]),
        )
    return env


def run_label(index, row):
    """Name the output directory after the condition that produced it."""
    if "component_first_vertex" in row:
        return f"run_{index:02d}_{row['daylight']}_tree{row['target_tree_index']}_v{row['component_first_vertex']}"
    return f"run_{index:02d}_{row['daylight']}_{row['photometric_normalization']}"


def configuration_matches(report, row, plan):
    """Confirm the capture really used the frozen condition, including the target."""
    matches = (
        report.get("photometric_normalization") == row["photometric_normalization"]
        and report.get("frame_count") == plan["frames"]
        and report.get("blender_scene", {}).get("daylight", {}).get("preset") == row["daylight"]
    )
    if "component_first_vertex" in row:
        expected = f"tree{row['target_tree_index']}_SPUR_component_{row['component_first_vertex']}"
        matches = matches and report.get("blender_scene", {}).get("target", {}).get("id") == expected
    return matches


def verify_snapshot(batch, plan):
    for name, digest in plan["source_sha256"].items():
        if hashlib.sha256((batch / "code" / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"Frozen source changed: {name}")


def execute(batch, index):
    batch = batch.resolve()
    plan = json.loads((batch / "plan.json").read_text())
    if index < 0 or index >= len(plan["runs"]):
        raise ValueError("experiment index is outside the frozen plan")
    row = plan["runs"][index]
    output = batch / run_label(index, row)
    output.mkdir(exist_ok=False)
    result = {"condition": row, "code_revision": plan["code_revision"], "job_id": os.environ.get("SLURM_JOB_ID")}
    try:
        verify_snapshot(batch, plan)
        verify_assets(plan["external_assets_sha256"])
        env = run_environment(plan, row, batch, output, os.environ)
        process = subprocess.run(
            ["bash", str(batch / "code/hpc/inner/render_pruning_workflow.sh")], env=env, check=False
        )
        result["capture_exit_code"] = process.returncode
        report = json.loads((output / "report.json").read_text())
        frames = json.loads((output / "frames.json").read_text())
        grade = grade_sequence(report, frames)
        (output / "sequence_grade.json").write_text(json.dumps(grade, indent=2, allow_nan=False) + "\n")
        result["capture_ok"] = process.returncode == 0
        result["sequence_ok"] = grade["ok"]
        result["task_outcome"] = report.get("task_outcome")
        result["configuration_matches"] = configuration_matches(report, row, plan)
        result["ok"] = result["capture_ok"] and result["sequence_ok"] and result["configuration_matches"]
    except Exception:
        result.update(ok=False, traceback=traceback.format_exc())
    (output / "experiment_result.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return 0 if result["ok"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", required=True, type=Path)
    parser.add_argument("--index", required=True, type=int)
    args = parser.parse_args()
    return execute(args.batch_dir, args.index)


if __name__ == "__main__":
    sys.exit(main())
