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
    # A strategy row names its approach; a plan without one keeps the renderer's
    # baseline defaults, so the September 23 frozen plans are unaffected.
    if "strategy" in row:
        strategy = row["strategy"]
        env.update(
            PRUNING_APPROACH_MODE=str(strategy["mode"]),
            PRUNING_STANDOFF_M=repr(float(strategy["standoff_m"])),
            PRUNING_MAX_STEP_M=repr(float(strategy["max_step_m"])),
        )
        if "motion_model" in strategy:
            env["PRUNING_MOTION_MODEL"] = str(strategy["motion_model"])
        if "mount_side_rule" in strategy:
            env["PRUNING_MOUNT_SIDE_RULE"] = str(strategy["mount_side_rule"])
        if "max_rotation_deg" in strategy:
            env["PRUNING_MAX_ROTATION_DEG"] = repr(float(strategy["max_rotation_deg"]))
        if strategy.get("planned_tool_quat_wxyz") is not None:
            env["PRUNING_PLANNED_TOOL_QUAT"] = ",".join(repr(float(v)) for v in strategy["planned_tool_quat_wxyz"])
        if "jaw_casts_shadow" in strategy:
            env["PRUNING_JAW_CASTS_SHADOW"] = "1" if strategy["jaw_casts_shadow"] else "0"
        if "jaw_self_mask" in strategy:
            env["PRUNING_JAW_SELF_MASK"] = "1" if strategy["jaw_self_mask"] else "0"
        if "closure_hold" in strategy:
            env["PRUNING_CLOSURE_HOLD"] = "1" if strategy["closure_hold"] else "0"
        if "depth_appearance" in strategy:
            env["PRUNING_DEPTH_APPEARANCE"] = "1" if strategy["depth_appearance"] else "0"
    return env


def run_label(index, row):
    """Name the output directory after the condition that produced it."""
    suffix = f"_{row['strategy']['name']}" if "strategy" in row else ""
    if "component_first_vertex" in row:
        label = f"run_{index:02d}_{row['daylight']}_tree{row['target_tree_index']}_v{row['component_first_vertex']}"
        return label + suffix
    return f"run_{index:02d}_{row['daylight']}_{row['photometric_normalization']}" + suffix


def _registered_jaw_arms():
    """The labelled arms' registered constants from the frozen source, JSON-normalized like a report."""
    try:
        from isaaclab_pruning.perception.depth_appearance import registered_depth_appearance
        from isaaclab_pruning.perception.jaw_self_mask import registered_closure_hold, registered_jaw_self_mask
    except ImportError:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "isaaclab_pruning"))
        from isaaclab_pruning.perception.depth_appearance import registered_depth_appearance
        from isaaclab_pruning.perception.jaw_self_mask import registered_closure_hold, registered_jaw_self_mask
    return json.loads(
        json.dumps(
            {
                "jaw_self_mask": registered_jaw_self_mask(),
                "closure_hold": registered_closure_hold(),
                "depth_appearance": registered_depth_appearance(),
            }
        )
    )


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
    if "strategy" in row:
        recorded = report.get("approach_strategy") or {}
        wanted = row["strategy"]
        matches = matches and all(recorded.get(key) == wanted[key] for key in ("mode", "standoff_m", "max_step_m"))
        if "motion_model" in wanted:
            tracker = report.get("tracker_config") or {}
            matches = matches and tracker.get("motion_model") == wanted["motion_model"]
        if "mount_side_rule" in wanted:
            mount = report.get("camera_mount_selection") or {}
            matches = matches and mount.get("rule") == wanted["mount_side_rule"]
        if "jaw_casts_shadow" in wanted:
            shadow = report.get("blender_scene", {}).get("jaw_proxy_shadow") or {}
            readback = shadow.get("do_not_cast_shadows_readback") or []
            if wanted["jaw_casts_shadow"]:
                matches = matches and shadow.get("casts_shadow_requested") is True and not any(readback)
            else:
                matches = (
                    matches
                    and shadow.get("casts_shadow_requested") is False
                    and len(readback) == 2
                    and all(value is True for value in readback)
                )
        for flag, registered in (
            ("jaw_self_mask", "constants"),
            ("closure_hold", "thresholds"),
            ("depth_appearance", "constants"),
        ):
            if flag in wanted:
                block = report.get(flag) or {}
                matches = matches and block.get("enabled") is bool(wanted[flag])
                if wanted[flag]:
                    # The capture must also show the registered constants, not merely the flag.
                    matches = matches and block.get(registered) == _registered_jaw_arms()[flag]
            else:
                # A row that does not ask for the arm must not have run with it.
                matches = matches and (report.get(flag) or {}).get("enabled") is not True
        if wanted.get("jaw_self_mask"):
            # The tracker's own minimum is the one that decided, not only the reported constant.
            tracker = report.get("tracker_config") or {}
            expected = _registered_jaw_arms()["jaw_self_mask"]["min_unmasked_patch_elements"]
            matches = matches and tracker.get("min_unmasked_patch_elements") == expected
        if "planned_tool_quat_wxyz" in wanted:
            recorded_quat = recorded.get("planned_tool_quat_wxyz")
            wanted_quat = wanted["planned_tool_quat_wxyz"]
            if wanted_quat is None:
                matches = matches and recorded_quat is None
            else:
                matches = (
                    matches
                    and recorded_quat is not None
                    and all(abs(float(a) - float(b)) < 1e-6 for a, b in zip(recorded_quat, wanted_quat, strict=True))
                )
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
