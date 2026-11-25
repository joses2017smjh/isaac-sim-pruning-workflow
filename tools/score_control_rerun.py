#!/usr/bin/env python3
"""Score the control rerun against docs/EVAL_PROTOCOL_CONTROL_RERUN_2026-11-25.md (R0 to R5). CPU only.

The jaw self-mask and closure-hold arm had one hold-engaging control, target 530, and every recording of it has the
robot beneath the orchard's ground (docs/GROUND_AUDIT_2026-11-25.md). This scores four registered runs of the same
arm on spurs the robot reaches from above the ground: three repeats on target 21386, and one run on control 14944
with the ground relief added to the ToF sensors' mesh list, the first simulator run of that switch.

It reads what the runs recorded (experiment result, report, frames and sequence grade) and checks the geometry
offline with the tools the ground audit used: tools/audit_ground.py for what is beneath the ground, and the offline
ToF cast with and without the ground. Committed with the protocol before any registered run exists.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "tools", ROOT / "source/isaaclab_pruning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

from run_vision_experiment import run_label  # noqa: E402

SCHEMA_VERSION = 1
PROTOCOL = "docs/EVAL_PROTOCOL_CONTROL_RERUN_2026-11-25.md"
REPEAT_BATCHES = ("control-rerun-a-20261125", "control-rerun-b-20261125", "control-rerun-c-20261125")
TOF_BATCH = "control-rerun-tof-ground-20261125"
STRATEGIES = {
    **{name: "planned_pose_jaw_hold_ctl" for name in REPEAT_BATCHES},
    TOF_BATCH: "planned_pose_jaw_hold_ctl_tof_ground",
}
TARGETS = {**{name: "tree0_SPUR_component_21386" for name in REPEAT_BATCHES}, TOF_BATCH: "tree1_SPUR_component_14944"}
#: The October 6 capture of the same target under the same arm: what the recorder's mesh list is compared with.
REFERENCE = {
    "tree0_SPUR_component_21386": "hold-sweep-20261006/run_14_source_tree0_v21386_planned_pose_jaw_hold_sweep",
    "tree1_SPUR_component_14944": "hold-sweep-ctl-20261006/run_01_source_tree1_v14944_planned_pose_jaw_hold_sweep",
}
FRAMES = 200
CHECKS = 17
#: The hold engages two frames after closure starts and the piece detaches six frames after, as on October 2 and 6.
FIRST_HELD_OFFSET = 2
DETACH_OFFSET = 6
#: The offline cast reproduces a recorded ToF grid when this share of zones agree on validity and this share of the
#: zones valid in both agree within 1 mm (the criterion of tools/study_mouth_evidence.py).
TOF_VALIDITY = 0.98
TOF_WITHIN_1MM = 0.95
GROUND_PREFIX = "/World/Orchard/ground"


def _load(path, default=None):
    path = Path(path)
    return json.loads(path.read_text()) if path.is_file() else default


def _cut(frame):
    return (frame.get("live_vision") or {}).get("cut") or {}


def hold_structure(frames):
    """Closure start, the held frames and the detach frame, and whether they are the registered pattern."""
    start = next((f["index"] for f in frames if _cut(f).get("phase") == "closing"), None)
    held = [f["index"] for f in frames if ((f.get("live_vision") or {}).get("closure_hold") or {}).get("held")]
    detach = next((f["index"] for f in frames if _cut(f).get("detach_event")), None)
    registered = (
        start is not None
        and held == list(range(start + FIRST_HELD_OFFSET, start + DETACH_OFFSET + 1))
        and detach == start + DETACH_OFFSET
    )
    return {"closure_start": start, "held_frames": held, "detach_frame": detach, "as_registered": bool(registered)}


def tof_agreement(check):
    """Shares from a tools/study_mouth_evidence.py ToF comparison, and whether they meet the criterion."""
    validity = check["validity_same"] / max(check["zones"], 1)
    within = check["within_1mm_as_ray_length"] / max(check["both_valid"], 1)
    return {
        "zones": check["zones"],
        "validity_agreement": round(validity, 5),
        "within_1mm": round(within, 5),
        "reproduced": bool(validity >= TOF_VALIDITY and within >= TOF_WITHIN_1MM),
    }


def first_tof_stop(frames):
    return next((f["index"] for f in frames if f.get("sensor_stop_reason")), None)


def score_run(batch, row, geometry, reference_paths):
    """One registered run. ``geometry(run_dir, report, frames, last)`` does the offline casts."""
    run_dir = batch / run_label(row["index"], row)
    result = _load(run_dir / "experiment_result.json", {})
    report = _load(run_dir / "report.json", {})
    document = _load(run_dir / "frames.json", {})
    frames = (document.get("frames") if isinstance(document, dict) else document) or []
    grade = _load(run_dir / "sequence_grade.json", {})
    scene = report.get("blender_scene") or {}
    target = (scene.get("target") or {}).get("id")
    unscored = []
    if not frames or not report:
        unscored.append("no_recording")
    if result.get("configuration_matches") is not True:
        unscored.append("configuration_does_not_match_the_plan")
    if target != TARGETS[batch.name]:
        unscored.append("not_the_registered_target")
    if len(frames) != FRAMES or [f.get("index") for f in frames] != list(range(FRAMES)):
        unscored.append(f"capture_is_not_{FRAMES}_frames_in_order")
    hold = hold_structure(frames)
    out = {
        "batch": batch.name,
        "run": run_dir.name,
        "target": target,
        "strategy": (row.get("strategy") or {}).get("name"),
        "node": report.get("node"),
        "job_id": report.get("job_id"),
        "hold": hold,
        "first_tof_stop_frame": first_tof_stop(frames),
        "task_outcome": report.get("task_outcome"),
        "failed_checks": grade.get("failed_checks"),
        "tof_ground_mesh_paths": scene.get("tof_ground_mesh_paths"),
        "collision_meshes_as_on_october_6": scene.get("collision_mesh_paths") == reference_paths.get(target),
    }
    checks = grade.get("checks") or {}
    out["pass"] = result.get("ok") is True and len(checks) == CHECKS and all(checks.values())
    if "no_recording" not in unscored:
        last = hold["detach_frame"] if hold["detach_frame"] is not None else len(frames) - 1
        geo = geometry(run_dir, report, frames, last)
        out["ground"] = geo["ground"]
        out["tof_trees_only"] = tof_agreement(geo["tof_trees_only"])
        out["tof_with_ground"] = tof_agreement(geo["tof_with_ground"])
        if geo["ground"]["class"] != "above" or not geo["ground"]["kinematics_ok"]:
            unscored.append("part_of_the_robot_is_beneath_the_ground")
    out["scored"], out["unscored_reasons"] = not unscored, unscored
    return out


def judge(runs, holds, expected):
    """Supported only if every registered run is scored and holds; refuted by any scored run that does not."""
    scored = [run for run in runs if run["scored"]]
    departures = [f"{run['batch']}/{run['run']}" for run in scored if not holds(run)]
    if departures:
        verdict = "refuted"
    elif len(scored) == len(runs) == expected:
        verdict = "supported"
    else:
        verdict = "not supported: unscored runs"
    return {"verdict": verdict, "departures": departures, "scored": len(scored), "runs": len(runs)}


def verdicts(runs):
    repeats = [run for run in runs if run["batch"] in REPEAT_BATCHES]
    switch = [run for run in runs if run["batch"] == TOF_BATCH]

    def ground_listed(run):
        paths = run["tof_ground_mesh_paths"] or []
        return bool(paths) and all(str(path).startswith(GROUND_PREFIX) for path in paths)

    return {
        "R0_physical_scenes": {
            "verdict": "supported" if runs and all(run["scored"] for run in runs) else "refuted",
            "scored": sum(run["scored"] for run in runs),
            "runs": len(runs),
            "unscored": {f"{run['batch']}/{run['run']}": run["unscored_reasons"] for run in runs if not run["scored"]},
        },
        "R1_three_repeats_pass": {
            "prediction": "each repeat on 21386 passes all 17 checks",
            **judge(repeats, lambda r: r["pass"], len(REPEAT_BATCHES)),
        },
        "R2_hold_engages_as_before": {
            "prediction": "the hold covers closure start + 2 to + 6 and the piece detaches at closure start + 6",
            **judge(repeats, lambda r: r["hold"]["as_registered"], len(REPEAT_BATCHES)),
        },
        "R3_switch_off_changes_nothing": {
            "prediction": "the repeats list the October 6 collision meshes, no ground mesh, and their ToF grids are "
            "the trees-only cast",
            **judge(
                repeats,
                lambda r: r["collision_meshes_as_on_october_6"]
                and not r["tof_ground_mesh_paths"]
                and r["tof_trees_only"]["reproduced"],
                len(REPEAT_BATCHES),
            ),
        },
        "R4_switch_on_adds_the_ground": {
            "prediction": "the run on 14944 lists the ground mesh, its ToF grids are the cast with the ground, and "
            "the trees-only cast no longer reproduces them",
            **judge(
                switch,
                lambda r: ground_listed(r)
                and r["collision_meshes_as_on_october_6"]
                and r["tof_with_ground"]["reproduced"]
                and not r["tof_trees_only"]["reproduced"],
                1,
            ),
        },
        "R5_switch_on_changes_no_outcome": {
            "prediction": "the run on 14944 passes all 17 checks, never holds and has no ToF stop",
            **judge(
                switch,
                lambda r: r["pass"] and not r["hold"]["held_frames"] and r["first_tof_stop_frame"] is None,
                1,
            ),
        },
    }


def offline_geometry():
    """The offline casts, built once: needs the exported orchard, so it runs where pxr is available."""
    import audit_ground
    import known_map_planner as kmp
    import study_mouth_evidence as study

    export = kmp.stream_export(kmp.EXPORT_DEFAULT, kmp.ISAAC_PYTHON_DEFAULT, {})
    names = [str(name) for name in export["names"]]
    ground = study.stream_ground()
    robot, _ = kmp.load_robot(kmp.URDF_DEFAULT)

    def geometry(run_dir, report, frames, last):
        trees = study.build_scene(export, names, report)
        full = study.build_scene(export, names, report, ground=ground)
        row = audit_ground.audit_capture(run_dir, frames, trees, full, robot)
        return {
            "ground": {
                "class": audit_ground.classify(row),
                "anything_beneath_frames": row["anything_beneath_frames"],
                "kinematics_ok": row["kinematics"]["ok"],
                "max_tool_point_error_mm": row["kinematics"]["max_tool_point_error_mm"],
            },
            "tof_trees_only": study.validate_tof(frames, last, trees),
            "tof_with_ground": study.validate_tof(frames, last, full),
        }

    return geometry


def reference_paths(runs_root):
    out = {}
    for target, run in REFERENCE.items():
        report = _load(Path(runs_root) / run / "report.json", {})
        out[target] = (report.get("blender_scene") or {}).get("collision_mesh_paths")
        if not out[target]:
            raise ValueError(f"the October 6 reference capture for {target} has no collision mesh list")
    return out


def score(batches, geometry, references):
    runs = []
    for batch in sorted(Path(b) for b in batches):
        plan = _load(batch / "plan.json")
        if plan is None or plan.get("protocol") != PROTOCOL or batch.name not in STRATEGIES:
            raise ValueError(f"{batch} is not a batch registered to {PROTOCOL}")
        if (plan.get("strategy") or {}).get("name") != STRATEGIES[batch.name] or len(plan["runs"]) != 1:
            raise ValueError(f"{batch}: not the registered arm, or not exactly one run")
        runs.append(score_run(batch, plan["runs"][0], geometry, references))
    if sorted(run["batch"] for run in runs) != sorted(STRATEGIES):
        raise ValueError("the four registered batches must be scored together")
    return {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/score_control_rerun.py",
        "protocol": PROTOCOL,
        "registered": {
            "repeat_batches": list(REPEAT_BATCHES),
            "tof_ground_batch": TOF_BATCH,
            "targets": TARGETS,
            "reference_captures": REFERENCE,
            "first_held_offset": FIRST_HELD_OFFSET,
            "detach_offset": DETACH_OFFSET,
            "tof_validity": TOF_VALIDITY,
            "tof_within_1mm": TOF_WITHIN_1MM,
        },
        "verdicts": verdicts(runs),
        "runs": runs,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--batches", required=True, nargs="+", type=Path)
    parser.add_argument("--runs-root", type=Path, default=ROOT / "artifacts/vision_robustness")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"refusing to overwrite {args.output}")
    result = score(args.batches, offline_geometry(), reference_paths(args.runs_root))
    args.output.write_text(json.dumps(result, indent=1, sort_keys=True) + "\n")
    print(json.dumps({name: item["verdict"] for name, item in result["verdicts"].items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
