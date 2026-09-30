#!/usr/bin/env python3
"""Score the known-map re-oriented approach (protocol of September 27) against Q1-Q3, and describe its stops.

Outcomes come from the per-batch evidence that ``aggregate_eval.py`` wrote (the
grader's classes, never re-graded here). Each run's own records supply what a
clause names: the cut phases reached, the largest contact force, the approach
leg at a stop, and the GPU model (from the node in its report: ``cn-r-*`` and
``cn-s-*`` are A40, ``cn-gpu*`` are RTX 8000). A post hoc section, labelled as
such, describes the two stop mechanisms from recorded depth and ground truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from score_perception_round import clause, prediction  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = "docs/EVAL_PROTOCOL_PLANNED_APPROACH_2026-09-27.md"
EVIDENCE_DIR = "docs/evidence/planned_approach_2026-09-30"
BATCHES = ("planned-pose-gpu-r1-20260929", "planned-pose-gpu-r2-20260929", "planned-pose-gpu-r3-20260929")
PLANNED, CONTROLS = (530, 19444), (14944, 15004)
GEOMETRY_STOPS = ("stopped_hazard_contact", "stopped_tof_minimum_clearance")
CONTACT_LIMIT_N = 5.0


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def gpu_model(node):
    if node is None:
        return None
    if node.startswith(("cn-r-", "cn-s-")):
        return "A40"
    if node.startswith("cn-gpu"):
        return "RTX 8000"
    return "unknown"


def contact_max_n(frame):
    forces = ((frame.get("contact") or {}).get("forces_w_n") or [[]])[0]
    return max((math.sqrt(sum(v * v for v in f)) for f in forces), default=0.0)


def run_details(batch_dir, row):
    """Cut phases reached, the largest contact force, the leg and tracker reason at a stop, and the GPU model."""
    run = Path(batch_dir) / row["run_directory"]
    report = json.loads((run / "report.json").read_text())
    frames = json.loads((run / "frames.json").read_text())["frames"]
    phases, stop = [], None
    for frame in frames:
        live = frame.get("live_vision") or {}
        phase = (live.get("cut") or {}).get("phase")
        if phase and phase not in phases:
            phases.append(phase)
        if stop is None and (live.get("cut") or {}).get("stopped_reason"):
            measurement = live.get("measurement") or {}
            stop = {
                "decision_frame": frame["index"],
                "approach_leg": (frame.get("visual_servo_decision") or {}).get("approach_phase"),
                "tracker_state": measurement.get("state"),
                "tracker_reason": measurement.get("reason"),
            }
    return {
        "node": report.get("node"),
        "gpu_model": gpu_model(report.get("node")),
        "cut_phases_reached": phases,
        "reached_align": "align" in phases,
        "max_contact_n": round(max((contact_max_n(f) for f in frames), default=0.0), 3),
        "stop": stop,
    }


def load(root):
    runs = {}
    for batch in BATCHES:
        document = json.loads((root / EVIDENCE_DIR / f"{batch}.json").read_text())
        batch_dir = root / document["batches"][0]["batch_dir"]
        for row in document["runs"]:
            key = (batch, int(row["component_first_vertex"]))
            runs[key] = {**row, **run_details(batch_dir, row)}
    return runs


def score_q1(runs):
    clauses = []
    for vertex in PLANNED:
        rows = [runs[(batch, vertex)] for batch in BATCHES]
        observed = [
            {
                "batch": batch,
                "outcome": row["outcome"],
                "reached_align": row["reached_align"],
                "max_contact_n": row["max_contact_n"],
                "stop": row["stop"],
                "gpu_model": row["gpu_model"],
            }
            for batch, row in zip(BATCHES, rows)
        ]
        geometry_stop = [
            r["outcome"] in GEOMETRY_STOPS and not r["reached_align"] for r in rows
        ]  # the protocol's refutation
        holds = all(r["reached_align"] and r["max_contact_n"] <= CONTACT_LIMIT_N for r in rows)
        clauses.append(
            clause(
                f"{vertex} reaches align with no contact over 5 N and no ToF stop in 3 of 3",
                observed,
                holds,
                refutes=any(geometry_stop),
            )
        )
    return prediction(
        "Q1",
        "530 and 19444 reach align with no contact over 5 N and no ToF stop in 3 of 3 repeats each.",
        clauses,
    )


def score_q2(runs):
    observed = {f"{v} {b}": runs[(b, v)]["outcome"] for v in CONTROLS for b in BATCHES}
    failed = [k for k, outcome in observed.items() if outcome != "pass"]
    return prediction(
        "Q2",
        "14944 and 15004 pass 17 of 17 checks in 3 of 3 repeats under the identity plan.",
        [clause("every control repeat passes", observed, not failed, refutes=bool(failed))],
    )


def score_q3(runs):
    passes = [f"{v} {b}" for v in PLANNED for b in BATCHES if runs[(b, v)]["outcome"] == "pass"]
    return prediction(
        "Q3",
        "At least one of 530 and 19444 passes all 17 checks in at least one repeat.",
        [clause("a planned target passes in some repeat", {"passes": passes}, bool(passes), refutes=not passes)],
    )


def closure_drift(root, batch, run_directory):
    """Post hoc: during 530's closure the tool and spur are still; the measured target drifts."""
    frames = json.loads((root / "artifacts/vision_robustness" / batch / run_directory / "frames.json").read_text())[
        "frames"
    ]
    rows = []
    for frame in frames:
        if frame.get("phase") not in ("align", "simulated_closure", "stopped_failure"):
            continue
        measurement = (frame.get("live_vision") or {}).get("measurement") or {}
        world = measurement.get("target_position_world_m")
        rows.append(
            {
                "frame": frame["index"],
                "phase": frame["phase"],
                "jaw_closure_progress": frame.get("visual_jaw_closure_progress"),
                "tool_position_m": frame["tool_position_m"],
                "true_target_m": frame["target_position_m"],
                "measured_minus_true_mm": None
                if world is None
                else float(np.linalg.norm(np.subtract(world, frame["target_position_m"])) * 1000),
                "feature_count": measurement.get("feature_count"),
                "pixel_xy": measurement.get("pixel_xy"),
            }
        )
        if frame["phase"] == "stopped_failure":
            break
    tool0, target0 = rows[0]["tool_position_m"], rows[0]["true_target_m"]
    for row in rows:
        row["tool_moved_mm"] = float(np.linalg.norm(np.subtract(row.pop("tool_position_m"), tool0)) * 1000)
        row["true_target_moved_mm"] = float(np.linalg.norm(np.subtract(row.pop("true_target_m"), target0)) * 1000)
    return rows


def patch_near_surface(root, batch, run_directory, frames_before=4):
    """Post hoc: pixels of the 13 x 13 patch nearer than the tracked depth by over 1 cm, before 19444's stop."""
    run = root / "artifacts/vision_robustness" / batch / run_directory
    frames = json.loads((run / "frames.json").read_text())["frames"]
    stop = next(f["index"] for f in frames if ((f.get("live_vision") or {}).get("cut") or {}).get("stopped_reason"))
    rows = []
    for index in range(stop - frames_before, stop + 1):
        measurement = frames[index]["live_vision"]["measurement"]
        depth = np.load(run / f"frames/depth_{index:05d}.npy")
        x, y = np.rint(measurement["pixel_xy"]).astype(int)
        target = float(depth[y, x])
        window = depth[y - 6 : y + 7, x - 6 : x + 7]
        rows.append(
            {
                "frame": index,
                "patch_correlation": measurement.get("patch_correlation"),
                "tracked_depth_m": target,
                "patch_pixels_nearer_by_1cm": int(np.sum(window < target - 0.01)),
                "tracker_state": measurement.get("state"),
                "tracker_reason": measurement.get("reason"),
            }
        )
    return {"stop_decision_frame": stop, "frames": rows}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    root = args.root
    runs = load(root)
    run_dirs = {key: row["run_directory"] for key, row in runs.items()}
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"]))
    document = {
        "schema_version": 1,
        "scope": (
            "Scoring of the known-map re-oriented approach (Q1-Q3) from the grader's outcomes of 12 planned simulator "
            "runs in three batches. Results are labelled 'known-map plan, jaw orientation set at the planned pose' "
            "and are never pooled with unchanged-gate results. Simulator ground truth; not a field success rate."
        ),
        "protocol": PROTOCOL,
        "code_revision": head,
        "code_tree_dirty": dirty,
        "inputs_sha256": {f"{EVIDENCE_DIR}/{b}.json": sha256(root / EVIDENCE_DIR / f"{b}.json") for b in BATCHES},
        "runs": [
            {"batch": batch, "target": vertex, **{k: row[k] for k in ("outcome", "checks_passed", "checks_total")}}
            | {k: row[k] for k in ("node", "gpu_model", "reached_align", "cut_phases_reached", "max_contact_n", "stop")}
            for (batch, vertex), row in sorted(runs.items())
        ],
        "predictions": [score_q1(runs), score_q2(runs), score_q3(runs)],
        "post_hoc": {
            "label": "post hoc diagnosis from recorded ground truth and depth, not a prediction",
            "530_closure": {b: closure_drift(root, b, run_dirs[(b, 530)]) for b in BATCHES},
            "19444_patch_occlusion": {b: patch_near_surface(root, b, run_dirs[(b, 19444)]) for b in BATCHES},
        },
    }
    serialized = json.dumps(document, indent=2, allow_nan=False, default=str) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({p["id"]: p["verdict"] for p in document["predictions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
