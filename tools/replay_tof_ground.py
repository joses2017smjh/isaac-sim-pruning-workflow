#!/usr/bin/env python3
"""What the simulated ToF sensors would have read with the ground relief in their mesh set. CPU only.

The simulated ToF sensors cast against the trees, posts and wires. The orchard's ground relief is drawn but was
never in that list. ``PRUNING_TOF_GROUND=1`` can now add it (isaaclab_pruning.sim.tof_targets). This tool measures
offline what that switch would change in the recordings; it was written before any simulator run had used it.

For every recorded frame it casts both 8x8 grids twice at the recorded sensor poses: against the meshes the sensors
used, and against those plus the ground. The first cast is the check: it must reproduce the recorded grids. Then it
replays the Python reference gate over each capture with each set of grids (tools/gate_replay.py) and compares the
first hold.

It is a replay. The controller's messages are the recorded ones, so a hold that moves earlier does not change what
the controller did next; only the first difference in each capture means anything. Captures in which the robot is
beneath the ground (tools/audit_ground.py) are reported separately: there the question is whether a sensor that saw
the ground would have stopped the arm, and when.

Writes one new JSON file outside the repository; numbers and capture names only.
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for extra in (ROOT / "tools", ROOT / "source/isaaclab_pruning"):
    if str(extra) not in sys.path:
        sys.path.insert(0, str(extra))

import cast_jaw_beams as cjb  # noqa: E402
import export_gate_parity as egp  # noqa: E402
import gate_replay  # noqa: E402
import known_map_planner as kmp  # noqa: E402
import orchard_ground  # noqa: E402
import study_mouth_evidence as study  # noqa: E402

SCHEMA_VERSION = 1
AGREE_M = 0.001
ZONES = 64


def cast_grid(frame, name, scene):
    """Ray lengths of one sensor's 64 zones at its recorded pose, NaN where out of range, and what each hit."""
    state = frame["tof_state"]["sensors"][name]
    origin = np.asarray(state["position_w_m"][0], float)
    x, y, z, w = state["quaternion_w_ros_xyzw"][0]
    directions = kmp.TOF_DIRS @ study.rot([w, x, y, z]).T
    t, g, _ = scene.all.cast(np.repeat(origin[None], ZONES, 0), directions)
    t = np.asarray(t, float)
    valid = np.isfinite(t) & (t >= kmp.TOF_RANGE_M[0]) & (t <= kmp.TOF_RANGE_M[1])
    labels = [scene.labels[int(i)] if ok else None for i, ok in zip(g, valid)]
    return np.where(valid, t, np.nan), labels


def as_record_grid(grid):
    """A grid as the cycle records carry it: metres, None where there is no usable zone."""
    return [None if not math.isfinite(v) else float(v) for v in grid]


def recorded_grid(record, key):
    return np.array([np.nan if v is None else v for v in record[key]], float)


def compare_grids(cast, recorded):
    """How well a trees-only cast reproduces a recorded grid."""
    cast_valid, recorded_valid = np.isfinite(cast), np.isfinite(recorded)
    both = cast_valid & recorded_valid
    return {
        "zones": int(cast.size),
        "validity_same": int((cast_valid == recorded_valid).sum()),
        "both_valid": int(both.sum()),
        "within_1mm": int((np.abs(cast[both] - recorded[both]) <= AGREE_M).sum()),
    }


def replay_capture(records, frames, trees, full):
    """Grids with and without the ground for one capture, and the gate's first hold under each."""
    with_trees, with_ground = [], []
    check = Counter()
    changed_zones = gained = nearer = ground_frames = 0
    nearest_ground = math.inf
    for record, frame in zip(records, frames):
        if int(frame["index"]) != record["index"]:
            raise RuntimeError(f"cycle {record['index']}: the record and the frame are not the same cycle")
        row_trees, row_ground = dict(record), dict(record)
        seen = False
        for name, key in study.TOF_KEYS:
            cast, _ = cast_grid(frame, name, trees)
            both, labels = cast_grid(frame, name, full)
            check.update(compare_grids(cast, recorded_grid(record, key)))
            on_ground = np.array([label == orchard_ground.GROUND for label in labels])
            differs = on_ground & ~((cast == both) | (np.isnan(cast) & np.isnan(both)))
            changed_zones += int(differs.sum())
            gained += int((differs & np.isnan(cast)).sum())
            nearer += int((differs & np.isfinite(cast)).sum())
            if on_ground.any():
                seen = True
                nearest_ground = min(nearest_ground, float(np.nanmin(both[on_ground])))
            row_trees[key], row_ground[key] = as_record_grid(cast), as_record_grid(both)
        ground_frames += seen
        with_trees.append(row_trees)
        with_ground.append(row_ground)
    stored = gate_replay.first_hold([record["python"] for record in records], records)
    cast_hold = gate_replay.first_hold(gate_replay.replay(with_trees), with_trees)
    ground_hold = gate_replay.first_hold(gate_replay.replay(with_ground), with_ground)
    return {
        "cycles": len(records),
        "trees_only_cast_against_recorded": dict(check),
        "zones_changed_by_the_ground": changed_zones,
        "zones_gained": gained,
        "zones_now_nearer": nearer,
        "frames_with_the_ground_in_view": ground_frames,
        "nearest_ground_return_mm": None if not math.isfinite(nearest_ground) else round(1000 * nearest_ground, 1),
        "first_hold_recorded": list(stored),
        "first_hold_trees_only_cast": list(cast_hold),
        "first_hold_with_ground": list(ground_hold),
        "cast_reproduces_the_recorded_hold": cast_hold == stored,
        "ground_changes_the_first_hold": ground_hold != cast_hold,
    }


def kind_of_change(row):
    """Why the first hold differs once the ground is in the mesh set."""
    if not row["ground_changes_the_first_hold"]:
        return "unchanged"
    before, after = row["first_hold_trees_only_cast"], row["first_hold_with_ground"]
    if after[1] == "tof_minimum_clearance" and (before[0] is None or after[0] < before[0]):
        return "ground_within_clearance_stops_earlier"
    if before[1] == "both_tof_missing_4_frames":
        return "blind_stop_removed_or_delayed"
    if after[0] is None:
        return "hold_removed"
    return "other"


def summarise(rows):
    check = Counter()
    for row in rows:
        check.update(row["trees_only_cast_against_recorded"])
    changed = [row for row in rows if row["ground_changes_the_first_hold"]]
    return {
        "captures": len(rows),
        "cycles": sum(row["cycles"] for row in rows),
        "trees_only_cast_against_recorded": dict(check),
        "captures_where_the_cast_reproduces_the_recorded_hold": sum(
            row["cast_reproduces_the_recorded_hold"] for row in rows
        ),
        "captures_with_the_ground_in_view": sum(row["frames_with_the_ground_in_view"] > 0 for row in rows),
        "frames_with_the_ground_in_view": sum(row["frames_with_the_ground_in_view"] for row in rows),
        "zones_changed_by_the_ground": sum(row["zones_changed_by_the_ground"] for row in rows),
        "zones_gained": sum(row["zones_gained"] for row in rows),
        "zones_now_nearer": sum(row["zones_now_nearer"] for row in rows),
        "captures_whose_first_hold_changes": len(changed),
        "kinds_of_change": dict(Counter(kind_of_change(row) for row in changed)),
        "first_hold_with_ground": dict(Counter(str(row["first_hold_with_ground"][1]) for row in rows)),
        "first_hold_trees_only": dict(Counter(str(row["first_hold_trees_only_cast"][1]) for row in rows)),
        "changed": [
            {
                "capture": row["capture"],
                "target": row["target"],
                "kind": kind_of_change(row),
                "trees_only": row["first_hold_trees_only_cast"],
                "with_ground": row["first_hold_with_ground"],
                "nearest_ground_return_mm": row["nearest_ground_return_mm"],
            }
            for row in changed
        ],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    parser.add_argument("--audit", type=Path, required=True, help="ground_audit JSON from tools/audit_ground.py")
    parser.add_argument("--roots", nargs="+", default=[str(ROOT / "artifacts")])
    parser.add_argument("--only", nargs="*", help="Limit to captures whose path contains one of these strings")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() or ROOT.resolve() in output.parents:
        parser.error(f"refusing {output}: it exists or is inside the repository")
    t0 = time.time()
    audit = json.loads(args.audit.read_text())
    classes = {row["capture"]: row["class"] for row in audit["rows"]}
    inputs = {kmp.relative(args.audit.resolve()): kmp.sha256(args.audit)}
    export = kmp.stream_export(kmp.EXPORT_DEFAULT, kmp.ISAAC_PYTHON_DEFAULT, inputs)
    names = [str(name) for name in export["names"]]
    ground = study.stream_ground()
    scenes, rows, skipped = {}, [], Counter()
    for directory in egp.discover(args.roots):
        label = str(directory.relative_to(ROOT)) if ROOT in directory.parents else str(directory)
        if args.only and not any(part in label for part in args.only):
            continue
        if label not in classes:
            skipped["not_in_the_audit"] += 1
            continue
        try:
            records, _, _ = egp.capture_cycles(directory)
        except egp.Skip as skip:
            skipped[str(skip)] += 1
            continue
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        document = json.loads((directory / "frames.json").read_text(encoding="utf-8"))
        frames = document["frames"] if isinstance(document, dict) else document
        key = cjb.scene_key(report)
        if key not in scenes:
            scenes[key] = (
                study.build_scene(export, names, report),
                study.build_scene(export, names, report, ground=ground),
            )
        row = replay_capture(records, frames, *scenes[key])
        row = {"capture": label, "target": records[0]["target"]["id"], "class": classes[label], **row}
        rows.append(row)
        print(
            f"{label}: {row['class']}, ground in {row['frames_with_the_ground_in_view']} frames, "
            f"hold {row['first_hold_trees_only_cast']} -> {row['first_hold_with_ground']}",
            flush=True,
        )
    above = [row for row in rows if row["class"] == "above"]
    beneath = [row for row in rows if row["class"] != "above"]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/replay_tof_ground.py",
        "scope": (
            "Offline: both ToF grids re-cast at recorded sensor poses with and without the orchard's ground relief, "
            "and the Python reference gate replayed on each. A replay: only each capture's first difference counts. "
            "It says what the recorded sensors would have read, not what a simulator run with the ground does."
        ),
        "definitions": {
            "range_m": list(kmp.TOF_RANGE_M),
            "agree_m": AGREE_M,
            "clearance_m": 0.06,
            "ground_triangles": int(len(ground[1])),
            "kinds_of_change": {
                "ground_within_clearance_stops_earlier": "the ground is nearer than the clearance limit and the "
                "hold comes earlier",
                "blind_stop_removed_or_delayed": "the trees-only hold was four blind cycles; the ground gives a return",
                "hold_removed": "no hold at all with the ground",
                "other": "anything else",
            },
        },
        "skipped": dict(skipped),
        "robot_above_the_ground": summarise(above),
        "robot_beneath_the_ground": summarise(beneath),
        "rows": rows,
        "inputs_sha256": inputs,
        "provenance": {
            "code_revision": head.stdout.strip(),
            "tracked_tree_clean": dirty.stdout.strip() == "",
            "host": socket.gethostname(),
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
    output.write_text(json.dumps(document, indent=1, default=float) + "\n")
    for name in ("robot_above_the_ground", "robot_beneath_the_ground"):
        print(name, json.dumps({k: v for k, v in document[name].items() if k != "changed"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
