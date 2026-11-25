#!/usr/bin/env python3
"""Screen the spur pool for layouts that put the robot beneath the orchard's ground. CPU only.

Every selected spur is moved to one fixed point in the robot's workspace, and the ground relief moves with the
orchard. For a spur that grows near the ground, the surface then lies above the robot. The simulator does not react
to that (the relief is drawn and nothing collides with it) and tools/known_map_planner.py casts against the trees
only. This screen answers, for every spur of the jaw-fit pool, a question the planner never asked: with this spur at
the target point, is any part of the robot at its home pose beneath the ground?

Why a separate tool and not a planner option: the planner refuses any acceptance evidence that was made by a
different version of its own file, so editing it would invalidate the committed acceptance. This screen runs beside
it, changes nothing unless it is run, and uses the planner's own pool, placement, kinematics and home pose.

What it tests at the reference home pose: every link hull (surface samples), the tool point, the jaw mouth and the
planner's model of the home wrist camera. It does not test a planned path. In the recorded captures the two agree:
every capture with anything beneath the ground has it from the first frame (tools/audit_ground.py), and --audit
checks this screen against that audit target by target.

Writes one new JSON file outside the repository; numbers and component ids only.
"""

from __future__ import annotations

import argparse
import json
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

import audit_ground  # noqa: E402
import known_map_planner as kmp  # noqa: E402
import orchard_ground  # noqa: E402

SCHEMA_VERSION = 1
BASE_LINK = "ur5e__base_link_inertia"


def home_points(robot, q_home, target, axis_w):
    """``(names, points, owner)`` at the home pose: link samples, then the tool point, the mouth and the camera."""
    names, points, owner = audit_ground.link_points(robot, q_home)
    p0, R0 = robot.kin.tool_pose(q_home)
    mount, _ = kmp.camera_for_home(p0, R0, target, axis_w)
    extra = np.array([p0, p0 + R0 @ kmp.MOUTH_IN_TOOL, p0 + R0 @ mount])
    labels = names + ["tool_point", "mouth", "camera"]
    owner = np.r_[owner, len(names) + np.arange(3)]
    return labels, np.r_[points, extra], owner


def screen_spur(ground_index, robot, q_home, entry, target=kmp.TARGET_W):
    """Which parts of the robot at home are beneath the ground when this spur sits at the target point."""
    translation, rotation = kmp.placement_from_component(entry["center"])
    axis_w = rotation @ np.asarray(entry["axis"], float)
    axis_w /= np.linalg.norm(axis_w)
    labels, points, owner = home_points(robot, q_home, target, axis_w)
    # The placement is a yaw and a translation, so vertical stays vertical: test in the export's own frame.
    local = (points - translation) @ rotation
    under = orchard_ground.beneath(ground_index, local)
    beneath = [labels[k] for k in sorted(set(owner[under].tolist()))]
    above, below = orchard_ground.vertical_gaps(ground_index, local[owner == labels.index("tool_point")])
    gap = float(above[0]) if np.isfinite(above[0]) else (-float(below[0]) if np.isfinite(below[0]) else None)
    return {
        "tree": int(entry["tree"]),
        "first_vertex": int(entry["first_vertex"]),
        "beneath": beneath,
        "any_beneath": bool(beneath),
        "base_link_beneath": BASE_LINK in beneath,
        "tool_point_beneath": "tool_point" in beneath,
        # Positive: the ground is this far above the home tool point. Negative: this far below it.
        "ground_above_tool_point_mm": None if gap is None else round(1000 * gap, 1),
    }


def summarise(rows):
    out = {}
    for tree in sorted({row["tree"] for row in rows}):
        group = [row for row in rows if row["tree"] == tree]
        out[f"tree{tree}"] = {
            "spurs": len(group),
            "any_part_beneath": sum(row["any_beneath"] for row in group),
            "base_link_beneath": sum(row["base_link_beneath"] for row in group),
            "tool_point_beneath": sum(row["tool_point_beneath"] for row in group),
            "first_vertices_beneath": sorted(row["first_vertex"] for row in group if row["any_beneath"]),
        }
    out["all"] = {
        "spurs": len(rows),
        "any_part_beneath": sum(row["any_beneath"] for row in rows),
        "base_link_beneath": sum(row["base_link_beneath"] for row in rows),
        "tool_point_beneath": sum(row["tool_point_beneath"] for row in rows),
        "parts": dict(Counter(part for row in rows for part in row["beneath"])),
    }
    return out


def against_audit(rows, audit):
    """Compare the home screen with what tools/audit_ground.py found in the recorded captures, per target."""
    screened = {(row["tree"], row["first_vertex"]): row for row in rows}
    recorded = {}
    for row in audit["rows"]:
        # tree<k>_SPUR_component_<first vertex>
        tree, vertex = int(row["target"].split("_")[0][4:]), int(row["target"].rsplit("_", 1)[1])
        recorded.setdefault((tree, vertex), set()).add(row["class"] != "above")
    agree, differ, missing = [], [], []
    for key, classes in sorted(recorded.items()):
        if key not in screened:
            missing.append(f"tree{key[0]}_{key[1]}")
        elif classes == {screened[key]["any_beneath"]}:
            agree.append(key)
        else:
            differ.append(
                {
                    "target": f"tree{key[0]}_{key[1]}",
                    "recorded_beneath": sorted(classes),
                    "screen": screened[key]["beneath"],
                }
            )
    return {
        "recorded_targets": len(recorded),
        "agree": len(agree),
        "agree_beneath": sum(screened[key]["any_beneath"] for key in agree),
        "differ": differ,
        "not_in_pool": missing,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    parser.add_argument("--runs-root", type=Path, default=kmp.RUNS_DEFAULT)
    parser.add_argument("--export-dir", type=Path, default=kmp.EXPORT_DEFAULT)
    parser.add_argument("--urdf", type=Path, default=kmp.URDF_DEFAULT)
    parser.add_argument("--isaac-python", default=kmp.ISAAC_PYTHON_DEFAULT)
    parser.add_argument("--audit", type=Path, help="ground_audit JSON from tools/audit_ground.py, to check against")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() or ROOT.resolve() in output.parents:
        parser.error(f"refusing {output}: it exists or is inside the repository")
    t0 = time.time()
    ctx = kmp.Context(args, lambda message: print(message, flush=True))
    ground = orchard_ground.stream_ground(args.export_dir, args.isaac_python)
    ground_index = kmp.Open3dMeshIndex([ground])
    # The planner leaves two spurs out of its search; the screen wants every spur of the pool.
    excluded, kmp.EXCLUDED_FROM_SEARCH = kmp.EXCLUDED_FROM_SEARCH, ()
    try:
        pool, pool_report = kmp.measure_pool(ctx)
    finally:
        kmp.EXCLUDED_FROM_SEARCH = excluded
    rows = [screen_spur(ground_index, ctx.robot, ctx.q_home_ref, entry) for entry in pool]
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/screen_ground.py",
        "scope": (
            "For every spur of the jaw-fit pool: with that spur moved to the target point, which parts of the robot at "
            "the reference home pose are beneath the orchard's ground relief. Offline geometry; home pose only."
        ),
        "definitions": {
            "beneath": "a vertical ray from the point meets the ground above it and none below it",
            "parts": "each link hull (surface samples), the tool point, the jaw mouth, the planner's home camera",
            "home": "the settled home of the reference recording, as the planner uses it",
            "target_point_w_m": [float(v) for v in kmp.TARGET_W],
            "link_samples": audit_ground.LINK_SAMPLES,
            "ground_triangles": int(len(ground[1])),
        },
        "pool": pool_report,
        "summary": summarise(rows),
        "against_recorded_captures": against_audit(rows, json.loads(args.audit.read_text())) if args.audit else None,
        "rows": rows,
        "inputs_sha256": {
            **ctx.inputs,
            **({kmp.relative(args.audit.resolve()): kmp.sha256(args.audit)} if args.audit else {}),
        },
        "provenance": {
            "code_revision": head.stdout.strip(),
            "tracked_tree_clean": dirty.stdout.strip() == "",
            "host": socket.gethostname(),
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
    output.write_text(json.dumps(document, indent=1, default=float) + "\n")
    print(json.dumps(document["summary"]["all"]))
    print(json.dumps({k: v for k, v in document["summary"].items() if k != "all"}))
    print(json.dumps(document["against_recorded_captures"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
