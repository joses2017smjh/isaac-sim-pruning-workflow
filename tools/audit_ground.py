#!/usr/bin/env python3
"""Where the robot is relative to the orchard's ground relief, in every recorded capture. CPU only.

The renderer draws the orchard's ground relief and nothing collides with it. The simulated ToF sensors cast
against the trees only, and so does tools/known_map_planner.py. Every selected spur is moved to one fixed point in
the robot's workspace and the ground moves with the orchard, so for a spur that grows near the ground the surface
can end up above the robot. This audit says where that happened.

For each capture that carries the orchard's placement, at every recorded frame:
  points  the tool point, the jaw mouth and the wrist camera;
  links   every collision hull of the robot, posed by forward kinematics from the recorded joint positions and
          sampled on its surface.
A point is beneath the ground when a vertical ray from it meets the ground above it and none below it
(tools/orchard_ground.py). A link is beneath the ground in a frame when any of its samples is.

Two checks that the geometry is right, reported per capture:
  kinematics  the tool point from forward kinematics against the recorded tool pose;
  depth       at the first and the last recorded frame, a grid of pixels of the recorded depth image against a
              cast through each pixel centre, once with the trees alone and once with the ground added. The
              robot, its jaws and the pedestal are in neither mesh set, so pixels on them agree with neither.

Writes one new JSON file outside the repository; numbers and capture names only.
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

import cast_jaw_beams as cjb  # noqa: E402
import export_gate_parity as egp  # noqa: E402
import known_map_planner as kmp  # noqa: E402
import orchard_ground  # noqa: E402
import study_mouth_evidence as study  # noqa: E402

SCHEMA_VERSION = 1
#: Surface samples per link hull, in addition to the hull's own vertices.
LINK_SAMPLES = 80
#: The depth check compares every this-many-th pixel in both directions.
DEPTH_GRID_STEP_PX = 20
DEPTH_AGREE_M = 0.001
#: Forward kinematics must reproduce the recorded tool point within this, or the link check is not trusted.
KINEMATICS_AGREE_M = 0.002
POINTS = ("tool_point", "mouth", "camera")


def frame_points(frame, mouth_offset=kmp.MOUTH_IN_TOOL):
    """The tool point, the jaw mouth and the wrist camera of one frame; the camera is NaN when not recorded."""
    pose = np.asarray(frame["tool_pose_wxyz"], float)
    tool = pose[:3]
    mouth = tool + kmp.quat_wxyz_to_R(pose[3:]) @ np.asarray(mouth_offset, float)
    camera = frame.get("wrist_position_w_m")
    return np.array([tool, mouth, np.full(3, np.nan) if camera is None else np.asarray(camera, float)])


def link_points(robot, joints):
    """``(names, points, owner)``: surface samples of every link hull in the world, and which link owns each."""
    fk = robot.kin.fk(joints)
    names, points, owner = list(robot.links), [], []
    for k, link in enumerate(names):
        T = fk[link]
        local = robot.local_points(link, LINK_SAMPLES)
        points.append(local @ T[:3, :3].T + T[:3, 3])
        owner.append(np.full(len(local), k))
    return names, np.concatenate(points), np.concatenate(owner)


def depth_check(run_dir, frame, trees, full):
    """Recorded depth against casts on a pixel grid; None when the frame has no depth image or camera pose."""
    path = Path(run_dir) / "frames" / f"depth_{int(frame['index']):05d}.npy"
    if not path.is_file() or frame.get("wrist_position_w_m") is None or frame.get("wrist_rotation_w_ros") is None:
        return None
    depth = np.load(path, mmap_mode="r")
    cam_p = np.asarray(frame["wrist_position_w_m"], float)
    cam_R = np.asarray(frame["wrist_rotation_w_ros"], float)
    half = DEPTH_GRID_STEP_PX // 2
    pixels = [
        (u, v)
        for v in range(half, kmp.IMAGE_H, DEPTH_GRID_STEP_PX)
        for u in range(half, kmp.IMAGE_W, DEPTH_GRID_STEP_PX)
    ]
    rays = np.array([cam_R @ np.linalg.solve(kmp.K_CAM, [u + 0.5, v + 0.5, 1.0]) for u, v in pixels])
    rays /= np.linalg.norm(rays, axis=1)[:, None]
    along = rays @ cam_R[:, 2]
    recorded = np.array([float(np.asarray(depth[v, u]).reshape(-1)[0]) for u, v in pixels])
    origins = np.repeat(cam_p[None], len(pixels), 0)
    out = {"pixels": len(pixels), "recorded_finite": int(np.isfinite(recorded).sum())}
    for name, scene in (("trees_only", trees), ("with_ground", full)):
        t, g, _ = scene.all.cast(origins, rays)
        cast = np.where(np.isfinite(t), np.asarray(t, float) * along, np.inf)
        both_empty = ~np.isfinite(cast) & ~np.isfinite(recorded)
        with np.errstate(invalid="ignore"):
            close = np.isfinite(cast) & np.isfinite(recorded) & (np.abs(cast - recorded) <= DEPTH_AGREE_M)
        agree = both_empty | close
        out[f"agree_{name}"] = int(agree.sum())
        if name == "with_ground":
            on_ground = np.array([scene.labels[int(i)] == orchard_ground.GROUND for i in g]) & np.isfinite(t)
            out["agree_on_the_ground"] = int((agree & on_ground).sum())
    return out


def audit_capture(run_dir, frames, trees, full, robot):
    """Per-frame flags for one capture, and its two geometry checks."""
    link_names = list(robot.links)
    point_flags = np.zeros((len(frames), len(POINTS)), bool)
    link_flags = np.zeros((len(frames), len(link_names)), bool)
    kinematics_error, with_joints = 0.0, 0
    for i, frame in enumerate(frames):
        points = frame_points(frame)
        recorded = np.isfinite(points).all(axis=1)
        point_flags[i, recorded] = orchard_ground.beneath(full.ground, points[recorded])
        joints = frame.get("joint_position_rad")
        if joints is not None and len(joints) == len(robot.kin.arm_joints):
            with_joints += 1
            tool, _ = robot.kin.tool_pose(joints)
            kinematics_error = max(kinematics_error, float(np.linalg.norm(tool - points[0])))
            _, samples, owner = link_points(robot, joints)
            under = orchard_ground.beneath(full.ground, samples)
            link_flags[i] = np.bincount(owner[under], minlength=len(link_names)) > 0
    kinematics_ok = with_joints == len(frames) and kinematics_error <= KINEMATICS_AGREE_M
    links_beneath = {name: int(link_flags[:, k].sum()) for k, name in enumerate(link_names) if link_flags[:, k].any()}
    any_link = link_flags.any(axis=1)
    any_point = point_flags.any(axis=1)
    return {
        "frames": len(frames),
        "beneath_frames": {name: int(point_flags[:, k].sum()) for k, name in enumerate(POINTS)},
        "any_point_beneath_frames": int(any_point.sum()),
        "links_beneath_frames": links_beneath,
        "any_link_beneath_frames": int(any_link.sum()),
        "anything_beneath_frames": int((any_link | any_point).sum()),
        "first_frame_beneath": int(np.argmax(any_link | any_point)) if (any_link | any_point).any() else None,
        "kinematics": {
            "frames_with_joints": with_joints,
            "max_tool_point_error_mm": round(1000 * kinematics_error, 3),
            "ok": bool(kinematics_ok),
        },
        "depth_first_frame": depth_check(run_dir, frames[0], trees, full),
        "depth_last_frame": depth_check(run_dir, frames[-1], trees, full),
    }


def classify(row):
    """above: nothing beneath in any frame. tool: a tool point, mouth or camera beneath. links_only: only arm links."""
    if row["any_point_beneath_frames"]:
        return "tool"
    return "links_only" if row["any_link_beneath_frames"] else "above"


def depth_totals(rows):
    out = Counter()
    for row in rows:
        for key in ("depth_first_frame", "depth_last_frame"):
            if row[key]:
                out["images"] += 1
                out.update(row[key])
    return dict(out)


def summarise(rows):
    groups = {name: [row for row in rows if row["class"] == name] for name in ("above", "tool", "links_only")}
    return {
        "captures": len(rows),
        "frames": sum(row["frames"] for row in rows),
        "by_class": {name: len(group) for name, group in groups.items()},
        "kinematics": {
            "captures_checked": sum(row["kinematics"]["frames_with_joints"] > 0 for row in rows),
            "captures_ok": sum(row["kinematics"]["ok"] for row in rows),
            "max_tool_point_error_mm": max(
                (row["kinematics"]["max_tool_point_error_mm"] for row in rows), default=None
            ),
        },
        "depth": {name: depth_totals(group) for name, group in groups.items()},
        "beneath_by_target": dict(Counter(row["target"] for row in rows if row["class"] != "above")),
        "beneath_whole_run": sum(row["anything_beneath_frames"] == row["frames"] for row in rows),
        "links_beneath": dict(sum((Counter(row["links_beneath_frames"].keys()) for row in rows), Counter())),
        "tool": [row["capture"] for row in groups["tool"]],
        "links_only": [row["capture"] for row in groups["links_only"]],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    parser.add_argument("--roots", nargs="+", default=[str(ROOT / "artifacts")])
    parser.add_argument("--only", nargs="*", help="Limit to captures whose path contains one of these strings")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() or ROOT.resolve() in output.parents:
        parser.error(f"refusing {output}: it exists or is inside the repository")
    t0 = time.time()
    inputs = {}
    export = kmp.stream_export(kmp.EXPORT_DEFAULT, kmp.ISAAC_PYTHON_DEFAULT, inputs)
    names = [str(name) for name in export["names"]]
    ground = study.stream_ground()
    robot, mesh_hashes = kmp.load_robot(kmp.URDF_DEFAULT)
    inputs.update({kmp.relative(path): digest for path, digest in mesh_hashes.items()})
    inputs[kmp.relative(kmp.URDF_DEFAULT)] = kmp.sha256(kmp.URDF_DEFAULT)
    scenes, rows, skipped = {}, [], Counter()
    for directory in egp.discover(args.roots):
        label = str(directory.relative_to(ROOT)) if ROOT in directory.parents else str(directory)
        if args.only and not any(part in label for part in args.only):
            continue
        report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
        document = json.loads((directory / "frames.json").read_text(encoding="utf-8"))
        frames = document["frames"] if isinstance(document, dict) else document
        if not frames:
            skipped["no_frames"] += 1
            continue
        if (
            not cjb.castable(report)
            or Path(report["blender_scene"].get("export_dir", "")).name != kmp.EXPORT_DEFAULT.name
        ):
            skipped["no_orchard_placement"] += 1
            continue
        if any(frame.get("tool_pose_wxyz") is None for frame in frames):
            skipped["frame_without_tool_pose"] += 1
            continue
        key = cjb.scene_key(report)
        if key not in scenes:
            scenes[key] = (
                study.build_scene(export, names, report),
                study.build_scene(export, names, report, ground=ground),
            )
        row = audit_capture(directory, frames, *scenes[key], robot)
        row = {"capture": label, "target": report["blender_scene"]["target"]["id"], **row}
        row["class"] = classify(row)
        rows.append(row)
        print(f"{label}: {row['class']}, {row['anything_beneath_frames']}/{row['frames']} frames", flush=True)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    document = {
        "schema_version": SCHEMA_VERSION,
        "tool": "tools/audit_ground.py",
        "scope": (
            "Every recorded capture that carries the orchard's placement: whether the tool point, the jaw mouth, the "
            "wrist camera or any robot link hull is beneath the orchard's ground relief, frame by frame. Offline "
            "geometry; the relief is visual only in the simulator."
        ),
        "definitions": {
            "beneath": "a vertical ray from the point meets the ground above it and none below it",
            "link_samples": LINK_SAMPLES,
            "links": list(robot.links),
            "classes": {
                "above": "nothing beneath the ground in any frame",
                "tool": "the tool point, the jaw mouth or the wrist camera beneath the ground in some frame",
                "links_only": "only robot link hulls beneath the ground in some frame",
            },
            "depth_grid_step_px": DEPTH_GRID_STEP_PX,
            "depth_agree_m": DEPTH_AGREE_M,
            "kinematics_agree_m": KINEMATICS_AGREE_M,
            "ground_triangles": int(len(ground[1])),
        },
        "skipped": dict(skipped),
        **summarise(rows),
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
    print(
        json.dumps(
            {
                k: document[k]
                for k in ("captures", "frames", "by_class", "kinematics", "beneath_by_target", "links_beneath")
            }
        )
    )
    print(json.dumps(document["depth"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
