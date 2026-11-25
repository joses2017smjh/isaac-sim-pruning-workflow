#!/usr/bin/env python3
"""Exploratory, CPU only: which sensor could tell the safety gate, independently of the controller, that
the target spur is in the jaw mouth?

The gate takes the target's position from the controller it gates, so a tracker that reports the wrong place
with confidence passes (docs/ROS2_SIL.md, November 22 addendum). Recorded runs give the tool pose, the camera
pose and both ToF sensors' poses in every frame, and the orchard meshes are known, so candidate sensor readings
can be re-cast offline at recorded poses and at poses displaced from them. This study:

1. validates the offline caster. It re-casts both 8x8 ToF grids at the recorded sensor poses and compares them
   with the recorded grids. The criterion was fixed before the first run: of the zones valid in both, at
   least 95% agree within 1 mm, and validity agrees on at least 98% of zones.
2. defines the truth from geometry alone, never from the tracker. Nine segments cross the jaw gap parallel to
   the closing axis (three heights along the tool axis x three positions along the jaw). ``between_jaws``:
   any of them intersects the target component. ``at_mouth_centre``: the middle one does. The first
   exploratory pass used only the mouth-centre segment and found it is not on the target at closure in any
   run, so both are reported; that choice was made after seeing data and is not a registered definition.
3. reads three candidate channels at the same poses:
   A. the existing ToF sensors: of the zones whose ray crosses the open mouth box, is a first return inside it?
   B. a beam grid across the jaw gap (a sensor the tool does not have): is a beam blocked by any mesh?
   C. the wrist camera: is the surface along the ray through the mouth centre inside the mouth box?
4. builds negatives. At the frame the cut became ready, the whole tool (jaws, sensors, camera) is displaced by
   known offsets along the jaw's three axes and every channel is re-cast. Recordings hold almost no frame
   where the tracker says "in the mouth" and the spur is not.

Exploratory: nothing here is a registered result and no threshold is final. Simulator geometry: the jaws are
not in the ToF or camera mesh sets, depth is exact, and the meshes are the exported orchard, as rendered.

CPU only. Writes one new JSON file outside the repository; numbers only.
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
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "source/isaaclab_pruning"))

import known_map_planner as kmp  # noqa: E402

from isaaclab_pruning.perception import jaw_self_mask as jsm  # noqa: E402

RUNS_ROOT = ROOT / "artifacts/vision_robustness"
#: One recorded run per distinct target and approach that reached closure, plus the control that failed in closure.
RUNS = (
    "hold-sweep-20261006/run_04_source_tree0_v8175_planned_pose_jaw_hold_sweep",
    "hold-sweep-20261006/run_08_source_tree0_v12261_planned_pose_jaw_hold_sweep",
    "hold-sweep-20261006/run_09_source_tree0_v12321_planned_pose_jaw_hold_sweep",
    "hold-sweep-20261006/run_14_source_tree0_v21386_planned_pose_jaw_hold_sweep",
    "hold-sweep-20261006/run_16_source_tree0_v28610_planned_pose_jaw_hold_sweep",
    "hold-sweep-20261006/run_27_source_tree1_v36786_planned_pose_jaw_hold_sweep",
    "hold-sweep-ctl-20261006/run_00_source_tree0_v530_planned_pose_jaw_hold_sweep",
    "hold-sweep-ctl-20261006/run_01_source_tree1_v14944_planned_pose_jaw_hold_sweep",
    "hold-gen-ctl-20261004/run_00_source_tree0_v530_planned_pose_jaw_hold_gen",
    "ros2-loop-delay-20261105/run_00_source_tree1_v14944_baseline_ros2_delay0",
    "depth-loop-eve-r1-20261001/run_01_evening_tree1_v15004_baseline_depth_appearance",
)

# ---- fixed before the first run ----
TOF_AGREE_M = 0.001
TOF_AGREE_FRACTION = 0.95
VALIDITY_AGREE_FRACTION = 0.98
#: A return counts as inside the mouth box within this margin of the box.
BOX_MARGIN_M = 0.002
#: The mouth box in the jaw frame (x = closing axis, y = along the jaw, z = tool axis), jaws open.
BOX_HALF_Y_M = jsm.JAW_HALF_EXTENTS_M[1]
BOX_Z_M = (jsm.JAW_HEIGHT_M - jsm.JAW_HALF_EXTENTS_M[2], jsm.JAW_HEIGHT_M + jsm.JAW_HALF_EXTENTS_M[2])
BEAM_Y_M = (-0.008, 0.0, 0.008)
BEAM_Z_M = (jsm.JAW_HEIGHT_M - 0.010, jsm.JAW_HEIGHT_M, jsm.JAW_HEIGHT_M + 0.010)
CENTRE_BEAM = 4  # y = 0 at the jaw height, in the z-major order beams() builds
OFFSETS_M = (0.005, 0.010, 0.015, 0.020, 0.030, 0.050)
# ---- added after the first pass, when the depth image disagreed with the cast in the two runs on target 530 ----
#: The camera's line of sight counts as verified when this share of the pixels around the mouth pixel agree.
DEPTH_NEIGHBOURHOOD_PX = 4
DEPTH_AGREE_M = 0.0005
DEPTH_AGREE_FRACTION = 0.95
AXES = ("closing", "lateral", "tool")
TOF_KEYS = (("tof0", "tof_left"), ("tof1", "tof_right"))


def rot(q_wxyz):
    return kmp.quat_wxyz_to_R(np.asarray(q_wxyz, float))


def rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def ray_box(origin, direction, lo, hi):
    """(t_in, t_out) of a ray with an axis-aligned box, both in the box's frame; None when it misses."""
    t0, t1 = 0.0, math.inf
    for axis in range(3):
        if abs(direction[axis]) < 1e-12:
            if not lo[axis] <= origin[axis] <= hi[axis]:
                return None
            continue
        a = (lo[axis] - origin[axis]) / direction[axis]
        b = (hi[axis] - origin[axis]) / direction[axis]
        t0, t1 = max(t0, min(a, b)), min(t1, max(a, b))
    return (t0, t1) if t1 >= t0 else None


def mouth_box(gap=jsm.JAW_OPEN_GAP_M):
    return np.array([-gap / 2, -BOX_HALF_Y_M, BOX_Z_M[0]]), np.array([gap / 2, BOX_HALF_Y_M, BOX_Z_M[1]])


class Scene:
    """Labelled meshes with one ray index over all of them and one over the target component alone."""

    def __init__(self, meshes, index_cls=None):
        index_cls = index_cls or kmp.Open3dMeshIndex
        meshes = [m for m in meshes if len(m[2])]
        self.labels = [m[0] for m in meshes]
        k = self.labels.index("TARGET")
        self.target_triangles = int(len(meshes[k][2]))
        self.all = index_cls([(m[1], m[2]) for m in meshes])
        self.target = index_cls([(meshes[k][1], meshes[k][2])])


def build_scene(export, names, report, index_cls=None):
    """The exported orchard placed as the renderer places it; the target component keeps its own triangles."""
    b = report["blender_scene"]
    tree = b.get("target_tree_index", 1 if b.get("tree_path", "").endswith("Tree1") else 0)
    t, R = kmp.placement_from_report(report)
    target_vertices = np.asarray(b["target"]["source_component"]["source_vertex_indices"], np.int64)
    meshes = []
    for i, name in enumerate(names):
        V = np.asarray(export[f"v{i}"], float) @ R.T + t
        F = np.asarray(export[f"t{i}"], np.int64)
        label = kmp.mesh_label(name)
        if label == f"tree{tree}_SPUR":
            tv = np.zeros(len(V), bool)
            tv[target_vertices] = True
            on_target = tv[F].all(axis=1)
            meshes.append((label + "(other spurs)", V, F[~on_target]))
            meshes.append(("TARGET", V, F[on_target]))
        else:
            meshes.append((label, V, F))
    return Scene(meshes, index_cls)


class Rig:
    """Everything that moves rigidly with the tool at one instant: jaws, both ToF sensors and the wrist camera."""

    def __init__(self, tool_pose_wxyz, roll, cam_position, cam_rotation):
        pose = np.asarray(tool_pose_wxyz, float)
        self.p, self.R = pose[:3].copy(), rot(pose[3:])
        self.RJ = self.R @ rot_z(roll)
        self.base = self.p - self.R @ kmp.TOOL_IN_BASE
        self.cam_p = np.asarray(cam_position, float).copy()
        self.cam_R = np.asarray(cam_rotation, float)

    def shifted(self, offset_jaw):
        """The same rig translated by an offset given in the jaw frame."""
        out = Rig.__new__(Rig)
        delta = self.RJ @ np.asarray(offset_jaw, float)
        out.p, out.R, out.RJ = self.p + delta, self.R, self.RJ
        out.base, out.cam_p, out.cam_R = self.base + delta, self.cam_p + delta, self.cam_R
        return out

    @property
    def mouth(self):
        return self.p + self.R @ np.array([0.0, 0.0, jsm.JAW_HEIGHT_M])

    def to_jaw(self, point):
        return self.RJ.T @ (np.asarray(point, float) - self.p)

    def tof_origin(self, name):
        return self.base + self.R @ kmp.TOF_IN_BASE[name]


def beams(rig, gap, scene):
    """Channel B and the geometric truth: nine segments across the gap, parallel to the closing axis."""
    origins = np.array([rig.p + rig.RJ @ np.array([-gap / 2, y, z]) for z in BEAM_Z_M for y in BEAM_Y_M])
    dirs = np.repeat(rig.RJ[:, 0][None], len(origins), 0)
    t_all, g_all, _ = scene.all.cast(origins, dirs)
    t_target, _, _ = scene.target.cast(origins, dirs)
    blocked = np.isfinite(t_all) & (t_all < gap)
    on_target = np.isfinite(t_target) & (t_target < gap)
    return {
        "gap_mm": round(1000 * gap, 2),
        "truth_between_jaws": bool(on_target.any()),
        "truth_at_mouth_centre": bool(on_target[CENTRE_BEAM]),
        "truth_target_beams": int(on_target.sum()),
        "beams_blocked": int(blocked.sum()),
        "beam_centre_blocked": bool(blocked[CENTRE_BEAM]),
        "beam_hits": dict(Counter(scene.labels[int(g)] for g, b in zip(g_all, blocked) if b)),
    }


def tof_channel(rig, scene):
    """Channel A: zones whose ray crosses the open mouth box; is the first return inside the box?"""
    lo, hi = mouth_box()
    in_mouth, mouth_zones, nearest = Counter(), 0, (math.inf, None)
    for name, _ in TOF_KEYS:
        origin = rig.tof_origin(name)
        dirs = kmp.TOF_DIRS @ rig.R.T
        t, g, _ = scene.all.cast(np.repeat(origin[None], 64, 0), dirs)
        valid = np.isfinite(t) & (t >= kmp.TOF_RANGE_M[0]) & (t <= kmp.TOF_RANGE_M[1])
        if valid.any():
            j = int(np.flatnonzero(valid)[np.argmin(t[valid])])
            if t[j] < nearest[0]:
                nearest = (float(t[j]), scene.labels[int(g[j])])
        origin_jaw = rig.to_jaw(origin)
        for k in range(64):
            span = ray_box(origin_jaw, rig.RJ.T @ dirs[k], lo, hi)
            if span is None:
                continue
            mouth_zones += 1
            if valid[k] and span[0] - BOX_MARGIN_M <= t[k] <= span[1] + BOX_MARGIN_M:
                in_mouth[scene.labels[int(g[k])]] += 1
    return {
        "mouth_zones": mouth_zones,
        "returns_in_mouth": int(sum(in_mouth.values())),
        "returns_in_mouth_by_label": dict(in_mouth),
        "nearest_mm": None if nearest[1] is None else round(1000 * nearest[0], 1),
        "nearest_label": nearest[1],
    }


def camera_channel(rig, scene):
    """Channel C: the surface the wrist camera sees along the ray through the mouth centre."""
    lo, hi = mouth_box()
    to_mouth = rig.mouth - rig.cam_p
    direction = to_mouth / np.linalg.norm(to_mouth)
    t, g, _ = scene.all.cast(rig.cam_p[None], direction[None])
    span = ray_box(rig.to_jaw(rig.cam_p), rig.RJ.T @ direction, lo, hi)
    hit = bool(np.isfinite(t[0]))
    optical = rig.cam_R.T @ to_mouth
    return {
        "mouth_optical_z_mm": round(1000 * float(optical[2]), 1),
        "cast_optical_z_mm": round(1000 * float(t[0]) * float(direction @ rig.cam_R[:, 2]), 1) if hit else None,
        "surface_in_mouth": bool(hit and span is not None and span[0] - BOX_MARGIN_M <= t[0] <= span[1] + BOX_MARGIN_M),
        "surface_label": scene.labels[int(g[0])] if hit else None,
        "pixel_xy": [float(v) for v in (kmp.K_CAM @ optical)[:2] / optical[2]],
    }


def read_channels(rig, gap, scene, target_centre=None):
    reading = {**beams(rig, gap, scene), "tof": tof_channel(rig, scene), "camera": camera_channel(rig, scene)}
    if target_centre is not None:
        offset = rig.RJ.T @ (np.asarray(target_centre, float) - rig.mouth)
        reading["target_centre_minus_mouth_jaw_mm"] = [round(1000 * float(v), 2) for v in offset]
    return reading


def validate_tof(frames, last, scene):
    """Re-cast both grids at the recorded sensor poses and compare with the recorded grids."""
    both = agree_range = agree_depth = validity_same = zones = 0
    diffs = []
    for fr in frames[: last + 1]:
        for name, key in TOF_KEYS:
            state = fr["tof_state"]["sensors"][name]
            origin = np.asarray(state["position_w_m"][0], float)
            x, y, z, w = state["quaternion_w_ros_xyzw"][0]
            dirs = kmp.TOF_DIRS @ rot([w, x, y, z]).T
            t, _, _ = scene.all.cast(np.repeat(origin[None], 64, 0), dirs)
            recorded = np.array([[np.nan if v is None else v for v in row] for row in fr[f"{key}_m"]], float).reshape(
                -1
            )
            recorded_valid = np.asarray(fr[f"{key}_valid"], bool).reshape(-1) & np.isfinite(recorded)
            cast_valid = np.isfinite(t) & (t >= kmp.TOF_RANGE_M[0]) & (t <= kmp.TOF_RANGE_M[1])
            zones += 64
            validity_same += int((recorded_valid == cast_valid).sum())
            m = recorded_valid & cast_valid
            both += int(m.sum())
            as_range = np.abs(t[m] - recorded[m])
            agree_range += int((as_range <= TOF_AGREE_M).sum())
            agree_depth += int((np.abs(t[m] * kmp.TOF_DIRS[m, 2] - recorded[m]) <= TOF_AGREE_M).sum())
            diffs.extend(as_range.tolist())
    diffs = np.sort(np.asarray(diffs)) if diffs else np.array([np.nan])
    return {
        "frames": last + 1,
        "zones": zones,
        "validity_same": validity_same,
        "both_valid": both,
        "within_1mm_as_ray_length": agree_range,
        "within_1mm_as_axis_depth": agree_depth,
        "abs_diff_mm_p50_p95_p99_max": [round(1000 * float(np.quantile(diffs, q)), 4) for q in (0.5, 0.95, 0.99, 1.0)],
    }


def validate_depth(run_dir, frame_index, rig, scene):
    """The recorded depth image against casts through pixel centres, at and around the mouth-centre pixel.

    The offline mesh set holds the orchard only. Where the recorded image shows something else on the line of
    sight to the mouth (the tool's own drawn jaw proxy, for one), the cast and the image disagree over a whole
    neighbourhood, and the camera channel's offline reading is not valid for that run.
    """
    path = Path(run_dir) / "frames" / f"depth_{frame_index:05d}.npy"
    optical = rig.cam_R.T @ (rig.mouth - rig.cam_p)
    u0, v0 = (int(math.floor(c)) for c in (kmp.K_CAM @ optical)[:2] / optical[2])
    reach = DEPTH_NEIGHBOURHOOD_PX
    inside = reach <= u0 < kmp.IMAGE_W - reach and reach <= v0 < kmp.IMAGE_H - reach
    if not path.is_file() or not inside:
        return None
    depth = np.load(path, mmap_mode="r")
    centre, agree, pixels = None, 0, 0
    for v in range(v0 - reach, v0 + reach + 1):
        for u in range(u0 - reach, u0 + reach + 1):
            ray = rig.cam_R @ np.linalg.solve(kmp.K_CAM, [u + 0.5, v + 0.5, 1.0])
            ray /= np.linalg.norm(ray)
            t, g, _ = scene.all.cast(rig.cam_p[None], ray[None])
            cast = float(t[0]) * float(ray @ rig.cam_R[:, 2]) if np.isfinite(t[0]) else None
            recorded = float(np.asarray(depth[v, u]).reshape(-1)[0])
            recorded = recorded if math.isfinite(recorded) else None
            same = (cast is None and recorded is None) or (
                cast is not None and recorded is not None and abs(cast - recorded) <= DEPTH_AGREE_M
            )
            agree, pixels = agree + int(same), pixels + 1
            if (u, v) == (u0, v0):
                centre = (recorded, cast, scene.labels[int(g[0])] if cast is not None else None)
    return {
        "pixel_uv": [u0, v0],
        "recorded_depth_mm": None if centre[0] is None else round(1000 * centre[0], 2),
        "cast_depth_mm": None if centre[1] is None else round(1000 * centre[1], 2),
        "cast_label": centre[2],
        "neighbourhood_pixels": pixels,
        "neighbourhood_agree": agree,
        "line_of_sight_verified": bool(agree >= DEPTH_AGREE_FRACTION * pixels),
    }


def study_run(name, export, names, runs_root=RUNS_ROOT, index_cls=None):
    run_dir = Path(runs_root) / name
    report, frames = kmp.load_run(run_dir)
    scene = build_scene(export, names, report, index_cls)
    b = report["blender_scene"]
    roll, radius = float(b["visual_proxy_roll_rad"]), float(b["target"]["radius_m"])
    target_centre = report["target_position_m"]

    def cut(fr):
        return (fr.get("live_vision") or {}).get("cut") or {}

    ready = next((f["index"] for f in frames if (cut(f).get("certificate") or {}).get("ready_to_close")), None)
    closing = next((f["index"] for f in frames if cut(f).get("phase") == "closing"), None)
    detach = next((f["index"] for f in frames if cut(f).get("detach_event")), None)
    stop = next((f["index"] for f in frames if cut(f).get("phase") == "stopped"), None)
    last = min(v for v in (detach, stop, len(frames) - 1) if v is not None)

    def rig_of(fr):
        return Rig(fr["tool_pose_wxyz"], roll, fr["wrist_position_w_m"], fr["wrist_rotation_w_ros"])

    def gap_of(fr):
        return jsm.jaw_gap_m(float(fr.get("visual_jaw_closure_progress") or 0.0), radius)

    out = {
        "run": name,
        "target": b["target"]["id"],
        "radius_mm": round(1000 * radius, 2),
        "target_triangles": scene.target_triangles,
        "frames": {"ready": ready, "closing": closing, "detach": detach, "stop": stop, "last_compared": last},
        "tof_validation": validate_tof(frames, last, scene),
        "timeline": [],
    }
    for fr in frames[: last + 1]:
        if ready is None or fr["index"] < ready - 2:
            continue
        certificate = cut(fr).get("certificate") or {}
        mouth_distance = certificate.get("mouth_distance_m")
        out["timeline"].append(
            {
                "frame": fr["index"],
                "cut_phase": cut(fr).get("phase"),
                "tracker_state": ((fr.get("live_vision") or {}).get("measurement") or {}).get("state"),
                "tracker_mouth_mm": None if mouth_distance is None else round(1000 * mouth_distance, 2),
                "tracker_ready": bool(certificate.get("ready_to_close")),
                **read_channels(rig_of(fr), gap_of(fr), scene, target_centre),
            }
        )
    if ready is not None:
        fr = frames[ready]
        rig, gap = rig_of(fr), gap_of(fr)
        depth = validate_depth(run_dir, ready, rig, scene)
        out["depth_validation_at_ready"] = depth
        out["camera_line_of_sight_verified"] = bool(depth and depth["line_of_sight_verified"])
        out["displacements"] = [{"axis": "none", "offset_mm": 0.0, **read_channels(rig, gap, scene, target_centre)}]
        for axis, label in enumerate(AXES):
            for sign in (-1, 1):
                for offset in OFFSETS_M:
                    vector = np.zeros(3)
                    vector[axis] = sign * offset
                    reading = read_channels(rig.shifted(vector), gap, scene, target_centre)
                    out["displacements"].append({"axis": label, "offset_mm": round(1000 * sign * offset, 1), **reading})
        for row in out["displacements"]:
            row["camera_line_of_sight_verified"] = out["camera_line_of_sight_verified"]
    return out


CHANNELS = {
    "A_tof_return_in_mouth_box": lambda r: r["tof"]["returns_in_mouth"] > 0,
    "B_any_beam_blocked": lambda r: r["beams_blocked"] > 0,
    "B_centre_beam_blocked": lambda r: r["beam_centre_blocked"],
    "C_camera_surface_in_mouth_box": lambda r: r["camera"]["surface_in_mouth"],
}


def confusion(rows, truth_key):
    out = {}
    for name, says in CHANNELS.items():
        table = Counter((bool(row[truth_key]), bool(says(row))) for row in rows)
        out[name] = {
            "in_and_says_in": table[(True, True)],
            "in_and_says_out": table[(True, False)],
            "out_and_says_in": table[(False, True)],
            "out_and_says_out": table[(False, False)],
        }
    return out


def response(rows):
    """Per displacement: in how many runs the truth holds and each channel says 'in'."""
    table = []
    for axis in ("none", *AXES):
        for offset in sorted({row["offset_mm"] for row in rows if row["axis"] == axis}):
            group = [row for row in rows if row["axis"] == axis and row["offset_mm"] == offset]
            table.append(
                {
                    "axis": axis,
                    "offset_mm": offset,
                    "runs": len(group),
                    "truth_between_jaws": sum(row["truth_between_jaws"] for row in group),
                    "truth_at_mouth_centre": sum(row["truth_at_mouth_centre"] for row in group),
                    **{name: sum(bool(says(row)) for row in group) for name, says in CHANNELS.items()},
                    "beam_blocked_by_other_wood": sum(any(k != "TARGET" for k in row["beam_hits"]) for row in group),
                }
            )
    return table


def summarise(runs):
    rows = [row for run in runs for row in run.get("displacements", [])]
    zones = sum(run["tof_validation"]["zones"] for run in runs)
    both = sum(run["tof_validation"]["both_valid"] for run in runs)
    validity = sum(run["tof_validation"]["validity_same"] for run in runs) / max(zones, 1)
    within = sum(run["tof_validation"]["within_1mm_as_ray_length"] for run in runs) / max(both, 1)
    return {
        "caster_validation": {
            "criterion": {
                "within_m": TOF_AGREE_M,
                "fraction_of_both_valid": TOF_AGREE_FRACTION,
                "validity_agreement": VALIDITY_AGREE_FRACTION,
            },
            "zones": zones,
            "both_valid": both,
            "validity_agreement": validity,
            "within_1mm_as_ray_length": within,
            "within_1mm_as_axis_depth": sum(r["tof_validation"]["within_1mm_as_axis_depth"] for r in runs)
            / max(both, 1),
            "met": bool(within >= TOF_AGREE_FRACTION and validity >= VALIDITY_AGREE_FRACTION),
        },
        "poses": len(rows),
        "against_truth_between_jaws": confusion(rows, "truth_between_jaws"),
        "against_truth_at_mouth_centre": confusion(rows, "truth_at_mouth_centre"),
        "camera_line_of_sight": {
            "verified_runs": sum(bool(run.get("camera_line_of_sight_verified")) for run in runs),
            "runs": len(runs),
            "unverified": [run["run"] for run in runs if not run.get("camera_line_of_sight_verified")],
            "verified_runs_at_recorded_pose_camera_says_in": sum(
                bool(row["camera"]["surface_in_mouth"])
                for row in rows
                if row["axis"] == "none" and row.get("camera_line_of_sight_verified", True)
            ),
            "C_camera_against_truth_between_jaws_verified_runs_only": confusion(
                [row for row in rows if row.get("camera_line_of_sight_verified", True)], "truth_between_jaws"
            )["C_camera_surface_in_mouth_box"],
        },
        "response_to_displacement": response(rows),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if output.exists() or ROOT.resolve() in output.parents:
        parser.error(f"refusing {output}: it exists or is inside the repository")
    t0 = time.time()
    inputs = {}
    export = kmp.stream_export(kmp.EXPORT_DEFAULT, kmp.ISAAC_PYTHON_DEFAULT, inputs)
    names = [str(n) for n in export["names"]]
    runs = []
    for name in RUNS:
        for part in ("report.json", "frames.json"):
            inputs[f"runs/{name}/{part}"] = kmp.sha256(RUNS_ROOT / name / part)
        runs.append(study_run(name, export, names))
        print(f"{name}: ready {runs[-1]['frames']['ready']}, closing {runs[-1]['frames']['closing']}", flush=True)
    head = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    dirty = subprocess.run(
        ["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=True,
    )
    document = {
        "schema_version": 1,
        "tool": "tools/study_mouth_evidence.py",
        "scope": (
            "Exploratory offline study on recorded simulator runs: candidate sensor readings re-cast on the CPU at "
            "recorded and displaced tool poses. Not a registered result; no threshold here is final."
        ),
        "definitions": {
            "truth_between_jaws": "any of nine segments across the jaw gap intersects the target component",
            "truth_at_mouth_centre": "the segment through the mouth centre intersects the target component",
            "truth_note": "the first pass used only the mouth-centre segment; both are reported after seeing it is "
            "not on the target at closure in any run",
            "mouth_box_jaw_frame_m": {"half_gap": jsm.JAW_OPEN_GAP_M / 2, "half_y": BOX_HALF_Y_M, "z": list(BOX_Z_M)},
            "box_margin_m": BOX_MARGIN_M,
            "beam_grid_m": {"y": list(BEAM_Y_M), "z": list(BEAM_Z_M)},
            "offsets_m": list(OFFSETS_M),
            "tof_in_base_m": {k: v.tolist() for k, v in kmp.TOF_IN_BASE.items()},
            "tool_in_base_m": kmp.TOOL_IN_BASE.tolist(),
            "mouth_in_tool_m": kmp.MOUTH_IN_TOOL.tolist(),
        },
        **summarise(runs),
        "runs": runs,
        "inputs_sha256": inputs,
        "provenance": {
            "code_revision": head.stdout.strip(),
            "tracked_tree_clean": dirty.stdout.strip() == "",
            "host": socket.gethostname(),
            "elapsed_s": round(time.time() - t0, 1),
        },
    }
    output.write_text(json.dumps(document, indent=1, default=float) + "\n")
    validation = document["caster_validation"]
    print(
        f"caster: validity {validation['validity_agreement']:.5f}, "
        f"within 1 mm {validation['within_1mm_as_ray_length']:.5f}, criterion met: {validation['met']}"
    )
    for row in document["response_to_displacement"]:
        print(
            f"  {row['axis']:8} {row['offset_mm']:6.1f} mm: "
            f"between jaws {row['truth_between_jaws']:2d}/{row['runs']} | "
            f"ToF {row['A_tof_return_in_mouth_box']:2d}  any beam {row['B_any_beam_blocked']:2d}  "
            f"centre beam {row['B_centre_beam_blocked']:2d}  camera {row['C_camera_surface_in_mouth_box']:2d}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
