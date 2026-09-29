#!/usr/bin/env python3
"""Diagnose a tracker appearance loss as a moving sun shadow on the tracked bark, post hoc.

For each recorded run and window of tracker updates this replays the repository
tracker on the saved RGB-D and stops unless the replay reproduces every recorded
correlation (to 1e-6) and pixel. For each update it then reports the geometric
(depth + pose) correspondence of the previous pixel, the patch statistics, and a
sun-visibility model of the bark points under the previous patch: trees and
orchard from a geometry cache, robot links posed by forward kinematics from the
recorded joints, and the visual jaw proxy posed from the recorded tool pose. The
flip mask (points whose modelled sun visibility changes between the two frames)
is removed from the tracker's correlation and compared with count-matched random
and shifted masks. At the stop update it follows the flipped points through the
neighbouring frames.

Limits: simulator renders, one render per run; the caster is the jaw surrogate,
not the pruner CAD; nothing was re-rendered, so the mask is a modelled
association, not an intervention; the size of the grey change and pass/fail are
not modelled (tools/validate_vision_sequence.py alone grades runs). Needs open3d
and caches from tools/extract_shadow_casters.py; the output holds numbers and
hashes only, never geometry.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source/isaaclab_pruning"))
from isaaclab_pruning.perception.visual_servo import VisualServoConfig, VisualServoTracker
from isaaclab_pruning.task.simulated_cut import tool_mouth_geometry

SCHEMA_VERSION = 1
CACHE_STEM = "shadow_casters"
HALF = 6  # the tracker compares 13 x 13 patches
REPLAY_TOLERANCE = 1e-6
FK_TOLERANCE = 1e-5  # recorded joints and tool quaternions are float32; residuals reach 1.6e-6
DEPTH_GATE_M = 1e-5
ON_TARGET_M = 0.01
SERIES_DEPTH_M = 3e-3
SERIES_BEFORE = 17
SHADOW_BACK_M, SHADOW_NORMAL_M = 1e-3, 2e-4
#: Shadow-ray origin offsets (back along the view ray, along the normal) re-evaluated at the stop update: the flip
#: count depends on them, so it is reported for each and never quoted without its offsets.
SHADOW_OFFSET_SENSITIVITY = ((0.0, 1e-4), (5e-4, 1e-4), (SHADOW_BACK_M, SHADOW_NORMAL_M), (2e-3, 5e-4))
NULL_DRAWS = 2000
JAW_HALF_EXTENTS_M = np.array([0.003, 0.0125, 0.015])
JAW_OPEN_GAP_M, JAW_HEIGHT_M = 0.032, 0.070
CUBE_FACES = np.array(
    [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1]]
    + [[2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
)
CAPTURE_KEYS = (
    "/rtx/rendermode",
    "/rtx/pathtracing/spp",
    "/rtx/pathtracing/totalSpp",
    "/rtx/pathtracing/optixDenoiser/enabled",
    "/rtx-transient/dldenoiser/enabled",
)
SCOPE = (
    "Post hoc diagnosis of saved simulator renders, n = 1 render per run. The modelled caster is the visual jaw "
    "surrogate (two 6 x 25 x 30 mm cubes posed from the recorded tool pose, proxy roll and closure), not the pruner "
    "CAD. No intervention was run (nothing was re-rendered without the jaw or with the sun moved), so the flip mask "
    "is a modelled association, not a demonstrated cause. The magnitude of the grey change and task pass/fail are "
    "not modelled."
)
MASK_DEFINITION = (
    "Update i compares the tracker's 13 x 13 patch of frame i-1 at its previous pixel p with the patch of frame i at "
    "the propagated pixel (cv2.getRectSubPix on RGB2GRAY float32). Element (r, c) is the bark point hit by the camera "
    "ray of frame i-1 through pixel p + (c - 6, r - 6), pixel centres at +0.5; it is on target when the first hit is "
    "tree geometry within 1 cm of the recorded depth along the ray. It is sun-visible at frame k when its "
    "camera-facing normal n has n.s > 0 and a ray toward the sun from 1 mm back along the view ray and 0.2 mm along n "
    "is unblocked by the trees, orchard, robot links (FK from the recorded joints) and jaw proxy posed at frame k "
    "(the flip count depends on this offset; see shadow_offset_sensitivity_at_stop). "
    "The flip mask holds the on-target elements whose sun visibility differs between frames i-1 and i; the dilated "
    "mask is its 3 x 3 dilation. Masked NCC is the tracker formula (population std) over the elements outside a mask."
)


class GateError(RuntimeError):
    """A validation gate failed; nothing is written."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def qmat(wxyz):
    w, x, y, z = np.asarray(wxyz, float) / np.linalg.norm(wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def axis_rotation(axis, angle):
    k = "XYZ".index(axis)
    i, j = (k + 1) % 3, (k + 2) % 3
    rotation = np.eye(3)
    rotation[i, i] = rotation[j, j] = np.cos(angle)
    rotation[i, j], rotation[j, i] = -np.sin(angle), np.sin(angle)
    return rotation


def rigid(position, rotation):
    transform = np.eye(4)
    transform[:3, :3], transform[:3, 3] = rotation, position
    return transform


def patch(gray, centre):
    return cv2.getRectSubPix(gray, (2 * HALF + 1, 2 * HALF + 1), (float(centre[0]), float(centre[1])))


def ncc(a, b, keep=None):
    """The tracker's correlation (population std) over the kept elements; NaN when undefined."""
    a, b = (np.asarray(x, np.float64) if keep is None else np.asarray(x, np.float64)[keep] for x in (a, b))
    if a.size < 10 or min(a.std(), b.std()) < 1e-9:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / (a.std() * b.std()))


def dilate(mask):
    return cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)


def shift_mask(mask, dx, dy, wrap):
    if wrap:
        return np.roll(mask, (dy, dx), axis=(0, 1))
    out = np.zeros_like(mask)
    h, w = mask.shape
    out[max(dy, 0) : h + min(dy, 0), max(dx, 0) : w + min(dx, 0)] = mask[
        max(-dy, 0) : h + min(-dy, 0), max(-dx, 0) : w + min(-dx, 0)
    ]
    return out


def masked_ncc(previous, current, flip, draws=NULL_DRAWS):
    """NCC outside the flip mask and its dilation, with count-matched random and shifted-mask nulls."""
    rng = np.random.default_rng(0)
    out = {}
    for name, mask in (("undilated", flip), ("dilated", dilate(flip))):
        true = ncc(previous, current, ~mask)
        null = []
        for _ in range(draws):
            removed = np.zeros(mask.size, bool)
            removed[rng.choice(mask.size, int(mask.sum()), replace=False)] = True
            null.append(ncc(previous, current, ~removed.reshape(mask.shape)))
        null = np.array(null)
        out[name] = {
            "n_removed": int(mask.sum()),
            "n_kept": int((~mask).sum()),
            "ncc": true,
            "random_p95": float(np.nanpercentile(null, 95)),
            "random_fraction_ge_true": float(np.mean(null >= true)),
        }
    out["dilated"]["shift_null"] = shift_null(previous, current, dilate(flip), out["dilated"]["ncc"])
    return out


def shift_null(previous, current, mask, true):
    """Chebyshev ring 1 (8 shifts) and ring >= 2 (160 shifts) within [-6, 6]^2, wrapped and zero-filled."""
    out = {}
    for mode, wrap in (("wrap", True), ("zero_fill", False)):
        for ring, member in (("ring1", lambda r: r == 1), ("ring_ge2", lambda r: r >= 2)):
            shifts = [s for s in itertools.product(range(-HALF, HALF + 1), repeat=2) if member(max(map(abs, s)))]
            values = np.array([ncc(previous, current, ~shift_mask(mask, dx, dy, wrap)) for dx, dy in shifts])
            out[f"{ring}_{mode}"] = {
                "n_shifts": len(shifts),
                "max": float(np.nanmax(values)),
                "n_ge_true": int(np.sum(values >= true)),
            }
    return out


def build_scene(meshes):
    """One open3d geometry id per named group of (vertices, faces)."""
    import open3d as o3d

    scene, names = o3d.t.geometry.RaycastingScene(), {}
    for name, (vertices, faces) in meshes.items():
        vertices, faces = np.asarray(vertices, np.float32), np.asarray(faces, np.uint32)
        names[scene.add_triangles(o3d.core.Tensor(vertices), o3d.core.Tensor(faces))] = name
    return scene, names


def raycast(scene, origins, directions):
    import open3d as o3d

    scene, names = scene
    hit = scene.cast_rays(o3d.core.Tensor(np.hstack([origins, directions]).astype(np.float32)))
    groups = np.array([names.get(int(g), "none") for g in hit["geometry_ids"].numpy()], dtype=object)
    return hit["t_hit"].numpy().astype(float), groups, hit["primitive_normals"].numpy().astype(float)


def sun_visibility(scene, points, normals, view_directions, sun, back=SHADOW_BACK_M, along_normal=SHADOW_NORMAL_M):
    """Sun-visible (n.s > 0 and the offset shadow ray unblocked) and the first-hit caster per point."""
    origins = points - view_directions * back + normals * along_normal
    t, groups, _ = raycast(scene, origins, np.repeat(sun[None], len(points), 0))
    blocked = np.isfinite(t)
    return (normals @ sun > 0) & ~blocked, np.where(blocked, groups, "none")


def jaw_cubes(frame, report):
    """The visual jaw proxy at a frame: two cubes in R(tool quat) Rz(proxy roll) at the tool position."""
    pose = np.asarray(frame["tool_pose_wxyz"], float)
    scene = report["blender_scene"]
    rotation = qmat(pose[3:]) @ axis_rotation("Z", scene["visual_proxy_roll_rad"])
    progress = float(frame.get("visual_jaw_closure_progress") or 0.0)
    gap = (1 - progress) * JAW_OPEN_GAP_M + progress * 2 * scene["target"]["radius_m"]
    corners = np.array(list(itertools.product((-1, 1), repeat=3)), float) * JAW_HALF_EXTENTS_M
    cubes = {}
    for sign, name in ((-1, "jaw_left"), (1, "jaw_right")):
        centre = np.array([sign * (gap + 2 * JAW_HALF_EXTENTS_M[0]) / 2, 0.0, JAW_HEIGHT_M])
        cubes[name] = ((corners + centre) @ rotation.T + pose[:3], CUBE_FACES)
    return cubes


class Geometry:
    """Static casters, FK-posed robot links and the jaw proxy from one run's cache."""

    def __init__(self, cache, report):
        arrays = np.load(cache.with_suffix(".npz"))
        self.meta = json.loads(cache.with_suffix(".json").read_text())
        self.report = report
        self.sun = np.asarray(self.meta["sun_direction_to_sun_w"], float)
        self.tree_groups = [g["name"] for g in self.meta["groups"] if g.get("is_tree")]
        self.static, self.robot = {}, {}
        for group in self.meta["groups"]:
            kind = group["kind"]
            faces = arrays[f"{kind}_faces"][arrays[f"{kind}_groups"] == group["index"]]
            used, local = np.unique(faces, return_inverse=True)
            mesh = (arrays[f"{kind}_vertices"][used], local.reshape(faces.shape))
            if kind == "static":
                self.static[group["name"]] = mesh
            else:
                self.robot[group["name"]] = (group["link"], *mesh)
        self.default = {name: np.asarray(t, float) for name, t in self.meta["link_world_default"].items()}
        self.fk_residual_max = 0.0
        self._scenes, self._static_scene = {}, None

    def links(self, joint_positions):
        world = {"ur5e__base_link_inertia": self.default["ur5e__base_link_inertia"]}
        for joint, angle in zip(self.meta["joints"], joint_positions, strict=True):
            parent = rigid(joint["pos0"], qmat(joint["rot0_wxyz"]))
            child = rigid(joint["pos1"], qmat(joint["rot1_wxyz"]))
            motion = rigid(np.zeros(3), axis_rotation(joint["axis"], angle))
            world[joint["body1"]] = world[joint["body0"]] @ parent @ motion @ np.linalg.inv(child)
        wrist = "ur5e__wrist_3_link"
        world["ur5e__tool0"] = world[wrist] @ np.linalg.inv(self.default[wrist]) @ self.default["ur5e__tool0"]
        return world

    def scene(self, frame):
        """Full scene posed at a frame; aborts unless FK reproduces the recorded tool rotation."""
        key = frame["index"]
        if key not in self._scenes:
            links = self.links(frame["joint_position_rad"])
            residual = float(np.abs(links["ur5e__tool0"][:3, :3] - qmat(frame["tool_pose_wxyz"][3:])).max())
            if residual > FK_TOLERANCE:
                raise GateError(f"FK tool rotation differs from tool_pose_wxyz by {residual:.2e} at frame {key}")
            self.fk_residual_max = max(self.fk_residual_max, residual)
            meshes = dict(self.static)
            for name, (link, vertices, faces) in self.robot.items():
                meshes[name] = (vertices @ links[link][:3, :3].T + links[link][:3, 3], faces)
            meshes.update(jaw_cubes(frame, self.report))
            if len(self._scenes) >= 3:
                self._scenes.pop(next(iter(self._scenes)))
            self._scenes[key] = build_scene(meshes)
        return self._scenes[key]

    def static_scene(self):
        if self._static_scene is None:
            self._static_scene = build_scene(self.static)
        return self._static_scene


class Run:
    def __init__(self, path, cache_root):
        self.path = Path(path)
        self.report = json.loads((self.path / "report.json").read_text())
        self.frames = json.loads((self.path / "frames.json").read_text())["frames"]
        if [frame["index"] for frame in self.frames] != list(range(len(self.frames))):
            raise GateError("frame indices are not 0..N-1")
        self.K = np.asarray(self.report["camera"]["wrist_intrinsics"], float)
        self.config = dict(self.frames[0]["live_vision"]["tracker_config"])
        self.config["roi_half_size_px"] = tuple(self.config["roi_half_size_px"])
        self.scene_sha256 = sha256(self.path / "scene.usda")
        matches = [
            meta.with_suffix("")
            for meta in sorted(Path(cache_root).glob(f"*/{CACHE_STEM}.json"))
            if json.loads(meta.read_text())["scene_sha256"] == self.scene_sha256
        ]
        if not matches:
            raise GateError(f"no geometry cache under {cache_root} matches its scene.usda")
        self.cache = matches[0]
        self.geometry = Geometry(self.cache, self.report)

    def gray(self, k):
        rgb = np.array(Image.open(self.path / f"frames/wrist_{k:05d}.png").convert("RGB"))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    def depth(self, k):
        depth = np.load(self.path / f"frames/depth_{k:05d}.npy", allow_pickle=False).astype(float)
        return depth[..., 0] if depth.ndim == 3 else depth

    def rays(self, k, us, vs):
        """Camera centre and world rays scaled to unit optical z; pixel (u, v) has its centre at u + 0.5."""
        frame = self.frames[k]
        rotation = np.asarray(frame["wrist_rotation_w_ros"], float)
        pixels = np.stack([us + 0.5, vs + 0.5, np.ones_like(us)], axis=-1)
        return np.asarray(frame["wrist_position_w_m"], float), pixels @ np.linalg.inv(self.K).T @ rotation.T

    def project(self, k, points):
        frame = self.frames[k]
        local = (points - np.asarray(frame["wrist_position_w_m"], float)) @ np.asarray(frame["wrist_rotation_w_ros"])
        uvw = local @ self.K.T
        return uvw[:, :2] / uvw[:, 2:] - 0.5, local[:, 2]

    def surface(self, k, us, vs):
        """First hit of each camera ray of frame k, its camera-facing normal and whether it is on target."""
        centre, rays = self.rays(k, us, vs)
        length = np.linalg.norm(rays, axis=1)
        directions = rays / length[:, None]
        t, groups, normals = raycast(
            self.geometry.scene(self.frames[k]), np.repeat(centre[None], len(us), 0), directions
        )
        depth = self.depth(k)
        z = depth[
            np.clip(np.rint(vs).astype(int), 0, depth.shape[0] - 1),
            np.clip(np.rint(us).astype(int), 0, depth.shape[1] - 1),
        ]
        with np.errstate(invalid="ignore"):
            error = np.abs(t - z * length)
        tree = np.isin(groups, self.geometry.tree_groups)
        normals = normals * np.where(np.sum(normals * directions, axis=1) > 0, -1.0, 1.0)[:, None]
        points = centre + directions * np.where(np.isfinite(t), t, 0.0)[:, None]
        return {
            "P": points,
            "n": normals,
            "view": directions,
            "tree": tree,
            "error": error,
            "on": tree & (error < ON_TARGET_M),
        }

    def sun_state(self, k, surface):
        """Sun visibility at frame k, first-hit caster, and shadow present only with the robot and jaw."""
        args = (surface["P"], surface["n"], surface["view"], self.geometry.sun)
        visible, caster = sun_visibility(self.geometry.scene(self.frames[k]), *args)
        static_visible, _ = sun_visibility(self.geometry.static_scene(), *args)
        return visible, caster, static_visible & ~visible


class _Probe(VisualServoTracker):
    """Record the two pixels ``_appearance`` compares."""

    captured = None

    def _appearance(self, gray, next_pixel):
        self.captured = (np.asarray(self._pixel, float).tolist(), np.asarray(next_pixel, float).tolist())
        return super()._appearance(gray, next_pixel)


def replay(run, last):
    """Replay as tools/replay_visual_tracking.py does, with the recorded tracker config; abort on any difference."""
    report, camera = run.report, run.report["camera"]
    pose = report["initial_tool_pose_wxyz"][0]
    rotation = np.column_stack([tool_mouth_geometry(pose, closing_axis_tool=axis)[1] for axis in np.eye(3)])
    initial = rigid(
        np.asarray(pose[:3]) + rotation @ camera["wrist_position_in_tool_m"],
        rotation @ np.asarray(camera["wrist_rotation_in_tool_ros"]),
    )
    tracker = _Probe(VisualServoConfig(**run.config))
    rgb = np.array(Image.open(run.path / "preview_wrist.png").convert("RGB"))
    depth = np.load(run.path / "preview_depth.npy", allow_pickle=False)
    tracker.initialize(rgb, report["vision_initialization"]["pixel_xy"], depth)
    result = tracker.update(rgb, depth, camera["wrist_intrinsics"], initial)
    updates = {}
    for frame in run.frames[: last + 1]:
        if result["state"] != "tracking":
            break
        i = frame["index"]
        tracker.captured = None
        result = tracker.update(
            np.array(Image.open(run.path / f"frames/wrist_{i:05d}.png").convert("RGB")),
            np.load(run.path / f"frames/depth_{i:05d}.npy", allow_pickle=False),
            camera["wrist_intrinsics"],
            rigid(frame["wrist_position_w_m"], frame["wrist_rotation_w_ros"]),
        )
        recorded = frame["live_vision"]["measurement"]
        corr, recorded_corr = result.get("patch_correlation"), recorded.get("patch_correlation")
        same_corr = (corr is None) == (recorded_corr is None) and (
            corr is None or abs(corr - recorded_corr) <= REPLAY_TOLERANCE
        )
        if not same_corr or result.get("pixel_xy") != recorded.get("pixel_xy") or result["state"] != recorded["state"]:
            raise GateError(f"replay differs from the recording at update {i}")
        updates[i] = {
            "state": result["state"],
            "reason": result["reason"],
            "corr": corr,
            "recorded_corr": recorded_corr,
        }
        if tracker.captured:
            updates[i]["prev_pixel"], updates[i]["next_pixel"] = tracker.captured
    return updates


def analyse_update(run, i, update):
    p, q = np.asarray(update["prev_pixel"]), np.asarray(update["next_pixel"])
    gray_now = run.gray(i)
    previous, current = patch(run.gray(i - 1), p), patch(gray_now, q)
    row = {
        "update": i,
        "phase": run.frames[i]["phase"],
        "jaw_closure_progress": run.frames[i].get("visual_jaw_closure_progress"),
        "state": update["state"],
        "recorded_corr": update["recorded_corr"],
        "replay_corr": update["corr"],
        "prev_pixel": p.tolist(),
        "propagated_pixel": q.tolist(),
        "patch_prev_mean_std": [float(previous.mean()), float(previous.std())],
        "patch_cur_mean_std": [float(current.mean()), float(current.std())],
    }
    # (b) geometric truth: back-project p with depth_{i-1} and pose_{i-1}, reproject with pose_i
    x, y = np.rint(p).astype(int)
    z = run.depth(i - 1)[y, x]
    row["geometric_truth"] = None
    if np.isfinite(z) and z > 0:
        centre, ray = run.rays(i - 1, p[:1], p[1:])
        truth = run.project(i, centre + ray * z)[0][0]
        row["geometric_truth"] = {
            "pixel": truth.tolist(),
            "distance_to_propagated_px": float(np.linalg.norm(q - truth)),
            "ncc_at_true": ncc(previous, patch(gray_now, truth)),
        }
    # (d) geometry gate: camera rays of frame i-1 reproduce recorded depth on tree pixels of the ROI
    half_x, half_y = run.config["roi_half_size_px"]
    height, width = gray_now.shape
    us, vs = np.meshgrid(
        np.arange(max(0, x - half_x), min(width, x + half_x + 1)),
        np.arange(max(0, y - half_y), min(height, y + half_y + 1)),
    )
    roi = run.surface(i - 1, us.ravel().astype(float), vs.ravel().astype(float))
    errors = roi["error"][roi["tree"] & np.isfinite(roi["error"])]
    if not errors.size or not np.median(errors) < DEPTH_GATE_M:
        raise GateError(f"ray-cast depth disagrees with recorded depth at frame {i - 1}")
    row["depth_gate_median_error_m"] = float(np.median(errors))
    # (e) bark-fixed points of the previous patch, sun visibility at frames i-1 and i
    cols, rows = np.meshgrid(np.arange(-HALF, HALF + 1.0), np.arange(-HALF, HALF + 1.0))
    surface = run.surface(i - 1, p[0] + cols.ravel(), p[1] + rows.ravel())
    visible_prev, caster_prev, robot_prev = run.sun_state(i - 1, surface)
    visible_cur, caster_cur, robot_cur = run.sun_state(i, surface)
    on = surface["on"]
    to_lit, to_shadow = on & ~visible_prev & visible_cur, on & visible_prev & ~visible_cur
    flip = (to_lit | to_shadow).reshape(cols.shape)
    n_on = int(on.sum())
    row |= {
        "n_on_target": n_on,
        "lit_fraction_prev": float((on & visible_prev).sum() / n_on) if n_on else None,
        "lit_fraction_cur": float((on & visible_cur).sum() / n_on) if n_on else None,
        "robot_only_shadow_fraction_prev": float((on & robot_prev).sum() / n_on) if n_on else None,
        "robot_only_shadow_fraction_cur": float((on & robot_cur).sum() / n_on) if n_on else None,
        "n_shadow_to_lit": int(to_lit.sum()),
        "n_lit_to_shadow": int(to_shadow.sum()),
        "flip_casters": dict(sorted(Counter(np.where(to_lit, caster_prev, caster_cur)[flip.ravel()]).items())),
        "masked_ncc": masked_ncc(previous, current, flip) if flip.any() else None,
    }
    sun_facing = surface["n"] @ run.geometry.sun > 0
    return row, surface, {"flip": flip.ravel(), "stay_shadowed": on & sun_facing & ~visible_prev & ~visible_cur}


def offset_sensitivity(run, i, row, surface):
    """Flip counts, casters and masked NCC at one update for each shadow-ray origin offset."""
    previous = patch(run.gray(i - 1), np.asarray(row["prev_pixel"]))
    current = patch(run.gray(i), np.asarray(row["propagated_pixel"]))
    args = (surface["P"], surface["n"], surface["view"], run.geometry.sun)
    scenes = (run.geometry.scene(run.frames[i - 1]), run.geometry.scene(run.frames[i]))
    on, out = surface["on"], []
    for back, along in SHADOW_OFFSET_SENSITIVITY:
        before, caster_before = sun_visibility(scenes[0], *args, back=back, along_normal=along)
        after, caster_after = sun_visibility(scenes[1], *args, back=back, along_normal=along)
        to_lit, to_shadow = on & ~before & after, on & before & ~after
        flip = (to_lit | to_shadow).reshape(2 * HALF + 1, 2 * HALF + 1)
        casters = np.where(to_lit, caster_before, caster_after)[flip.ravel()]
        out.append(
            {
                "back_m": back,
                "along_normal_m": along,
                "n_shadow_to_lit": int(to_lit.sum()),
                "n_lit_to_shadow": int(to_shadow.sum()),
                "casters": dict(sorted(Counter(casters).items())),
                "masked_ncc_undilated": ncc(previous, current, ~flip),
                "n_kept_undilated": int((~flip).sum()),
                "masked_ncc_dilated": ncc(previous, current, ~dilate(flip)),
                "n_kept_dilated": int((~dilate(flip)).sum()),
            }
        )
    return out


def point_series(run, i, surface, sets):
    """Median grey (bilinear, visible within 3 mm of recorded depth) and modelled sun visibility, frames i-17..i+1."""
    members = np.flatnonzero(np.logical_or.reduce(list(sets.values())))
    points = {key: surface[key][members] for key in ("P", "n", "view")}
    sets = {name: mask[members] for name, mask in sets.items()}
    frames = []
    for k in range(max(0, i - SERIES_BEFORE), min(len(run.frames), i + 2)):
        (uv, z), depth = run.project(k, points["P"]), run.depth(k)
        height, width = depth.shape
        inside = (uv[:, 0] >= 0) & (uv[:, 0] <= width - 1) & (uv[:, 1] >= 0) & (uv[:, 1] <= height - 1)
        cols = np.clip(np.rint(uv[:, 0]).astype(int), 0, width - 1)
        rows = np.clip(np.rint(uv[:, 1]).astype(int), 0, height - 1)
        with np.errstate(invalid="ignore"):
            seen = inside & (np.abs(depth[rows, cols] - z) < SERIES_DEPTH_M)
        maps = uv.astype(np.float32).reshape(1, -1, 2)
        grey = cv2.remap(run.gray(k), maps[..., 0].copy(), maps[..., 1].copy(), cv2.INTER_LINEAR).ravel()
        visible = run.sun_state(k, points)[0]
        row = {"frame": k}
        for name, mask in sets.items():
            kept = mask & seen
            row[name] = {
                "n_visible": int(kept.sum()),
                "median_grey": float(np.median(grey[kept])) if kept.any() else None,
                "modelled_lit_fraction": float(visible[mask].mean()) if mask.any() else None,
            }
        frames.append(row)
    return {"update": i, "counts": {name: int(mask.sum()) for name, mask in sets.items()}, "frames": frames}


def analyse_run(path, window, cache_root):
    first, last = window
    run = Run(path, cache_root)
    updates = replay(run, last)
    rows, stop = [], None
    for i in range(first, last + 1):
        if i not in updates:
            break
        if "prev_pixel" not in updates[i]:
            rows.append({"update": i, "state": updates[i]["state"], "note": "no appearance comparison at this update"})
            continue
        row, surface, sets = analyse_update(run, i, updates[i])
        rows.append(row)
        if updates[i]["state"] != "tracking":
            stop = (i, row, surface, sets)
    stop_row = stop[1] if stop else {}
    series = point_series(run, stop[0], stop[2], stop[3]) if stop else None  # poses more frames: before FK max
    capture = run.report.get("capture_quality", {}).get("actual", {})
    return {
        "run": f"{run.path.parent.name}/{run.path.name}",
        "window": [first, last],
        "job_id": run.report.get("job_id"),
        "task_outcome": run.report.get("task_outcome"),
        "sha256": {
            "report.json": sha256(run.path / "report.json"),
            "frames.json": sha256(run.path / "frames.json"),
            "scene.usda": run.scene_sha256,
            f"{CACHE_STEM}.npz": sha256(run.cache.with_suffix(".npz")),
            f"{CACHE_STEM}.json": sha256(run.cache.with_suffix(".json")),
        },
        "replay": {
            "updates_replayed": len(updates),
            "last_replayed_update": max(updates),
            "max_abs_corr_difference": max(
                (abs(u["corr"] - u["recorded_corr"]) for u in updates.values() if u["corr"] is not None), default=None
            ),
            "pixel_xy_and_state_identical": True,
        },
        "lighting": {key: run.geometry.meta[key] for key in ("sun_direction_to_sun_w", "sun_intensity", "sun_color")}
        | {key: run.geometry.meta[key] for key in ("dome_intensity", "dome_color")},
        "fk_rotation_residual_max": run.geometry.fk_residual_max,
        "updates": rows,
        "stop_update": stop[0] if stop else None,
        "tool_target_distance_m_at_stop": run.frames[stop[0]]["target_distance_m"] if stop else None,
        "visual_proxy_roll_rad": run.report["blender_scene"]["visual_proxy_roll_rad"],
        "casting_jaw_at_stop": stop_row.get("flip_casters"),
        "shadow_offset_sensitivity_at_stop": offset_sensitivity(run, stop[0], stop[1], stop[2]) if stop else None,
        "capture_quality_actual": {key: capture.get(key) for key in CAPTURE_KEYS},
        "point_series": series,
    }


def finite(value):
    """JSON-safe copy: NaN becomes null, numpy scalars become Python numbers."""
    if isinstance(value, dict):
        return {key: finite(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(item) for item in value]
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value.item() if isinstance(value, np.generic) else value


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True, action="append", help="RUN_DIR:FIRST-LAST window of tracker updates")
    parser.add_argument("--cache-root", required=True, type=Path, help="Directory of extract_shadow_casters caches")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Refusing to overwrite an existing diagnosis")
    jobs = []
    for spec in args.run:
        path, _, window = spec.rpartition(":")
        jobs.append((Path(path), tuple(int(v) for v in window.split("-"))))
    tracker = Path(sys.modules[VisualServoTracker.__module__].__file__)
    runs = []
    for path, window in jobs:
        try:
            runs.append(analyse_run(path, window, args.cache_root))
        except GateError as exc:
            parser.exit(1, f"{path}: gate failed, nothing written: {exc}\n")
    result = {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "mask_definition": MASK_DEFINITION,
        "code_revision": git("rev-parse", "HEAD"),
        "tracked_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "tool_sha256": sha256(__file__),
        "tracker_sha256": sha256(tracker),
        "runs": runs,
    }
    with args.output.open("x") as stream:
        json.dump(finite(result), stream, indent=2, allow_nan=False)
        stream.write("\n")
    for run in runs:
        rows = [row for row in run["updates"] if "n_on_target" in row]
        flips = sum(row["n_shadow_to_lit"] + row["n_lit_to_shadow"] for row in rows)
        low = min((row["recorded_corr"] for row in rows), default=float("nan"))
        line = f"{run['run']}: min corr {low:.3f}, flips in window {flips}"
        stop = next((row for row in rows if row["update"] == run["stop_update"]), None)
        if stop:
            dilated = (stop["masked_ncc"] or {}).get("dilated", {})
            line += (
                f"; stop {stop['update']} corr {stop['recorded_corr']:.3f} shadow->lit {stop['n_shadow_to_lit']} "
                f"lit->shadow {stop['n_lit_to_shadow']} casters {stop['flip_casters']} "
                f"masked NCC {dilated.get('ncc')} on {dilated.get('n_kept')} kept"
            )
        print(line)


if __name__ == "__main__":
    main()
