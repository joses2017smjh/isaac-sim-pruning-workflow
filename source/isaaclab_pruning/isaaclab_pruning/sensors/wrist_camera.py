"""Wrist-camera extrinsic candidates. Selection is geometric, not rendered."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from isaaclab_pruning.geometry.cut_point import CutPoint
from isaaclab_pruning.geometry.cylinders import Cylinder

RANGE_BAND_M = (0.3, 0.5)
APPROACH_STANDOFF_M = 0.40
_EPS_T = 0.01


@dataclass(frozen=True)
class WristCameraCandidate:
    name: str
    position_eef_m: tuple[float, float, float]
    notes: str


# The upstream YAML camera_offset is empty, but the Xacro ignores that argument
# and hard-codes a camera0 frame from mock_pruner__base. It does not identify a
# camera model or calibrated optical-frame rotation. These EEF-frame candidates
# are simulation sweep seeds, not replacements for that unresolved hardware
# contract. The selected candidate lives in mock_pruner_vl53l8cx.yaml.
CANDIDATES: tuple[WristCameraCandidate, ...] = (
    WristCameraCandidate(
        "close_lateral",
        (0.0, -0.06, 0.10),
        "Sees the mouth at 0.3-0.5 m; jaws may occlude the lower third.",
    ),
    WristCameraCandidate(
        "raised_clearance",
        (0.0, -0.08, 0.14),
        "More jaw clearance, less angular resolution on a spur.",
    ),
    WristCameraCandidate(
        "aft_dovetail",
        (0.0, 0.0, 0.04),
        "Mounted back toward the dovetail; likely occluded by the cutter body.",
    ),
)


def pinhole_in_view(
    point_camera: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
    *,
    z_near: float = 0.105,
    z_far: float = 10.0,
) -> bool:
    """Whether a camera-frame point projects inside the image with valid depth."""
    x, y, z = np.asarray(point_camera, dtype=np.float64).reshape(3)
    if not (z_near < z < z_far):
        return False
    pixel = intrinsic @ np.array([x / z, y / z, 1.0], dtype=np.float64)
    u, v = pixel[0], pixel[1]
    return 0.0 <= u < width and 0.0 <= v < height


def score_candidate(
    candidate: WristCameraCandidate,
    cut_point_eef: np.ndarray,
    intrinsic: np.ndarray,
    width: int,
    height: int,
) -> dict[str, float | bool | str]:
    """Score visibility of a cut point. Does not pick a winner."""
    offset = np.asarray(candidate.position_eef_m, dtype=np.float64)
    point_camera = np.asarray(cut_point_eef, dtype=np.float64) - offset
    visible = pinhole_in_view(point_camera, intrinsic, width, height)
    return {
        "name": candidate.name,
        "visible": visible,
        "range_m": float(np.linalg.norm(point_camera)),
        "selected": False,
    }


def ray_finite_cylinder_t(
    origin: np.ndarray,
    direction: np.ndarray,
    cylinder: Cylinder,
    *,
    t_max: float = 10.0,
) -> float:
    """Smallest ``t >= 0`` where the ray hits the finite cylinder, else ``inf``."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return float("inf")
    direction = direction / norm
    center = np.asarray(cylinder.centroid, dtype=np.float64)
    axis = np.asarray(cylinder.orientation, dtype=np.float64)
    radius = float(cylinder.radius)
    half = 0.5 * float(cylinder.length)
    offset = origin - center
    d_perp = direction - np.dot(direction, axis) * axis
    o_perp = offset - np.dot(offset, axis) * axis
    a = float(np.dot(d_perp, d_perp))
    hits: list[float] = []
    if a > 1e-12:
        b = 2.0 * float(np.dot(o_perp, d_perp))
        c = float(np.dot(o_perp, o_perp) - radius * radius)
        disc = b * b - 4.0 * a * c
        if disc >= 0.0:
            root = np.sqrt(disc)
            for t in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
                if 0.0 <= t <= t_max:
                    axial = abs(float(np.dot(offset + t * direction, axis)))
                    if axial <= half + 1e-9:
                        hits.append(float(t))
    # Caps: plane at ±half along axis, disk of radius r.
    denom = float(np.dot(direction, axis))
    if abs(denom) > 1e-12:
        for cap in (-half, half):
            t = (cap - float(np.dot(offset, axis))) / denom
            if 0.0 <= t <= t_max:
                radial = offset + t * direction - cap * axis
                radial = radial - np.dot(radial, axis) * axis
                if float(np.linalg.norm(radial)) <= radius + 1e-9:
                    hits.append(float(t))
    return min(hits) if hits else float("inf")


def ray_obb_t(
    origin: np.ndarray,
    direction: np.ndarray,
    center: np.ndarray,
    rotation_bw: np.ndarray,
    half_extents: np.ndarray,
    *,
    t_max: float = 10.0,
) -> float:
    """Slab-method ray vs OBB. ``rotation_bw`` maps box-local axes into world."""
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    direction = np.asarray(direction, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return float("inf")
    direction = direction / norm
    rotation_wb = np.asarray(rotation_bw, dtype=np.float64).T
    o_local = rotation_wb @ (origin - np.asarray(center, dtype=np.float64))
    d_local = rotation_wb @ direction
    extents = np.asarray(half_extents, dtype=np.float64).reshape(3)
    t_enter = -np.inf
    t_exit = np.inf
    for axis in range(3):
        if abs(d_local[axis]) < 1e-12:
            if abs(o_local[axis]) > extents[axis]:
                return float("inf")
            continue
        inv = 1.0 / d_local[axis]
        t1 = (-extents[axis] - o_local[axis]) * inv
        t2 = (extents[axis] - o_local[axis]) * inv
        t_near, t_far = (t1, t2) if t1 < t2 else (t2, t1)
        t_enter = max(t_enter, t_near)
        t_exit = min(t_exit, t_far)
        if t_enter > t_exit:
            return float("inf")
    if t_exit < 0.0 or t_enter > t_max:
        return float("inf")
    hit = t_enter if t_enter >= 0.0 else t_exit
    return float(hit) if 0.0 <= hit <= t_max else float("inf")


def approach_pose_looking_at(
    cut_w: np.ndarray,
    *,
    standoff_m: float = APPROACH_STANDOFF_M,
) -> tuple[np.ndarray, np.ndarray]:
    """EEF pose with +Z aimed at the cut, horizontal approach from the origin."""
    cut_w = np.asarray(cut_w, dtype=np.float64).reshape(3)
    delta = cut_w.copy()
    delta[2] = 0.0
    norm = float(np.linalg.norm(delta))
    approach = np.array([0.0, 1.0, 0.0], dtype=np.float64) if norm < 1e-6 else delta / norm
    eef_pos = cut_w - standoff_m * approach
    z_axis = approach
    x_axis = np.cross(np.array([0.0, 0.0, 1.0]), z_axis)
    if float(np.linalg.norm(x_axis)) < 1e-6:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))
    return eef_pos, rotation


def _mouth_world(
    eef_pos: np.ndarray,
    rotation: np.ndarray,
    mouth_offset_eef: tuple[float, float, float],
    mouth_half_extents: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    offset = np.asarray(mouth_offset_eef, dtype=np.float64)
    center = eef_pos + rotation @ offset
    return center, rotation, np.asarray(mouth_half_extents, dtype=np.float64)


def score_candidate_on_tree(
    candidate: WristCameraCandidate,
    cylinders: list[Cylinder],
    cuts: list[CutPoint],
    *,
    mouth_offset_eef: tuple[float, float, float],
    mouth_half_extents: tuple[float, float, float],
    intrinsic: np.ndarray,
    width: int,
    height: int,
    range_band_m: tuple[float, float] = RANGE_BAND_M,
) -> dict[str, Any]:
    """Ray-cast visibility / jaw occlusion. No renderer."""
    visible = 0
    in_band = 0
    jaw_occluded = 0
    wood_occluded = 0
    for cut in cuts:
        eef_pos, rotation = approach_pose_looking_at(cut.position_w)
        camera_w = eef_pos + rotation @ np.asarray(candidate.position_eef_m, dtype=np.float64)
        to_cut = np.asarray(cut.position_w, dtype=np.float64) - camera_w
        t_cut = float(np.linalg.norm(to_cut))
        if t_cut < 1e-9:
            continue
        direction = to_cut / t_cut
        point_camera = rotation.T @ to_cut
        in_view = pinhole_in_view(point_camera, intrinsic, width, height, z_near=0.05, z_far=10.0)
        band = range_band_m[0] <= t_cut <= range_band_m[1]
        if band:
            in_band += 1
        mouth_center, mouth_rot, mouth_he = _mouth_world(eef_pos, rotation, mouth_offset_eef, mouth_half_extents)
        t_jaw = ray_obb_t(camera_w, direction, mouth_center, mouth_rot, mouth_he, t_max=t_cut)
        jaw_hit = t_jaw < t_cut - _EPS_T
        t_wood = float("inf")
        for cylinder in cylinders:
            if cylinder.record_id == cut.record_id:
                continue
            t_wood = min(t_wood, ray_finite_cylinder_t(camera_w, direction, cylinder, t_max=t_cut))
        wood_hit = t_wood < t_cut - _EPS_T
        if jaw_hit:
            jaw_occluded += 1
        if wood_hit:
            wood_occluded += 1
        if in_view and band and not jaw_hit and not wood_hit:
            visible += 1
    n_cuts = len(cuts)
    return {
        "name": candidate.name,
        "offset_eef_m": list(candidate.position_eef_m),
        "n_cuts": n_cuts,
        "n_in_band": in_band,
        "n_visible": visible,
        "n_jaw_occluded": jaw_occluded,
        "n_wood_occluded": wood_occluded,
        "visible_fraction": (visible / n_cuts) if n_cuts else 0.0,
        "selected": False,
        "notes": candidate.notes,
    }


def select_wrist_camera(scores: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the candidate with the most unoccluded in-band cuts. None if all are zero."""
    if not scores:
        return None
    ranked = sorted(
        scores,
        key=lambda item: (-int(item["n_visible"]), int(item["n_jaw_occluded"]), item["name"]),
    )
    winner = dict(ranked[0])
    if int(winner["n_visible"]) <= 0:
        return None
    winner["selected"] = True
    return winner
