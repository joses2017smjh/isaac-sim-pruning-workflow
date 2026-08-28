"""Wrist-camera extrinsic candidates. No pose is selected yet."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WristCameraCandidate:
    name: str
    position_eef_m: tuple[float, float, float]
    notes: str


# camera_offset is empty on the real URDF. These are sweep seeds, not a choice.
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
