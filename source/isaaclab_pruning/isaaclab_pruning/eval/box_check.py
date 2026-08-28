"""30 cm camera-rect check shared by Blender, Isaac, and the field."""

from __future__ import annotations

import numpy as np

CAMERA_RECT_DEPTH_M = 0.30


def camera_rect_extent(points_camera: np.ndarray, depth_m: float = CAMERA_RECT_DEPTH_M) -> dict[str, float]:
    """Metric size of a fronto-parallel rectangle reconstructed at ``depth_m``."""
    points = np.asarray(points_camera, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points_camera must have shape (N, 3).")
    on_plane = np.abs(points[:, 2] - depth_m) <= 0.01
    if not np.any(on_plane):
        raise ValueError(f"No reconstructed points lie on the {depth_m} m camera rect.")
    subset = points[on_plane]
    width = float(subset[:, 0].max() - subset[:, 0].min())
    height = float(subset[:, 1].max() - subset[:, 1].min())
    return {"width_m": width, "height_m": height, "depth_m": depth_m, "count": int(on_plane.sum())}


def assert_box_agrees(measured_m: float, reference_m: float = CAMERA_RECT_DEPTH_M, *, atol_m: float = 0.005) -> None:
    if abs(measured_m - reference_m) > atol_m:
        raise AssertionError(f"Camera-rect depth {measured_m:.4f} m disagrees with {reference_m:.4f} m ± {atol_m:.4f}.")
