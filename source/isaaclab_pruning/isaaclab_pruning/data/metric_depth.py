"""Metric depth helpers using the OpenCV planar-z convention."""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_depth_and_intrinsics(depth: Any, intrinsic: Any) -> tuple[np.ndarray, np.ndarray]:
    depth_array = np.asarray(depth, dtype=np.float32)
    intrinsic_array = np.asarray(intrinsic, dtype=np.float32)
    if depth_array.ndim != 2:
        raise ValueError(f"Expected depth with shape (H, W), got {depth_array.shape}.")
    if intrinsic_array.shape != (3, 3):
        raise ValueError(f"Expected K with shape (3, 3), got {intrinsic_array.shape}.")
    if intrinsic_array[0, 0] <= 0 or intrinsic_array[1, 1] <= 0:
        raise ValueError("Camera focal lengths must be positive.")
    return depth_array, intrinsic_array


def _ray_norms(shape: tuple[int, int], intrinsic: np.ndarray) -> np.ndarray:
    height, width = shape
    y_pixels, x_pixels = np.mgrid[:height, :width]
    x_normalized = (x_pixels - intrinsic[0, 2]) / intrinsic[0, 0]
    y_normalized = (y_pixels - intrinsic[1, 2]) / intrinsic[1, 1]
    return np.sqrt(x_normalized**2 + y_normalized**2 + 1.0).astype(np.float32)


def distance_to_camera_from_planar_depth(depth_z: Any, intrinsic: Any) -> np.ndarray:
    """Convert optical-axis z-depth to Euclidean distance from the camera origin."""
    depth_array, intrinsic_array = _validate_depth_and_intrinsics(depth_z, intrinsic)
    return depth_array * _ray_norms(depth_array.shape, intrinsic_array)


def planar_depth_from_distance_to_camera(distance: Any, intrinsic: Any) -> np.ndarray:
    """Convert Euclidean camera range to optical-axis z-depth."""
    distance_array, intrinsic_array = _validate_depth_and_intrinsics(distance, intrinsic)
    return distance_array / _ray_norms(distance_array.shape, intrinsic_array)


def unproject_planar_depth(
    depth_z: Any,
    intrinsic: Any,
    transform_wc: Any,
    *,
    mask: Any | None = None,
) -> np.ndarray:
    """Unproject planar z-depth into world points using OpenCV ``T_wc``."""
    depth_array, intrinsic_array = _validate_depth_and_intrinsics(depth_z, intrinsic)
    transform = np.asarray(transform_wc, dtype=np.float32)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected T_wc with shape (4, 4), got {transform.shape}.")

    valid = np.isfinite(depth_array) & (depth_array > 0)
    if mask is not None:
        mask_array = np.asarray(mask, dtype=bool)
        if mask_array.shape != depth_array.shape:
            raise ValueError(f"Mask shape {mask_array.shape} does not match depth shape {depth_array.shape}.")
        valid &= mask_array

    y_pixels, x_pixels = np.nonzero(valid)
    z = depth_array[y_pixels, x_pixels]
    x = (x_pixels.astype(np.float32) - intrinsic_array[0, 2]) * z / intrinsic_array[0, 0]
    y = (y_pixels.astype(np.float32) - intrinsic_array[1, 2]) * z / intrinsic_array[1, 1]
    points_camera = np.stack((x, y, z), axis=1)

    rotation_wc = transform[:3, :3]
    translation_wc = transform[:3, 3]
    points_world = (rotation_wc.T @ (points_camera - translation_wc).T).T
    return points_world.astype(np.float32)
