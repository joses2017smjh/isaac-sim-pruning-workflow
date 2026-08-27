"""Bridge Isaac Lab camera poses to the existing spur-depth annotation schema.

Isaac Lab exposes a camera-to-world pose with OpenCV camera axes. The historical
Blender annotations store a camera origin plus Blender XYZ Euler angles. This
module writes both the legacy fields and an explicit world-to-camera matrix so
old and new consumers reconstruct the same points.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

_BLENDER_TO_OPENCV = np.diag([1.0, -1.0, -1.0]).astype(np.float64)


def _as_numpy(value: Any, *, dtype: np.dtype = np.float64) -> np.ndarray:
    """Convert numpy, torch, or array-like values without retaining gradients."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value, dtype=dtype)


def quaternion_wxyz_to_matrix(quaternion_wxyz: Any) -> np.ndarray:
    """Return the 3x3 rotation matrix for a normalized ``(w, x, y, z)`` quaternion."""
    quaternion = _as_numpy(quaternion_wxyz).reshape(4)
    norm = float(np.linalg.norm(quaternion))
    if not np.isfinite(norm) or norm < 1e-12:
        raise ValueError("Camera quaternion must be finite and non-zero.")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_xyz_euler(rotation: np.ndarray) -> np.ndarray:
    """Invert the ``Rz @ Ry @ Rx`` convention used by Blender XYZ Euler poses."""
    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"Expected a 3x3 rotation matrix, got {rotation.shape}.")

    cos_y = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if cos_y > 1e-8:
        x = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        y = math.atan2(-float(rotation[2, 0]), cos_y)
        z = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        # At gimbal lock, choose z=0 and preserve the represented rotation.
        x = math.atan2(-float(rotation[1, 2]), float(rotation[1, 1]))
        y = math.atan2(-float(rotation[2, 0]), cos_y)
        z = 0.0
    return np.array([x, y, z], dtype=np.float64)


def world_to_camera_from_opencv_pose(position_w: Any, quaternion_wxyz: Any) -> np.ndarray:
    """Build OpenCV ``T_wc`` from an Isaac camera-to-world pose."""
    position = _as_numpy(position_w).reshape(3)
    rotation_cw = quaternion_wxyz_to_matrix(quaternion_wxyz)
    rotation_wc = rotation_cw.T

    transform_wc = np.eye(4, dtype=np.float32)
    transform_wc[:3, :3] = rotation_wc
    transform_wc[:3, 3] = -rotation_wc @ position
    return transform_wc


def _blender_euler_from_opencv_quaternion(quaternion_wxyz: Any) -> np.ndarray:
    """Return legacy Blender camera Euler angles for an Isaac OpenCV pose."""
    rotation_cw_opencv = quaternion_wxyz_to_matrix(quaternion_wxyz)
    rotation_cw_blender = rotation_cw_opencv @ _BLENDER_TO_OPENCV
    return _matrix_to_xyz_euler(rotation_cw_blender)


def blender_pose_to_world_to_camera(location: Any, rotation_euler: Any) -> np.ndarray:
    """Match ``spur_depth.data.trunk_stereo_triplet._euler_xyz_to_T`` exactly."""
    rx, ry, rz = _as_numpy(rotation_euler).reshape(3)
    cx, sx = math.cos(float(rx)), math.sin(float(rx))
    cy, sy = math.cos(float(ry)), math.sin(float(ry))
    cz, sz = math.cos(float(rz)), math.sin(float(rz))

    rotation_cw_blender = np.array(
        [
            [cy * cz, sx * sy * cz - cx * sz, cx * sy * cz + sx * sz],
            [cy * sz, sx * sy * sz + cx * cz, cx * sy * sz - sx * cz],
            [-sy, sx * cy, cx * cy],
        ],
        dtype=np.float64,
    )
    rotation_cw_opencv = rotation_cw_blender @ _BLENDER_TO_OPENCV
    position = _as_numpy(location).reshape(3)

    transform_wc = np.eye(4, dtype=np.float32)
    transform_wc[:3, :3] = rotation_cw_opencv.T
    transform_wc[:3, 3] = -rotation_cw_opencv.T @ position
    return transform_wc


def _camera_image_shape(camera: Any) -> tuple[int, int]:
    """Return ``(height, width)`` across Isaac Lab camera API variants."""
    image_shape = getattr(camera, "image_shape", None)
    if image_shape is not None:
        if callable(image_shape):
            image_shape = image_shape()
        if len(image_shape) >= 2:
            return int(image_shape[-2]), int(image_shape[-1])

    cfg = getattr(camera, "cfg", None)
    if cfg is not None and hasattr(cfg, "height") and hasattr(cfg, "width"):
        return int(cfg.height), int(cfg.width)

    output = getattr(getattr(camera, "data", None), "output", {})
    for key in ("rgb", "distance_to_image_plane", "depth"):
        if key not in output:
            continue
        shape = tuple(output[key].shape)
        if len(shape) >= 3:
            return int(shape[-3]), int(shape[-2])
    raise AttributeError("Could not infer camera image shape from image_shape, cfg, or output tensors.")


def isaac_camera_to_ann(
    camera: Any,
    env_id: int = 0,
    *,
    tree_id: str | None = None,
    shot: int | None = None,
    variant: str = "c",
    rgb_path: str = "",
    depth_path: str = "",
    mask_path: str = "",
) -> dict[str, Any]:
    """Convert Isaac Lab ``CameraData`` into a spur-depth-compatible annotation.

    The legacy ``rotation_euler`` field is deliberately retained. Current
    spur-depth consumers do not read ``_T_wc`` yet, so omitting it would not
    make the annotation backward compatible.
    """
    data = camera.data
    intrinsic = _as_numpy(data.intrinsic_matrices[env_id], dtype=np.float32)
    position = _as_numpy(data.pos_w[env_id], dtype=np.float32).reshape(3)
    quaternion = _as_numpy(data.quat_w_opencv[env_id]).reshape(4)
    height, width = _camera_image_shape(camera)

    transform_wc = world_to_camera_from_opencv_pose(position, quaternion)
    rotation_euler = _blender_euler_from_opencv_quaternion(quaternion)
    annotation: dict[str, Any] = {
        "camera": {
            "location": position.tolist(),
            "rotation_euler": rotation_euler.astype(np.float32).tolist(),
            "intrinsics": {
                "width": width,
                "height": height,
                "K": intrinsic.tolist(),
            },
        },
        "_T_wc": transform_wc.tolist(),
    }

    if tree_id is not None:
        annotation.update(
            {
                "tree_id": tree_id,
                "shot": 0 if shot is None else int(shot),
                "variant": variant,
                "rgb_path": rgb_path,
                "depth_path": depth_path,
                "masks": {"tree_only": mask_path, "union": None},
            }
        )
    return annotation


def pose_from_ann(annotation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Load ``(K, T_wc)`` while accepting explicit or legacy pose fields."""
    camera = annotation["camera"]
    intrinsic = np.asarray(camera["intrinsics"]["K"], dtype=np.float32)
    if "_T_wc" in annotation:
        transform_wc = np.asarray(annotation["_T_wc"], dtype=np.float32)
    else:
        transform_wc = blender_pose_to_world_to_camera(camera["location"], camera["rotation_euler"])
    if intrinsic.shape != (3, 3):
        raise ValueError(f"Expected K with shape (3, 3), got {intrinsic.shape}.")
    if transform_wc.shape != (4, 4):
        raise ValueError(f"Expected T_wc with shape (4, 4), got {transform_wc.shape}.")
    return intrinsic, transform_wc
