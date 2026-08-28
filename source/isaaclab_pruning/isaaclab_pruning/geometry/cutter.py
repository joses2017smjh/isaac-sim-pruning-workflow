"""Binary STL loading and oriented-box fits for cutter volumes."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from isaaclab_pruning.task.success import OrientedBox


@dataclass(frozen=True)
class FittedBox:
    center: np.ndarray
    rotation: np.ndarray
    half_extents: np.ndarray
    vertex_count: int
    source_path: str


def load_binary_stl(path: str | Path) -> np.ndarray:
    """Return triangle vertices with shape ``(N, 3, 3)`` from a binary STL."""
    payload = Path(path).read_bytes()
    if len(payload) < 84:
        raise ValueError(f"{path} is too small to be a binary STL.")
    count = struct.unpack_from("<I", payload, 80)[0]
    expected = 84 + count * 50
    if len(payload) < expected:
        raise ValueError(f"{path} truncated: expected {expected} bytes, got {len(payload)}.")
    triangles = np.empty((count, 3, 3), dtype=np.float64)
    offset = 84
    for index in range(count):
        values = struct.unpack_from("<12f", payload, offset)
        triangles[index] = np.array(values[3:12], dtype=np.float64).reshape(3, 3)
        offset += 50
    return triangles


def fit_oriented_box(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """PCA-aligned OBB. ``rotation`` maps box-local axes into world axes."""
    points = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if points.shape[0] < 4:
        raise ValueError("Need at least four vertices to fit an oriented box.")
    center = points.mean(axis=0)
    centered = points - center
    covariance = centered.T @ centered / max(points.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    rotation = eigenvectors[:, order]
    if np.linalg.det(rotation) < 0:
        rotation[:, 2] *= -1
    local = centered @ rotation
    half_extents = np.maximum(np.abs(local).max(axis=0), 1e-6)
    return center, rotation, half_extents


def fit_oriented_box_from_stl(path: str | Path) -> FittedBox:
    triangles = load_binary_stl(path)
    center, rotation, half_extents = fit_oriented_box(triangles)
    return FittedBox(
        center=center,
        rotation=rotation,
        half_extents=half_extents,
        vertex_count=int(triangles.shape[0] * 3),
        source_path=str(path),
    )


def cutter_boxes_from_spec(
    *,
    eef_pose_w: torch.Tensor,
    mouth_half_extents: tuple[float, float, float],
    failure_half_extents: tuple[float, float, float],
    failure_offset_eef: tuple[float, float, float],
    mouth_offset_eef: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[OrientedBox, OrientedBox]:
    """Place mouth/failure AABBs in the EEF frame.

    ``eef_pose_w`` is ``(N, 7)`` position + wxyz quaternion.
    """
    if eef_pose_w.ndim != 2 or eef_pose_w.shape[-1] != 7:
        raise ValueError("eef_pose_w must have shape (N, 7).")
    rotation = _quat_wxyz_to_matrix(eef_pose_w[:, 3:7])
    position = eef_pose_w[:, 0:3]
    mouth_offset = torch.as_tensor(mouth_offset_eef, device=eef_pose_w.device, dtype=eef_pose_w.dtype)
    mouth = OrientedBox(
        center_w=position + torch.matmul(rotation, mouth_offset.unsqueeze(-1)).squeeze(-1),
        rotation_bw=rotation,
        half_extents=torch.as_tensor(mouth_half_extents, device=eef_pose_w.device, dtype=eef_pose_w.dtype).expand(
            eef_pose_w.shape[0], 3
        ),
    )
    offset = torch.as_tensor(failure_offset_eef, device=eef_pose_w.device, dtype=eef_pose_w.dtype)
    failure_center = position + torch.matmul(rotation, offset.unsqueeze(-1)).squeeze(-1)
    failure = OrientedBox(
        center_w=failure_center,
        rotation_bw=rotation,
        half_extents=torch.as_tensor(failure_half_extents, device=eef_pose_w.device, dtype=eef_pose_w.dtype).expand(
            eef_pose_w.shape[0], 3
        ),
    )
    return mouth, failure


def _quat_wxyz_to_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.stack(
        (
            torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)), dim=-1),
            torch.stack((2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)), dim=-1),
            torch.stack((2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)), dim=-1),
        ),
        dim=-2,
    )
