"""Scripted ToF final-approach baseline.

Reimplements the lab's closed-loop division of labour from the published
controller description: estimate a branch from sparse ToF returns, then servo
pan / pitch / roll / approach. This is original code, not a copy of the
unlicensed ROS nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ToFServoGains:
    pan: float = 1.0
    pitch: float = 1.0
    roll: float = 1.0
    approach: float = 0.4
    max_delta_m: float = 0.03
    max_delta_rad: float = 0.15


def _pinhole_rays(width: int, height: int, dfov_deg: float, device, dtype) -> torch.Tensor:
    diagonal = torch.tensor(dfov_deg, device=device, dtype=dtype).deg2rad()
    focal = (0.5 * (width**2 + height**2) ** 0.5) / torch.tan(diagonal / 2.0)
    us = torch.arange(width, device=device, dtype=dtype) + 0.5
    vs = torch.arange(height, device=device, dtype=dtype) + 0.5
    grid_v, grid_u = torch.meshgrid(vs, us, indexing="ij")
    x = (grid_u - 0.5 * width) / focal
    y = (grid_v - 0.5 * height) / focal
    z = torch.ones_like(x)
    rays = torch.stack((x, y, z), dim=-1)
    return rays / torch.linalg.vector_norm(rays, dim=-1, keepdim=True)


def deproject_tof(
    ranges_m: torch.Tensor,
    valid: torch.Tensor,
    offset_m: tuple[float, float, float],
    *,
    dfov_deg: float = 65.0,
) -> torch.Tensor:
    """Deproject an ``(..., H, W)`` ToF image into EEF-frame points."""
    *batch, height, width = ranges_m.shape
    rays = _pinhole_rays(width, height, dfov_deg, ranges_m.device, ranges_m.dtype)
    points = ranges_m.unsqueeze(-1) * rays
    offset = torch.as_tensor(offset_m, device=ranges_m.device, dtype=ranges_m.dtype)
    points = points + offset
    points = points.reshape(*batch, height * width, 3)
    mask = valid.reshape(*batch, height * width)
    points = torch.where(mask.unsqueeze(-1), points, torch.full_like(points, float("nan")))
    return points


def fit_branch_axis(points_eef: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fit a 3D line to stacked ToF points. Returns centroid, axis, inlier count."""
    finite = torch.isfinite(points_eef).all(dim=-1)
    centroid = torch.zeros(points_eef.shape[0], 3, device=points_eef.device, dtype=points_eef.dtype)
    axis = torch.zeros_like(centroid)
    axis[:, 1] = 1.0
    counts = finite.sum(dim=-1)
    for env_id in range(points_eef.shape[0]):
        pts = points_eef[env_id, finite[env_id]]
        if pts.shape[0] < 3:
            continue
        mean = pts.mean(dim=0)
        centered = pts - mean
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        centroid[env_id] = mean
        axis[env_id] = vh[0]
    return centroid, axis, counts


def scripted_tof_action(
    ranges_tof0: torch.Tensor,
    ranges_tof1: torch.Tensor,
    valid_tof0: torch.Tensor,
    valid_tof1: torch.Tensor,
    *,
    tof0_offset: tuple[float, float, float] = (0.04685226669, 0.0, 0.14444246761),
    tof1_offset: tuple[float, float, float] = (-0.04685226669, 0.0, 0.14444246761),
    gains: ToFServoGains | None = None,
) -> torch.Tensor:
    """Return a 7-D EEF pose delta ``(dx, dy, dz, qx, qy, qz, qw relative)`` as xyz + axis-angle-ish xyzw.

    The action is ``(N, 7)``: position delta in the EEF frame and a wxyz quaternion
    delta approximating pan/pitch/roll corrections.
    """
    cfg = gains or ToFServoGains()
    points0 = deproject_tof(ranges_tof0, valid_tof0, tof0_offset)
    points1 = deproject_tof(ranges_tof1, valid_tof1, tof1_offset)
    points = torch.cat((points0, points1), dim=1)
    centroid, axis, counts = fit_branch_axis(points)

    # Pan/pitch: drive the estimated centroid onto the cutter forward axis (z).
    lateral = centroid[:, 0].clamp(-cfg.max_delta_m, cfg.max_delta_m)
    vertical = centroid[:, 1].clamp(-cfg.max_delta_m, cfg.max_delta_m)
    # Move the mouth forward along EEF +z until a short standoff remains.
    standoff = 0.08
    approach = (centroid[:, 2] - standoff).clamp(-cfg.max_delta_m, cfg.max_delta_m) * cfg.approach

    # Roll: cutter closing axis is EEF x; want it perpendicular to the branch.
    # A branch along y is already perpendicular to x.
    branch_xy = axis[:, 0:2]
    branch_xy = branch_xy / torch.clamp(torch.linalg.vector_norm(branch_xy, dim=-1, keepdim=True), min=1e-6)
    roll = torch.atan2(branch_xy[:, 0], branch_xy[:, 1]).clamp(-cfg.max_delta_rad, cfg.max_delta_rad) * cfg.roll

    no_lock = counts < 3
    zeros = torch.zeros_like(lateral)
    dx = torch.where(no_lock, zeros, -lateral * cfg.pan)
    dy = torch.where(no_lock, zeros, -vertical * cfg.pitch)
    dz = torch.where(no_lock, zeros, approach)
    half = 0.5 * torch.where(no_lock, zeros, roll)
    quat = torch.stack((torch.cos(half), torch.sin(half), zeros, zeros), dim=-1)
    return torch.cat((torch.stack((dx, dy, dz), dim=-1), quat), dim=-1)


def _quat_rotate(quaternion_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    qvec = quaternion_wxyz[..., 1:]
    uv = torch.cross(qvec, vector, dim=-1)
    uuv = torch.cross(qvec, uv, dim=-1)
    return vector + 2.0 * (quaternion_wxyz[..., :1] * uv + uuv)


def _quat_multiply(left_wxyz: torch.Tensor, right_wxyz: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = left_wxyz.unbind(-1)
    w2, x2, y2, z2 = right_wxyz.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def scripted_absolute_pose(eef_pose_w: torch.Tensor, delta_eef: torch.Tensor) -> torch.Tensor:
    """Compose an EEF-frame scripted delta onto an absolute world pose (xyz + wxyz)."""
    position = eef_pose_w[:, :3] + _quat_rotate(eef_pose_w[:, 3:7], delta_eef[:, :3])
    quaternion = _quat_multiply(eef_pose_w[:, 3:7], delta_eef[:, 3:7])
    quaternion = quaternion / torch.clamp(torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True), min=1e-8)
    return torch.cat((position, quaternion), dim=-1)
