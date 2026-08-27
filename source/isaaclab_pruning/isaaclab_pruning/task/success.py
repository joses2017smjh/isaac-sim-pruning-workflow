"""Batched geometric pruning success primitives."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


def _normalize(vectors: torch.Tensor, *, name: str) -> torch.Tensor:
    if vectors.shape[-1] != 3 or not torch.is_floating_point(vectors):
        raise ValueError(f"{name} must be a floating tensor with final dimension 3.")
    norms = torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    if torch.any(norms <= 1e-12):
        raise ValueError(f"{name} contains a zero-length vector.")
    return vectors / norms


@dataclass(frozen=True)
class OrientedBox:
    """An OBB whose rotation maps box-local axes into world axes."""

    center_w: torch.Tensor
    rotation_bw: torch.Tensor
    half_extents: torch.Tensor

    def __post_init__(self) -> None:
        if self.center_w.shape[-1] != 3:
            raise ValueError("center_w must have final dimension 3.")
        if self.rotation_bw.shape[-2:] != (3, 3):
            raise ValueError("rotation_bw must have final dimensions (3, 3).")
        if self.half_extents.shape[-1] != 3 or torch.any(self.half_extents <= 0):
            raise ValueError("half_extents must be positive with final dimension 3.")


@dataclass(frozen=True)
class CutSuccess:
    success: torch.Tensor
    mouth_hit: torch.Tensor
    failure_clear: torch.Tensor
    perpendicular: torch.Tensor
    collision_free: torch.Tensor
    perpendicularity_error_deg: torch.Tensor


def segment_intersects_obb(
    segment_start_w: torch.Tensor,
    segment_end_w: torch.Tensor,
    box: OrientedBox,
    *,
    epsilon: float = 1e-9,
) -> torch.Tensor:
    """Test finite line segments against oriented boxes with the slab method."""
    if segment_start_w.shape[-1] != 3 or segment_end_w.shape[-1] != 3:
        raise ValueError("Segment endpoints must have final dimension 3.")
    if segment_start_w.shape != segment_end_w.shape:
        raise ValueError("Segment endpoint tensors must have the same shape.")

    rotation_wb = box.rotation_bw.transpose(-1, -2)
    start_b = torch.matmul(rotation_wb, (segment_start_w - box.center_w).unsqueeze(-1)).squeeze(-1)
    end_b = torch.matmul(rotation_wb, (segment_end_w - box.center_w).unsqueeze(-1)).squeeze(-1)
    delta_b = end_b - start_b

    non_parallel = delta_b.abs() > epsilon
    safe_delta = torch.where(non_parallel, delta_b, torch.ones_like(delta_b))
    t1 = (-box.half_extents - start_b) / safe_delta
    t2 = (box.half_extents - start_b) / safe_delta
    near = torch.minimum(t1, t2)
    far = torch.maximum(t1, t2)

    negative_infinity = torch.full_like(near, -torch.inf)
    positive_infinity = torch.full_like(far, torch.inf)
    near = torch.where(non_parallel, near, negative_infinity)
    far = torch.where(non_parallel, far, positive_infinity)

    parallel_outside = (~non_parallel) & (start_b.abs() > box.half_extents)
    entry = near.amax(dim=-1)
    exit_ = far.amin(dim=-1)
    return ~parallel_outside.any(dim=-1) & (entry <= exit_) & (exit_ >= 0.0) & (entry <= 1.0)


def evaluate_cut_success(
    *,
    branch_centroid_w: torch.Tensor,
    branch_axis_w: torch.Tensor,
    branch_length_m: torch.Tensor,
    cutter_closing_axis_w: torch.Tensor,
    mouth_box: OrientedBox,
    failure_box: OrientedBox,
    arm_collision: torch.Tensor | None = None,
    other_wood_in_failure_zone: torch.Tensor | None = None,
    perpendicularity_tolerance_deg: float = 15.0,
) -> CutSuccess:
    """Evaluate mouth entry, failure-zone clearance, alignment, and collisions.

    ``other_wood_in_failure_zone`` lets the environment add a broad-phase test
    over every nearby cylinder. This function always checks the target branch.
    """
    if not 0 <= perpendicularity_tolerance_deg <= 90:
        raise ValueError("perpendicularity_tolerance_deg must be in [0, 90].")

    branch_axis = _normalize(branch_axis_w, name="branch_axis_w")
    cutter_axis = _normalize(cutter_closing_axis_w, name="cutter_closing_axis_w")
    if branch_centroid_w.shape[-1] != 3:
        raise ValueError("branch_centroid_w must have final dimension 3.")

    length = branch_length_m
    if length.shape == branch_centroid_w.shape[:-1] + (1,):
        length = length.squeeze(-1)
    if length.shape != branch_centroid_w.shape[:-1] or torch.any(length <= 0):
        raise ValueError("branch_length_m must be positive and match the batch shape.")

    half_axis = 0.5 * length.unsqueeze(-1) * branch_axis
    segment_start = branch_centroid_w - half_axis
    segment_end = branch_centroid_w + half_axis
    mouth_hit = segment_intersects_obb(segment_start, segment_end, mouth_box)
    target_in_failure_zone = segment_intersects_obb(segment_start, segment_end, failure_box)

    failure_hit = target_in_failure_zone
    if other_wood_in_failure_zone is not None:
        failure_hit = failure_hit | other_wood_in_failure_zone.to(dtype=torch.bool)
    failure_clear = ~failure_hit

    absolute_dot = torch.sum(branch_axis * cutter_axis, dim=-1).abs().clamp(0.0, 1.0)
    perpendicularity_error = torch.asin(absolute_dot)
    perpendicularity_error_deg = torch.rad2deg(perpendicularity_error)
    tolerance_rad = math.radians(perpendicularity_tolerance_deg)
    perpendicular = perpendicularity_error <= tolerance_rad

    if arm_collision is None:
        collision_free = torch.ones_like(mouth_hit)
    else:
        collision_free = ~arm_collision.to(dtype=torch.bool)
    success = mouth_hit & failure_clear & perpendicular & collision_free
    return CutSuccess(
        success=success,
        mouth_hit=mouth_hit,
        failure_clear=failure_clear,
        perpendicular=perpendicular,
        collision_free=collision_free,
        perpendicularity_error_deg=perpendicularity_error_deg,
    )
