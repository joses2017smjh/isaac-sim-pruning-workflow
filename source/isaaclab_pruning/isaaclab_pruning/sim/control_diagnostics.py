"""CPU-testable contracts for nested articulation sensing and control evidence."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import torch

from isaaclab_pruning.robot.tool_frame import (
    normalize_quaternion_wxyz,
    quaternion_multiply_wxyz,
    rotate_vector_wxyz,
)


def collect_nested_rigid_prims(root: Any, is_rigid: Callable[[Any], bool]) -> list[Any]:
    """Visit every descendant, including children of an already-rigid link.

    Lab 3's generic contact activation stops descending at the first rigid body.
    That assumption does not hold for this unmerged URDF articulation hierarchy.
    The predicate is injectable so the traversal can be tested without USD.
    """
    pending = [root]
    result = []
    while pending:
        prim = pending.pop()
        if is_rigid(prim):
            result.append(prim)
        pending.extend(reversed(list(prim.GetChildren())))
    return result


def jacobian_body_index(*, body_index: int, body_count: int, jacobian_body_count: int, fixed_base: bool) -> int:
    """Validate the pinned Lab 3 body axis, independent of USD path nesting."""
    expected = body_count - int(fixed_base)
    if jacobian_body_count != expected:
        raise ValueError(f"Jacobian has {jacobian_body_count} body rows; expected {expected}.")
    index = body_index - int(fixed_base)
    if not 0 <= index < jacobian_body_count:
        raise ValueError(f"Body {body_index} has no movable Jacobian row.")
    return index


def relative_pose_wxyz(parent_pose_w: torch.Tensor, child_pose_w: torch.Tensor) -> torch.Tensor:
    """Express a world pose in a parent frame, using the core xyz+wxyz format."""
    if parent_pose_w.shape[-1] != 7 or child_pose_w.shape[-1] != 7:
        raise ValueError("Expected xyz+wxyz poses with last dimension seven.")
    quaternion = normalize_quaternion_wxyz(parent_pose_w[..., 3:7])
    inverse = torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)
    position = rotate_vector_wxyz(inverse, child_pose_w[..., :3] - parent_pose_w[..., :3])
    rotation = quaternion_multiply_wxyz(inverse, normalize_quaternion_wxyz(child_pose_w[..., 3:7]))
    return torch.cat((position, rotation), dim=-1)


def jacobian_joint_columns(joint_ids: Sequence[int] | slice, *, joint_count: int, base_dofs: int) -> list[int]:
    """Resolve Lab's optimized all-joint slice before offsetting free-base DoFs."""
    indices = list(range(joint_count))[joint_ids] if isinstance(joint_ids, slice) else list(joint_ids)
    if base_dofs < 0 or any(index < 0 or index >= joint_count for index in indices):
        raise ValueError("Invalid joint index or base DoF count")
    return [index + base_dofs for index in indices]


def contact_coverage(expected_names: Sequence[str], actual_names: Sequence[str]) -> dict:
    """Report measured coverage; shape alone cannot establish which links exist."""
    expected, actual = set(expected_names), set(actual_names)
    duplicates = sorted({name for name in actual_names if actual_names.count(name) > 1})
    return {
        "expected_body_names": sorted(expected),
        "actual_body_names": list(actual_names),
        "missing_body_names": sorted(expected - actual),
        "unexpected_body_names": sorted(actual - expected),
        "duplicate_body_names": duplicates,
        "complete": expected == actual and not duplicates,
    }


def bounded_damped_joint_delta(
    jacobian: torch.Tensor,
    pose_error: torch.Tensor,
    *,
    damping: float = 0.05,
    max_joint_delta_rad: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """SVD damped least squares with a direction-preserving joint trust region.

    Explicitly inverting ``J J.T + 1e-8 I`` in float32 loses its damping near
    the zero-pose wrist singularity. SVD evaluates the same DLS objective via
    ``s / (s*s + damping*damping)`` without forming normal equations. The
    uniform per-environment scale bounds every joint correction while retaining
    the descent direction; this is not a teleport or a joint-velocity claim.
    """
    if not math.isfinite(damping) or damping <= 0:
        raise ValueError("damping must be finite and positive")
    if not math.isfinite(max_joint_delta_rad) or max_joint_delta_rad <= 0:
        raise ValueError("max_joint_delta_rad must be finite and positive")
    if jacobian.ndim < 2 or pose_error.shape != jacobian.shape[:-1]:
        raise ValueError("Expected Jacobian (..., task_dofs, joints) and matching pose error")
    if not bool(torch.isfinite(jacobian).all()) or not bool(torch.isfinite(pose_error).all()):
        raise ValueError("IK inputs must be finite")
    u, singular_values, vh = torch.linalg.svd(jacobian, full_matrices=False)
    spectral_gain = singular_values / (singular_values.square() + damping**2)
    raw_delta = (
        vh.transpose(-2, -1)
        @ (spectral_gain * (u.transpose(-2, -1) @ pose_error.unsqueeze(-1)).squeeze(-1)).unsqueeze(-1)
    ).squeeze(-1)
    scale = (max_joint_delta_rad / raw_delta.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)).clamp_max(1.0)
    bounded_delta = raw_delta * scale
    return bounded_delta, {
        "singular_values": singular_values,
        "raw_joint_delta_rad": raw_delta,
        "bounded_joint_delta_rad": bounded_delta,
        "joint_delta_scale": scale.squeeze(-1),
    }
