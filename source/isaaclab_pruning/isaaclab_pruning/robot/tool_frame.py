"""Isaac-independent control-tool frame transforms and Jacobian shifts.

All poses are ``[..., 7]`` tensors containing position followed by a ``wxyz``
quaternion.  A geometric Jacobian is ordered ``[linear; angular]``, matching
Isaac Lab's differential-IK controller.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

_IDENTITY_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)


def _check_last_dimension(tensor: torch.Tensor, size: int, name: str) -> None:
    if tensor.ndim == 0 or tensor.shape[-1] != size:
        raise ValueError(f"{name} must have shape (..., {size}); got {tuple(tensor.shape)}.")


def _as_pose_tensor(pose: torch.Tensor, name: str) -> torch.Tensor:
    if not isinstance(pose, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor.")
    _check_last_dimension(pose, 7, name)
    if not pose.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype.")
    return pose


def _as_vector_like(
    value: torch.Tensor | Sequence[float],
    reference: torch.Tensor,
    size: int,
    name: str,
) -> torch.Tensor:
    vector = torch.as_tensor(value, dtype=reference.dtype, device=reference.device)
    _check_last_dimension(vector, size, name)
    return vector


def _join_pose(position: torch.Tensor, quaternion: torch.Tensor) -> torch.Tensor:
    leading_shape = torch.broadcast_shapes(position.shape[:-1], quaternion.shape[:-1])
    position = torch.broadcast_to(position, leading_shape + (3,))
    quaternion = torch.broadcast_to(quaternion, leading_shape + (4,))
    return torch.cat((position, quaternion), dim=-1)


def normalize_quaternion_wxyz(quaternion: torch.Tensor, *, eps: float = 1.0e-12) -> torch.Tensor:
    """Return unit ``wxyz`` quaternions, rejecting undefined zero quaternions."""
    _check_last_dimension(quaternion, 4, "quaternion")
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    if torch.any(norm <= eps):
        raise ValueError("Quaternion norm must be greater than zero.")
    return quaternion / norm


def quaternion_multiply_wxyz(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Compose broadcastable ``wxyz`` quaternions as ``lhs * rhs``."""
    _check_last_dimension(lhs, 4, "lhs")
    _check_last_dimension(rhs, 4, "rhs")
    lhs, rhs = torch.broadcast_tensors(lhs, rhs)
    lw, lv = lhs[..., :1], lhs[..., 1:]
    rw, rv = rhs[..., :1], rhs[..., 1:]
    scalar = lw * rw - torch.sum(lv * rv, dim=-1, keepdim=True)
    vector = lw * rv + rw * lv + torch.linalg.cross(lv, rv, dim=-1)
    return torch.cat((scalar, vector), dim=-1)


def rotate_vector_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate broadcastable vectors by ``wxyz`` quaternions."""
    _check_last_dimension(quaternion, 4, "quaternion")
    _check_last_dimension(vector, 3, "vector")
    quaternion = normalize_quaternion_wxyz(quaternion)
    q_vector, vector = torch.broadcast_tensors(quaternion[..., 1:], vector)
    q_scalar = torch.broadcast_to(quaternion[..., :1], q_vector.shape[:-1] + (1,))
    twice_cross = 2.0 * torch.linalg.cross(q_vector, vector, dim=-1)
    return vector + q_scalar * twice_cross + torch.linalg.cross(q_vector, twice_cross, dim=-1)


def compose_physics_body_to_control_tool_pose(
    physics_body_pose: torch.Tensor,
    tool_translation_in_body: torch.Tensor | Sequence[float],
    tool_quaternion_in_body_wxyz: torch.Tensor | Sequence[float] = _IDENTITY_QUATERNION_WXYZ,
) -> torch.Tensor:
    """Compose a physics-body pose with its fixed control-tool transform.

    Args:
        physics_body_pose: Body pose in an arbitrary parent frame, shaped
            ``(..., 7)`` as position plus ``wxyz`` quaternion.
        tool_translation_in_body: Vector from the body origin to the tool
            origin, expressed in the body frame.
        tool_quaternion_in_body_wxyz: Tool orientation relative to the body.

    Returns:
        The control-tool pose in the same parent frame as ``physics_body_pose``.
    """
    body_pose = _as_pose_tensor(physics_body_pose, "physics_body_pose")
    translation = _as_vector_like(tool_translation_in_body, body_pose, 3, "tool_translation_in_body")
    tool_quaternion = _as_vector_like(
        tool_quaternion_in_body_wxyz,
        body_pose,
        4,
        "tool_quaternion_in_body_wxyz",
    )
    body_quaternion = normalize_quaternion_wxyz(body_pose[..., 3:7])
    tool_quaternion = normalize_quaternion_wxyz(tool_quaternion)
    position = body_pose[..., :3] + rotate_vector_wxyz(body_quaternion, translation)
    quaternion = normalize_quaternion_wxyz(quaternion_multiply_wxyz(body_quaternion, tool_quaternion))
    return _join_pose(position, quaternion)


def control_tool_pose_to_physics_body_pose(
    control_tool_pose: torch.Tensor,
    tool_translation_in_body: torch.Tensor | Sequence[float],
    tool_quaternion_in_body_wxyz: torch.Tensor | Sequence[float] = _IDENTITY_QUATERNION_WXYZ,
) -> torch.Tensor:
    """Convert an absolute control-tool pose command to the physics-body pose.

    This is the inverse of :func:`compose_physics_body_to_control_tool_pose` for
    a fixed body-to-tool transform.
    """
    tool_pose = _as_pose_tensor(control_tool_pose, "control_tool_pose")
    translation = _as_vector_like(tool_translation_in_body, tool_pose, 3, "tool_translation_in_body")
    tool_quaternion_in_body = _as_vector_like(
        tool_quaternion_in_body_wxyz,
        tool_pose,
        4,
        "tool_quaternion_in_body_wxyz",
    )
    tool_quaternion = normalize_quaternion_wxyz(tool_pose[..., 3:7])
    tool_quaternion_in_body = normalize_quaternion_wxyz(tool_quaternion_in_body)
    inverse_tool_quaternion = torch.cat(
        (tool_quaternion_in_body[..., :1], -tool_quaternion_in_body[..., 1:]),
        dim=-1,
    )
    body_quaternion = normalize_quaternion_wxyz(quaternion_multiply_wxyz(tool_quaternion, inverse_tool_quaternion))
    body_position = tool_pose[..., :3] - rotate_vector_wxyz(body_quaternion, translation)
    return _join_pose(body_position, body_quaternion)


def point_offset_in_jacobian_frame(
    physics_body_quaternion: torch.Tensor,
    tool_translation_in_body: torch.Tensor | Sequence[float],
) -> torch.Tensor:
    """Express a body-frame tool offset in the Jacobian's coordinate frame.

    ``physics_body_quaternion`` must rotate body-frame vectors into the frame in
    which the Jacobian rows are expressed (world for PhysX's raw Jacobian).
    """
    _check_last_dimension(physics_body_quaternion, 4, "physics_body_quaternion")
    translation = _as_vector_like(
        tool_translation_in_body,
        physics_body_quaternion,
        3,
        "tool_translation_in_body",
    )
    return rotate_vector_wxyz(physics_body_quaternion, translation)


def skew_symmetric_matrix(vector: torch.Tensor) -> torch.Tensor:
    """Return matrices ``[vector]x`` such that ``[vector]x @ x = vector × x``."""
    _check_last_dimension(vector, 3, "vector")
    x, y, z = vector.unbind(dim=-1)
    zero = torch.zeros_like(x)
    return torch.stack(
        (
            zero,
            -z,
            y,
            z,
            zero,
            -x,
            -y,
            x,
            zero,
        ),
        dim=-1,
    ).reshape(vector.shape[:-1] + (3, 3))


def shift_spatial_jacobian_to_point(
    body_jacobian: torch.Tensor,
    point_offset_in_jacobian_coordinates: torch.Tensor,
) -> torch.Tensor:
    r"""Shift a body-origin geometric Jacobian to an offset point.

    The Jacobian must use Isaac Lab's row order ``[linear; angular]`` and the
    offset ``r`` must point from the body origin to the target point, expressed
    in the same coordinates as the Jacobian rows.  The shift is

    ``Jv_point = Jv_body + Jw × r = Jv_body - [r]x @ Jw``.

    The angular block is reference-point invariant.  The input tensor is not
    modified.
    """
    if not isinstance(body_jacobian, torch.Tensor):
        raise TypeError("body_jacobian must be a torch.Tensor.")
    if body_jacobian.ndim < 2 or body_jacobian.shape[-2] != 6:
        raise ValueError(f"body_jacobian must have shape (..., 6, num_dofs); got {tuple(body_jacobian.shape)}.")
    if not body_jacobian.is_floating_point():
        raise TypeError("body_jacobian must have a floating-point dtype.")
    offset = _as_vector_like(
        point_offset_in_jacobian_coordinates,
        body_jacobian,
        3,
        "point_offset_in_jacobian_coordinates",
    )
    leading_shape = torch.broadcast_shapes(body_jacobian.shape[:-2], offset.shape[:-1])
    body_jacobian = torch.broadcast_to(body_jacobian, leading_shape + body_jacobian.shape[-2:])
    offset = torch.broadcast_to(offset, leading_shape + (3,))
    shifted_linear = body_jacobian[..., :3, :] - torch.matmul(
        skew_symmetric_matrix(offset),
        body_jacobian[..., 3:6, :],
    )
    return torch.cat((shifted_linear, body_jacobian[..., 3:6, :]), dim=-2)
