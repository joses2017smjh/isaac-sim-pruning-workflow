"""Explicit quaternion boundary between pinned Lab 3 and pruning core tensors.

Lab 3 articulation poses, math, and differential IK use ``xyzw``. The pruning
robot specification, tool geometry, observations, and public pose actions use
``wxyz``. These Isaac-independent conversions preserve all leading dimensions.
"""

from __future__ import annotations

import torch


def quaternion_xyzw_to_wxyz(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert a Lab 3 quaternion to the pruning core convention."""
    if quaternion.ndim == 0 or quaternion.shape[-1] != 4:
        raise ValueError("Expected quaternion shape (..., 4).")
    return quaternion[..., [3, 0, 1, 2]]


def quaternion_wxyz_to_xyzw(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert a pruning core quaternion to the Lab 3 convention."""
    if quaternion.ndim == 0 or quaternion.shape[-1] != 4:
        raise ValueError("Expected quaternion shape (..., 4).")
    return quaternion[..., [1, 2, 3, 0]]


def pose_xyzw_to_wxyz(pose: torch.Tensor) -> torch.Tensor:
    """Convert an xyz + xyzw pose to xyz + wxyz without changing position."""
    if pose.ndim == 0 or pose.shape[-1] != 7:
        raise ValueError("Expected pose shape (..., 7).")
    return torch.cat((pose[..., :3], quaternion_xyzw_to_wxyz(pose[..., 3:7])), dim=-1)


def pose_wxyz_to_xyzw(pose: torch.Tensor) -> torch.Tensor:
    """Convert an xyz + wxyz pose to xyz + xyzw without changing position."""
    if pose.ndim == 0 or pose.shape[-1] != 7:
        raise ValueError("Expected pose shape (..., 7).")
    return torch.cat((pose[..., :3], quaternion_wxyz_to_xyzw(pose[..., 3:7])), dim=-1)
