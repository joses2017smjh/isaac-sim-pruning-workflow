from __future__ import annotations

import math

import pytest
import torch

from isaaclab_pruning.baselines.tof_servo import scripted_absolute_pose
from isaaclab_pruning.robot.tool_frame import compose_physics_body_to_control_tool_pose
from isaaclab_pruning.sim.pose_conventions import (
    pose_wxyz_to_xyzw,
    pose_xyzw_to_wxyz,
    quaternion_wxyz_to_xyzw,
    quaternion_xyzw_to_wxyz,
)


def test_lab_identity_pose_preserves_the_physical_tool_offset() -> None:
    # Lab 3 identity is xyzw=(0, 0, 0, 1); interpreting it as wxyz would
    # rotate this non-axis-aligned offset by 180 degrees around Z.
    native_body = torch.tensor([[1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 1.0]], dtype=torch.float64)
    original = native_body.clone()
    core_tool = compose_physics_body_to_control_tool_pose(pose_xyzw_to_wxyz(native_body), (0.02, -0.03, 0.16))
    torch.testing.assert_close(core_tool, torch.tensor([[1.02, 1.97, 3.16, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float64))
    torch.testing.assert_close(native_body, original)


def test_rotated_body_tool_and_scripted_action_cross_lab_boundary() -> None:
    half_turn = math.sqrt(0.5)
    # A 90-degree rotation around Y takes the optical +Z into parent +X.
    native_body = torch.tensor([[1.0, 2.0, 3.0, 0.0, half_turn, 0.0, half_turn]], dtype=torch.float64)
    tool_wxyz = compose_physics_body_to_control_tool_pose(pose_xyzw_to_wxyz(native_body), (0.0, 0.0, 0.16))
    delta_wxyz = torch.tensor([[0.0, 0.0, 0.005, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float64)
    command_wxyz = scripted_absolute_pose(tool_wxyz, delta_wxyz)
    native_command = pose_wxyz_to_xyzw(command_wxyz)
    torch.testing.assert_close(native_command[:, :3], torch.tensor([[1.165, 2.0, 3.0]], dtype=torch.float64))
    torch.testing.assert_close(native_command[:, 3:], native_body[:, 3:])


def test_pose_conversion_preserves_batches_dtype_and_gradient() -> None:
    native_pose = torch.arange(42, dtype=torch.float64).reshape(2, 3, 7).requires_grad_()
    core_pose = pose_xyzw_to_wxyz(native_pose)
    torch.testing.assert_close(core_pose[..., :3], native_pose[..., :3])
    torch.testing.assert_close(core_pose[..., 3:], native_pose[..., [6, 3, 4, 5]])
    torch.testing.assert_close(pose_wxyz_to_xyzw(core_pose), native_pose)
    core_pose.sum().backward()
    torch.testing.assert_close(native_pose.grad, torch.ones_like(native_pose))


@pytest.mark.parametrize("convert", [quaternion_xyzw_to_wxyz, quaternion_wxyz_to_xyzw])
def test_quaternion_conversion_rejects_pose_input(convert) -> None:
    with pytest.raises(ValueError, match="quaternion shape"):
        convert(torch.zeros(2, 7))


@pytest.mark.parametrize("convert", [pose_xyzw_to_wxyz, pose_wxyz_to_xyzw])
def test_pose_conversion_rejects_quaternion_input(convert) -> None:
    with pytest.raises(ValueError, match="pose shape"):
        convert(torch.zeros(2, 4))
