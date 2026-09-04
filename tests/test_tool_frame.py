from __future__ import annotations

import math

import torch

from isaaclab_pruning.robot.tool_frame import (
    compose_physics_body_to_control_tool_pose,
    control_tool_pose_to_physics_body_pose,
    point_offset_in_jacobian_frame,
    shift_spatial_jacobian_to_point,
)


def test_compose_body_to_tool_pose_rotates_fixed_translation() -> None:
    half_sqrt_two = math.sqrt(0.5)
    body_pose = torch.tensor([1.0, 2.0, 3.0, half_sqrt_two, 0.0, 0.0, half_sqrt_two])

    tool_pose = compose_physics_body_to_control_tool_pose(body_pose, (0.1, 0.0, 0.0))

    torch.testing.assert_close(tool_pose[:3], torch.tensor([1.0, 2.1, 3.0]))
    torch.testing.assert_close(tool_pose[3:], body_pose[3:])


def test_absolute_tool_command_round_trips_to_body_pose_in_batches() -> None:
    half_sqrt_two = math.sqrt(0.5)
    body_poses = torch.tensor(
        [
            [0.2, -0.4, 1.0, 1.0, 0.0, 0.0, 0.0],
            [-0.3, 0.7, 0.1, half_sqrt_two, half_sqrt_two, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    translation = torch.tensor([0.04, -0.02, 0.16], dtype=torch.float64)
    relative_quaternion = torch.tensor(
        [half_sqrt_two, 0.0, 0.0, half_sqrt_two],
        dtype=torch.float64,
    )

    tool_commands = compose_physics_body_to_control_tool_pose(
        body_poses,
        translation,
        relative_quaternion,
    )
    recovered_body_poses = control_tool_pose_to_physics_body_pose(
        tool_commands,
        translation,
        relative_quaternion,
    )

    torch.testing.assert_close(recovered_body_poses, body_poses, atol=1.0e-12, rtol=1.0e-12)


def test_pose_helpers_broadcast_offsets_over_a_batch() -> None:
    body_poses = torch.tensor(
        [
            [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0],
        ]
    )
    offsets = torch.tensor([[0.0, 0.0, 0.1], [0.0, 0.0, 0.2]])

    tool_poses = compose_physics_body_to_control_tool_pose(body_poses, offsets)

    assert tool_poses.shape == (2, 7)
    torch.testing.assert_close(tool_poses[:, :3], body_poses[:, :3] + offsets)


def test_jacobian_shift_sign_for_pure_rotation() -> None:
    # A +z angular velocity at a point +x from the body origin produces +y
    # linear velocity: omega x r = [0, 1, 0].
    body_jacobian = torch.zeros(6, 1)
    body_jacobian[5, 0] = 1.0
    offset = torch.tensor([1.0, 0.0, 0.0])

    tool_jacobian = shift_spatial_jacobian_to_point(body_jacobian, offset)

    torch.testing.assert_close(tool_jacobian[:3, 0], torch.tensor([0.0, 1.0, 0.0]))
    torch.testing.assert_close(tool_jacobian[3:, :], body_jacobian[3:, :])
    torch.testing.assert_close(body_jacobian[:3, 0], torch.zeros(3))


def test_jacobian_shift_broadcasts_one_jacobian_over_offsets() -> None:
    body_jacobian = torch.zeros(6, 1)
    body_jacobian[5, 0] = 1.0
    offsets = torch.tensor([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0]])

    tool_jacobians = shift_spatial_jacobian_to_point(body_jacobian, offsets)

    assert tool_jacobians.shape == (2, 6, 1)
    torch.testing.assert_close(
        tool_jacobians[:, :3, 0],
        torch.tensor([[0.0, 1.0, 0.0], [-2.0, 0.0, 0.0]]),
    )


def test_batched_shift_matches_rigid_point_velocity_identity() -> None:
    generator = torch.Generator().manual_seed(7)
    body_jacobian = torch.randn(3, 6, 5, generator=generator, dtype=torch.float64)
    offsets = torch.randn(3, 3, generator=generator, dtype=torch.float64)
    joint_velocity = torch.randn(3, 5, generator=generator, dtype=torch.float64)

    tool_jacobian = shift_spatial_jacobian_to_point(body_jacobian, offsets)
    body_twist = torch.matmul(body_jacobian, joint_velocity.unsqueeze(-1)).squeeze(-1)
    tool_twist = torch.matmul(tool_jacobian, joint_velocity.unsqueeze(-1)).squeeze(-1)
    expected_linear = body_twist[:, :3] + torch.linalg.cross(body_twist[:, 3:], offsets, dim=-1)

    torch.testing.assert_close(tool_twist[:, :3], expected_linear)
    torch.testing.assert_close(tool_twist[:, 3:], body_twist[:, 3:])


def test_body_offset_is_rotated_into_jacobian_coordinates() -> None:
    half_sqrt_two = math.sqrt(0.5)
    body_quaternion_wxyz = torch.tensor(
        [half_sqrt_two, 0.0, 0.0, half_sqrt_two],
        dtype=torch.float64,
    )

    offset = point_offset_in_jacobian_frame(body_quaternion_wxyz, (0.2, 0.0, 0.0))

    torch.testing.assert_close(offset, torch.tensor([0.0, 0.2, 0.0], dtype=torch.float64), atol=1.0e-12, rtol=0.0)
