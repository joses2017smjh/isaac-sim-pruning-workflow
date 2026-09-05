from __future__ import annotations

import pytest
import torch

from isaaclab_pruning.baselines.tof_servo import deproject_tof, fit_branch_axis, scripted_tof_action
from isaaclab_pruning.robot.tool_frame import rotate_vector_wxyz
from isaaclab_pruning.sensors.fusion import fuse_depths


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("sign", [-1, 1])
def test_lateral_servo_reduces_measured_target_error(axis, sign):
    ranges = torch.full((1, 8, 8), 0.4)
    valid = torch.zeros_like(ranges, dtype=torch.bool)
    if axis == 0:
        valid[:, :, : 3 if sign < 0 else 0] = True
        if sign > 0:
            valid[:, :, 5:] = True
    elif sign < 0:
        valid[:, :3, :] = True
    else:
        valid[:, 5:, :] = True
    points = deproject_tof(ranges, valid, (0.0, 0.0, 0.0))
    centroid, _, _ = fit_branch_axis(points)
    action = scripted_tof_action(
        ranges,
        ranges,
        valid,
        valid,
        tof0_offset=(0, 0, 0),
        tof1_offset=(0, 0, 0),
        eef_translation_in_sensor_parent_m=(0, 0, 0),
    )
    assert abs(float(centroid[0, axis] - action[0, axis])) < abs(float(centroid[0, axis]))


def test_roll_improves_perpendicularity_and_preserves_forward_axis():
    ranges = torch.full((1, 8, 8), 0.4)
    valid = torch.eye(8, dtype=torch.bool).unsqueeze(0)
    points = deproject_tof(ranges, valid, (0, 0, 0))
    _, branch_axis, _ = fit_branch_axis(points)
    action = scripted_tof_action(
        ranges,
        ranges,
        valid,
        valid,
        tof0_offset=(0, 0, 0),
        tof1_offset=(0, 0, 0),
        eef_translation_in_sensor_parent_m=(0, 0, 0),
    )
    closing = rotate_vector_wxyz(action[:, 3:], torch.tensor([[1.0, 0.0, 0.0]]))
    forward = rotate_vector_wxyz(action[:, 3:], torch.tensor([[0.0, 0.0, 1.0]]))
    assert abs(float((closing * branch_axis).sum())) < abs(float(branch_axis[0, 0]))
    torch.testing.assert_close(forward, torch.tensor([[0.0, 0.0, 1.0]]))


def test_fusion_survives_invalid_sensors_and_preserves_gradients():
    depths = torch.tensor([[float("nan"), float("inf"), 100.0], [2.0, 3.0, 4.0]], requires_grad=True)
    valid = torch.tensor([[False, False, False], [True, True, True]])
    fused, variance = fuse_depths(depths, torch.ones_like(depths), valid=valid)
    torch.testing.assert_close(fused, torch.tensor([2.0, 3.0, 4.0]))
    torch.testing.assert_close(variance, torch.ones(3))
    fused.sum().backward()
    torch.testing.assert_close(depths.grad, torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]))


def test_all_missing_fusion_stays_explicitly_invalid():
    fused, variance = fuse_depths(torch.full((2, 3), float("nan")), torch.ones(2, 3))
    assert torch.isnan(fused).all()
    assert torch.isinf(variance).all()
