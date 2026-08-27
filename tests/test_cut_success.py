from __future__ import annotations

import torch

from isaaclab_pruning.task import OrientedBox, evaluate_cut_success, segment_intersects_obb


def _boxes(batch_size: int) -> tuple[OrientedBox, OrientedBox]:
    rotations = torch.eye(3).repeat(batch_size, 1, 1)
    mouth = OrientedBox(
        center_w=torch.zeros(batch_size, 3),
        rotation_bw=rotations,
        half_extents=torch.tensor([0.10, 0.10, 0.10]).repeat(batch_size, 1),
    )
    failure = OrientedBox(
        center_w=torch.tensor([0.0, 0.30, 0.0]).repeat(batch_size, 1),
        rotation_bw=rotations,
        half_extents=torch.tensor([0.05, 0.05, 0.05]).repeat(batch_size, 1),
    )
    return mouth, failure


def test_segment_obb_handles_parallel_inside_and_outside() -> None:
    box, _ = _boxes(2)
    starts = torch.tensor([[-0.2, 0.0, 0.0], [-0.2, 0.2, 0.0]])
    ends = torch.tensor([[0.2, 0.0, 0.0], [0.2, 0.2, 0.0]])
    hits = segment_intersects_obb(starts, ends, box)
    assert hits.tolist() == [True, False]


def test_cut_success_is_a_conjunction_of_geometric_gates() -> None:
    batch_size = 3
    mouth, failure = _boxes(batch_size)
    result = evaluate_cut_success(
        branch_centroid_w=torch.zeros(batch_size, 3),
        branch_axis_w=torch.tensor([[1.0, 0.0, 0.0]]).repeat(batch_size, 1),
        branch_length_m=torch.ones(batch_size),
        cutter_closing_axis_w=torch.tensor(
            [
                [0.0, 1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ]
        ),
        mouth_box=mouth,
        failure_box=failure,
        arm_collision=torch.tensor([False, False, True]),
    )

    assert result.mouth_hit.tolist() == [True, True, True]
    assert result.failure_clear.tolist() == [True, True, True]
    assert result.perpendicular.tolist() == [True, False, True]
    assert result.collision_free.tolist() == [True, True, False]
    assert result.success.tolist() == [True, False, False]
    torch.testing.assert_close(result.perpendicularity_error_deg, torch.tensor([0.0, 90.0, 0.0]))


def test_other_wood_can_invalidate_an_otherwise_valid_cut() -> None:
    mouth, failure = _boxes(1)
    result = evaluate_cut_success(
        branch_centroid_w=torch.zeros(1, 3),
        branch_axis_w=torch.tensor([[1.0, 0.0, 0.0]]),
        branch_length_m=torch.ones(1),
        cutter_closing_axis_w=torch.tensor([[0.0, 1.0, 0.0]]),
        mouth_box=mouth,
        failure_box=failure,
        other_wood_in_failure_zone=torch.tensor([True]),
    )
    assert not result.failure_clear.item()
    assert not result.success.item()
