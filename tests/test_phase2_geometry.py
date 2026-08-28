from __future__ import annotations

import struct

import numpy as np
import torch

from isaaclab_pruning.geometry.cutter import (
    cutter_boxes_from_spec,
    fit_oriented_box,
    fit_oriented_box_from_stl,
    load_binary_stl,
)
from isaaclab_pruning.geometry.wood import nearby_wood_in_failure_zone
from isaaclab_pruning.robot import load_ur5e_pruner_spec
from isaaclab_pruning.task.success import OrientedBox


def _write_axis_aligned_box_stl(path, minimum, maximum) -> None:
    corners = np.array(
        [
            [minimum[0], minimum[1], minimum[2]],
            [maximum[0], minimum[1], minimum[2]],
            [maximum[0], maximum[1], minimum[2]],
            [minimum[0], maximum[1], minimum[2]],
            [minimum[0], minimum[1], maximum[2]],
            [maximum[0], minimum[1], maximum[2]],
            [maximum[0], maximum[1], maximum[2]],
            [minimum[0], maximum[1], maximum[2]],
        ]
    )
    faces = (
        (0, 1, 2),
        (0, 2, 3),
        (4, 6, 5),
        (4, 7, 6),
        (0, 4, 5),
        (0, 5, 1),
        (3, 2, 6),
        (3, 6, 7),
        (0, 3, 7),
        (0, 7, 4),
        (1, 5, 6),
        (1, 6, 2),
    )
    payload = bytearray(80)
    payload.extend(struct.pack("<I", 12))
    for i0, i1, i2 in faces:
        triangle = np.stack((corners[i0], corners[i1], corners[i2]))
        normal = np.cross(triangle[1] - triangle[0], triangle[2] - triangle[0])
        payload.extend(struct.pack("<12fH", *normal, *triangle.reshape(-1), 0))
    path.write_bytes(payload)


def test_ur5e_spec_uses_hardware_prefixes_and_does_not_spawn_slider() -> None:
    spec = load_ur5e_pruner_spec()
    assert spec.eef_body == "mock_pruner__tool0"
    assert spec.slider_held_fixed
    assert spec.action_dim == 7  # EE pose xyz+wxyz, not 6R+slider
    assert spec.joint_names_expr == ("ur5e__.*",)
    assert all("linear_slider" not in expr for expr in spec.joint_names_expr)
    assert [joint.name for joint in spec.arm_joints][0] == "ur5e__shoulder_pan_joint"
    assert spec.slider_joint.name == "linear_slider__joint1"
    assert spec.ik_method == "dls"
    assert spec.ik_relative_mode is False
    assert spec.cutter_source.startswith("stl_aabb_pybullet_tree_sim_")
    assert spec.mouth_half_extents_m[0] < spec.mouth_half_extents_m[1]


def test_stl_obb_recovers_a_known_box(tmp_path) -> None:
    path = tmp_path / "box.stl"
    _write_axis_aligned_box_stl(path, (-0.1, -0.2, -0.05), (0.1, 0.2, 0.05))
    triangles = load_binary_stl(path)
    assert triangles.shape == (12, 3, 3)
    corners = np.unique(np.round(triangles.reshape(-1, 3), decimals=6), axis=0)
    assert corners.shape == (8, 3)
    fitted = fit_oriented_box_from_stl(path)
    np.testing.assert_allclose(fitted.center, [0.0, 0.0, 0.0], atol=1e-5)
    _, _, extents = fit_oriented_box(corners)
    np.testing.assert_allclose(np.sort(extents), [0.05, 0.1, 0.2], atol=1e-5)
    assert fitted.vertex_count == 36


def test_pca_box_is_rotation_invariant() -> None:
    rng = np.random.default_rng(0)
    points = rng.uniform(-0.5, 0.5, size=(200, 3)) * np.array([2.0, 1.0, 0.5])
    _, _, extents_a = fit_oriented_box(points)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    _, _, extents_b = fit_oriented_box(points @ rotation.T)
    np.testing.assert_allclose(np.sort(extents_a), np.sort(extents_b), atol=1e-6)


def test_cutter_boxes_follow_eef_pose() -> None:
    pose = torch.tensor([[0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]])
    mouth, failure = cutter_boxes_from_spec(
        eef_pose_w=pose,
        mouth_half_extents=(0.02, 0.02, 0.01),
        failure_half_extents=(0.03, 0.03, 0.02),
        failure_offset_eef=(0.0, 0.0, -0.04),
    )
    torch.testing.assert_close(mouth.center_w, torch.tensor([[0.0, 0.0, 1.0]]))
    torch.testing.assert_close(failure.center_w, torch.tensor([[0.0, 0.0, 0.96]]))

    mouth_off, _ = cutter_boxes_from_spec(
        eef_pose_w=pose,
        mouth_half_extents=(0.02, 0.02, 0.01),
        failure_half_extents=(0.03, 0.03, 0.02),
        failure_offset_eef=(0.0, 0.0, -0.04),
        mouth_offset_eef=(0.0, 0.0, 0.02),
    )
    torch.testing.assert_close(mouth_off.center_w, torch.tensor([[0.0, 0.0, 1.02]]))


def test_nearby_wood_flags_a_non_target_cylinder() -> None:
    box = OrientedBox(
        center_w=torch.zeros(1, 3),
        rotation_bw=torch.eye(3).unsqueeze(0),
        half_extents=torch.tensor([[0.1, 0.1, 0.1]]),
    )
    centroids = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.3, 0.0]]])
    axes = torch.tensor([[[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    lengths = torch.tensor([[0.2, 0.2]])
    exclude = torch.tensor([[True, False]])
    # second cylinder is outside the box
    assert not nearby_wood_in_failure_zone(centroids, axes, lengths, box, exclude_mask=exclude).item()

    centroids = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]])
    hits = nearby_wood_in_failure_zone(centroids, axes, lengths, box, exclude_mask=exclude)
    assert hits.item()
