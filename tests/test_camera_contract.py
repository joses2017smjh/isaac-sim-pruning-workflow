from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from isaaclab_pruning.data import (
    distance_to_camera_from_planar_depth,
    isaac_camera_to_ann,
    planar_depth_from_distance_to_camera,
    pose_from_ann,
    unproject_planar_depth,
)


def _fake_camera(
    position_w: tuple[float, float, float],
    quaternion_wxyz: tuple[float, float, float, float],
    intrinsic: np.ndarray,
    shape: tuple[int, int],
) -> SimpleNamespace:
    data = SimpleNamespace(
        intrinsic_matrices=intrinsic[None],
        pos_w=np.asarray([position_w], dtype=np.float32),
        quat_w_opencv=np.asarray([quaternion_wxyz], dtype=np.float32),
        output={},
    )
    return SimpleNamespace(data=data, image_shape=shape)


def test_explicit_and_legacy_camera_poses_match() -> None:
    intrinsic = np.array([[150.0, 0.0, 50.0], [0.0, 150.0, 50.0], [0.0, 0.0, 1.0]])
    camera = _fake_camera(
        position_w=(-2.0, 0.0, 0.0),
        quaternion_wxyz=(0.5, 0.5, 0.5, 0.5),
        intrinsic=intrinsic,
        shape=(101, 101),
    )

    annotation = isaac_camera_to_ann(camera, tree_id="lpy_envy_00042", shot=1)
    explicit_intrinsic, explicit_pose = pose_from_ann(annotation)
    legacy_annotation = {**annotation}
    legacy_annotation.pop("_T_wc")
    legacy_intrinsic, legacy_pose = pose_from_ann(legacy_annotation)

    np.testing.assert_allclose(explicit_intrinsic, intrinsic)
    np.testing.assert_allclose(legacy_intrinsic, intrinsic)
    np.testing.assert_allclose(legacy_pose, explicit_pose, atol=1e-6)
    assert annotation["camera"]["intrinsics"]["width"] == 101
    assert annotation["camera"]["intrinsics"]["height"] == 101
    assert annotation["tree_id"] == "lpy_envy_00042"


def test_three_view_cube_contract_recovers_metric_aabb() -> None:
    """Analytic counterpart of the required Isaac cube integration test."""
    image_shape = (101, 101)
    intrinsic = np.array([[150.0, 0.0, 50.0], [0.0, 150.0, 50.0], [0.0, 0.0, 1.0]])
    depth = np.full(image_shape, 1.5, dtype=np.float32)
    views = [
        ((0.0, 0.0, -2.0), (1.0, 0.0, 0.0, 0.0)),
        ((-2.0, 0.0, 0.0), (0.5, 0.5, 0.5, 0.5)),
        ((0.0, -2.0, 0.0), (2**-0.5, -(2**-0.5), 0.0, 0.0)),
    ]

    clouds = []
    for position, quaternion in views:
        camera = _fake_camera(position, quaternion, intrinsic, image_shape)
        annotation = isaac_camera_to_ann(camera)
        camera_intrinsic, transform_wc = pose_from_ann(annotation)
        clouds.append(unproject_planar_depth(depth, camera_intrinsic, transform_wc))

    points = np.concatenate(clouds)
    minimum = points.min(axis=0)
    maximum = points.max(axis=0)
    np.testing.assert_allclose(maximum - minimum, np.ones(3), atol=5e-6)
    np.testing.assert_allclose((maximum + minimum) / 2, np.zeros(3), atol=5e-6)


def test_planar_depth_is_flat_while_camera_distance_is_a_bowl() -> None:
    intrinsic = np.array([[2.0, 0.0, 2.0], [0.0, 2.0, 2.0], [0.0, 0.0, 1.0]])
    planar = np.full((5, 5), 2.0, dtype=np.float32)
    distance = distance_to_camera_from_planar_depth(planar, intrinsic)

    assert distance[2, 2] == 2.0
    assert distance[0, 0] > distance[2, 2]
    np.testing.assert_allclose(
        planar_depth_from_distance_to_camera(distance, intrinsic),
        planar,
        atol=1e-6,
    )
