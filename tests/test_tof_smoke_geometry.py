from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from isaaclab_pruning.sensors.tof_raycaster import VL53L8CX_INTRINSICS
from isaaclab_pruning.sim.prim_paths import TOF_SMOKE_TARGET_PRIM_EXPR, TREE_PRIM_EXPR
from isaaclab_pruning.sim.tof_smoke_geometry import TOF_SMOKE_TARGET, enable_tof_smoke_target

# Zero-joint poses independently calculated from the reviewed flattened URDF.
_SENSOR_ORIGINS_W_M = (
    np.array([0.77055861, 0.42223695, 0.06422783]),
    np.array([0.86426313, 0.42218546, 0.06422754]),
)
_CAMERA_TO_WORLD = np.array(
    [
        [-0.99999985, 0.00000050, 0.00054957],
        [0.00054956, -0.00313400, 0.99999494],
        [0.00000270, 0.99999500, 0.00313419],
    ]
)


def _pixel_rays_ros() -> np.ndarray:
    """Match v60's pixel-centre pinhole pattern in ROS camera coordinates."""
    intrinsics = VL53L8CX_INTRINSICS
    fx = intrinsics.focal_length_px
    cx, cy = intrinsics.principal_point_px
    directions = []
    for row in range(intrinsics.height):
        for column in range(intrinsics.width):
            direction = np.array(
                [(column + 0.5 - cx) / fx, (row + 0.5 - cy) / fx, 1.0],
                dtype=np.float64,
            )
            directions.append(direction / np.linalg.norm(direction))
    return np.asarray(directions)


def test_opt_in_routes_both_raycasters_without_changing_default() -> None:
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=1),
        tof0_cfg=SimpleNamespace(mesh_prim_paths=[TREE_PRIM_EXPR]),
        tof1_cfg=SimpleNamespace(mesh_prim_paths=[TREE_PRIM_EXPR]),
        tof_smoke_target_enabled=False,
    )

    assert cfg.tof_smoke_target_enabled is False
    enable_tof_smoke_target(cfg)

    assert cfg.tof_smoke_target_enabled is True
    assert TOF_SMOKE_TARGET.prim_expr == TOF_SMOKE_TARGET_PRIM_EXPR
    assert TOF_SMOKE_TARGET.prim_expr.startswith("/World/")
    assert "{ENV_REGEX_NS}" not in TOF_SMOKE_TARGET.prim_expr
    assert cfg.tof0_cfg.mesh_prim_paths == [TOF_SMOKE_TARGET.prim_expr]
    assert cfg.tof1_cfg.mesh_prim_paths == [TOF_SMOKE_TARGET.prim_expr]


def test_opt_in_rejects_unvalidated_batched_scene() -> None:
    cfg = SimpleNamespace(
        scene=SimpleNamespace(num_envs=2),
        tof0_cfg=SimpleNamespace(mesh_prim_paths=[]),
        tof1_cfg=SimpleNamespace(mesh_prim_paths=[]),
    )

    with pytest.raises(ValueError, match="only for one environment"):
        enable_tof_smoke_target(cfg)


def test_smoke_wall_covers_every_pixel_ray_from_both_reviewed_sites() -> None:
    target = TOF_SMOKE_TARGET
    center = np.asarray(target.position_w_m)
    size = np.asarray(target.size_m)
    front_y = center[1] - 0.5 * size[1]
    lower = center - 0.5 * size
    upper = center + 0.5 * size
    rays_w = _pixel_rays_ros() @ _CAMERA_TO_WORLD.T

    for origin in _SENSOR_ORIGINS_W_M:
        distance = (front_y - origin[1]) / rays_w[:, 1]
        hits = origin + distance[:, None] * rays_w

        assert np.all(distance > 0.03)
        assert np.all(distance < 3.4)
        assert np.all(hits[:, 0] > lower[0])
        assert np.all(hits[:, 0] < upper[0])
        assert np.all(hits[:, 2] > lower[2])
        assert np.all(hits[:, 2] < upper[2])
        # Preserve a generous margin for the hold tolerance and tiny URDF FK
        # rounding differences on the imported PhysX articulation.
        margin = min(
            float((hits[:, 0] - lower[0]).min()),
            float((upper[0] - hits[:, 0]).min()),
            float((hits[:, 2] - lower[2]).min()),
            float((upper[2] - hits[:, 2]).min()),
        )
        assert margin > 0.14


def test_five_mm_optical_motion_changes_every_wall_range_toward_target() -> None:
    target = TOF_SMOKE_TARGET
    front_y = target.position_w_m[1] - 0.5 * target.size_m[1]
    rays_w = _pixel_rays_ros() @ _CAMERA_TO_WORLD.T
    optical_forward_w = _CAMERA_TO_WORLD[:, 2]

    for origin in _SENSOR_ORIGINS_W_M:
        before = (front_y - origin[1]) / rays_w[:, 1]
        moved_origin = origin + 0.005 * optical_forward_w
        after = (front_y - moved_origin[1]) / rays_w[:, 1]

        assert np.all(before - after > 0.005)
