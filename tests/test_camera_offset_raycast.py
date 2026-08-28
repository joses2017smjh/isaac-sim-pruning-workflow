from __future__ import annotations

import numpy as np

from isaaclab_pruning.geometry import Cylinder
from isaaclab_pruning.geometry.cut_point import CutPoint
from isaaclab_pruning.sensors.wrist_camera import (
    CANDIDATES,
    ray_finite_cylinder_t,
    ray_obb_t,
    score_candidate_on_tree,
    select_wrist_camera,
)


def _unit_cylinder() -> Cylinder:
    return Cylinder(
        record_id="target",
        part_name="spur_1",
        centroid=np.array([0.0, 0.4, 0.3]),
        orientation=np.array([1.0, 0.0, 0.0]),
        radius=0.006,
        length=0.08,
    )


def test_ray_hits_a_unit_cylinder_on_the_axis() -> None:
    cylinder = Cylinder(
        "0",
        "trunk_1",
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        0.5,
        2.0,
    )
    t = ray_finite_cylinder_t(np.array([-2.0, 0.0, 0.0]), np.array([1.0, 0.0, 0.0]), cylinder)
    assert abs(t - 1.5) < 1e-6


def test_ray_obb_hits_a_box() -> None:
    t = ray_obb_t(
        np.array([-1.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.eye(3),
        np.array([0.2, 0.2, 0.2]),
    )
    assert abs(t - 0.8) < 1e-6


def test_occluder_between_camera_and_cut_is_wood_occlusion() -> None:
    target = _unit_cylinder()
    occluder = Cylinder(
        "block",
        "branch_1",
        np.array([0.0, 0.2, 0.3]),
        np.array([1.0, 0.0, 0.0]),
        0.05,
        0.2,
    )
    cut = CutPoint(
        record_id=target.record_id,
        part_name=target.part_name,
        position_w=target.centroid.copy(),
        axis_w=target.orientation.copy(),
        radius_m=target.radius,
        length_m=target.length,
        neighbor_count=0,
    )
    intrinsic = np.array([[500.0, 0.0, 320.0], [0.0, 500.0, 240.0], [0.0, 0.0, 1.0]])
    clear = score_candidate_on_tree(
        CANDIDATES[0],
        [target],
        [cut],
        mouth_offset_eef=(0.0, 0.0, 0.025),
        mouth_half_extents=(0.001, 0.001, 0.001),
        intrinsic=intrinsic,
        width=640,
        height=480,
    )
    blocked = score_candidate_on_tree(
        CANDIDATES[0],
        [target, occluder],
        [cut],
        mouth_offset_eef=(0.0, 0.0, 0.025),
        mouth_half_extents=(0.001, 0.001, 0.001),
        intrinsic=intrinsic,
        width=640,
        height=480,
    )
    assert blocked["n_wood_occluded"] >= clear["n_wood_occluded"]
    assert blocked["n_visible"] <= clear["n_visible"]


def test_select_wrist_camera_requires_a_visible_cut() -> None:
    assert select_wrist_camera([]) is None
    assert select_wrist_camera([{"name": "a", "n_visible": 0, "n_jaw_occluded": 0}]) is None
    winner = select_wrist_camera(
        [
            {"name": "aft_dovetail", "n_visible": 1, "n_jaw_occluded": 9},
            {"name": "close_lateral", "n_visible": 4, "n_jaw_occluded": 1},
        ]
    )
    assert winner is not None
    assert winner["name"] == "close_lateral"
    assert winner["selected"] is True
