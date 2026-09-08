from __future__ import annotations

import math

import pytest
import torch

from isaaclab_pruning.robot import compose_physics_body_to_control_tool_pose, load_ur5e_pruner_spec
from isaaclab_pruning.sim.control_diagnostics import (
    bounded_damped_joint_delta,
    collect_nested_rigid_prims,
    contact_coverage,
    jacobian_body_index,
    jacobian_joint_columns,
    relative_pose_wxyz,
)


class _Prim:
    def __init__(self, name, rigid=False, children=()):
        self.name, self.rigid, self.children = name, rigid, children

    def GetChildren(self):
        return self.children


def test_nested_contact_discovery_does_not_stop_at_first_rigid_ancestor():
    wrist = _Prim("wrist", True, [_Prim("camera_site")])
    shoulder = _Prim("shoulder", True, [wrist])
    base = _Prim("base", True, [shoulder])
    root = _Prim("Robot", children=[base])
    assert [prim.name for prim in collect_nested_rigid_prims(root, lambda prim: prim.rigid)] == [
        "base",
        "shoulder",
        "wrist",
    ]


def test_contact_coverage_rejects_base_only_even_with_finite_tensor():
    report = contact_coverage(["base", "shoulder", "wrist"], ["base"])
    assert report["missing_body_names"] == ["shoulder", "wrist"]
    assert not report["complete"]
    assert contact_coverage(["base", "wrist"], ["wrist", "base"])["complete"]
    assert not contact_coverage(["base"], ["base", "base"])["complete"]


@pytest.mark.parametrize("fixed,rows,index", [(True, 9, 6), (False, 10, 7)])
def test_jacobian_body_axis_is_fixed_root_contract_not_usd_path_depth(fixed, rows, index):
    assert jacobian_body_index(body_index=7, body_count=10, jacobian_body_count=rows, fixed_base=fixed) == index


def test_jacobian_body_axis_rejects_backend_shape_change():
    with pytest.raises(ValueError, match="expected 9"):
        jacobian_body_index(body_index=7, body_count=10, jacobian_body_count=10, fixed_base=True)
    with pytest.raises(ValueError, match="no movable"):
        jacobian_body_index(body_index=0, body_count=10, jacobian_body_count=9, fixed_base=True)


@pytest.mark.parametrize(
    "indices,base,expected",
    [(slice(None), 0, [0, 1, 2, 3, 4, 5]), (slice(1, 5, 2), 6, [7, 9]), ([5, 0], 6, [11, 6])],
)
def test_scene_entity_joint_slice_is_valid_for_jacobian_columns(indices, base, expected):
    assert jacobian_joint_columns(indices, joint_count=6, base_dofs=base) == expected


@pytest.mark.parametrize("angle", [0.0, 0.7, math.pi / 2])
def test_tool_composition_agrees_before_or_after_root_frame_change(angle):
    """Nonidentity roots expose errors hidden by the original smoke's identity root."""
    spec = load_ur5e_pruner_spec()
    root = torch.tensor([[0.23, -0.41, 0.7, math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]], dtype=torch.float64)
    body = torch.tensor([[0.8, 0.4, 0.22, math.cos(0.3), math.sin(0.3), 0.0, 0.0]], dtype=torch.float64)
    world_tool = compose_physics_body_to_control_tool_pose(
        body, spec.control_tool_translation_in_physics_body_m, spec.control_tool_quaternion_wxyz_in_physics_body
    )
    root_body = relative_pose_wxyz(root, body)
    root_tool = compose_physics_body_to_control_tool_pose(
        root_body, spec.control_tool_translation_in_physics_body_m, spec.control_tool_quaternion_wxyz_in_physics_body
    )
    torch.testing.assert_close(root_tool, relative_pose_wxyz(root, world_tool), atol=1e-12, rtol=0.0)


def test_svd_dls_matches_regularized_solution_without_clipping():
    generator = torch.Generator().manual_seed(19)
    jacobian = torch.randn(3, 6, 6, generator=generator, dtype=torch.float64)
    error = torch.randn(3, 6, generator=generator, dtype=torch.float64) * 0.001
    delta, report = bounded_damped_joint_delta(jacobian, error)
    expected = jacobian.transpose(-2, -1) @ torch.linalg.solve(
        jacobian @ jacobian.transpose(-2, -1) + 0.05**2 * torch.eye(6), error.unsqueeze(-1)
    )
    torch.testing.assert_close(delta, expected.squeeze(-1), atol=1e-10, rtol=1e-6)
    torch.testing.assert_close(report["joint_delta_scale"], torch.ones(3, dtype=torch.float64))


def test_joint_trust_region_preserves_direction_and_descent_at_singularity():
    jacobian = torch.diag(torch.tensor([2.1, 1.6, 0.64, 0.53, 0.07, 1e-8])).unsqueeze(0)
    error = torch.tensor([[0.2, 0.3, -0.1, 0.2, 0.8, 0.5]])
    delta, report = bounded_damped_joint_delta(jacobian, error)
    assert torch.isfinite(delta).all()
    assert delta.abs().max() <= 0.05 + 1e-8
    assert report["joint_delta_scale"].item() < 1
    torch.testing.assert_close(delta, report["raw_joint_delta_rad"] * report["joint_delta_scale"][:, None])
    residual = error - (jacobian @ delta.unsqueeze(-1)).squeeze(-1)
    assert torch.linalg.vector_norm(residual) < torch.linalg.vector_norm(error)


def test_zero_pose_error_produces_no_joint_offset_even_at_rank_zero():
    delta, report = bounded_damped_joint_delta(torch.zeros(2, 6, 6), torch.zeros(2, 6))
    assert torch.equal(delta, torch.zeros_like(delta))
    assert torch.equal(report["joint_delta_scale"], torch.ones(2))


@pytest.mark.parametrize("parameter,value", [("damping", 0), ("damping", float("nan")), ("max_joint_delta_rad", -1)])
def test_invalid_controller_parameters_fail_closed(parameter, value):
    with pytest.raises(ValueError):
        bounded_damped_joint_delta(torch.eye(6)[None], torch.zeros(1, 6), **{parameter: value})


def test_nonfinite_controller_measurement_fails_closed():
    error = torch.zeros(1, 6)
    error[0, 2] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        bounded_damped_joint_delta(torch.eye(6)[None], error)


def test_smoke_elevates_robot_and_wall_without_weakening_hold_gate():
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "hpc/inner/smoke_env.py").read_text()
    assert "robot_cfg.init_state.pos = (0.0, 0.0, smoke_base_height_m)" in source
    assert "tof_smoke_target_position_w_m = smoke_wall_position" in source
    assert "hold_translation_drift_m < 5.0e-3" in source
