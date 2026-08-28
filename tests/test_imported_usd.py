from __future__ import annotations

import pytest

from isaaclab_pruning.robot import imported_usd_path, load_ur5e_pruner_spec
from isaaclab_pruning.usd import imported_usd_payload_paths, load_imported_usd

_USD = imported_usd_path()


def test_yaml_points_at_the_imported_usd() -> None:
    path = imported_usd_path()
    assert path.name == "ur5e_pruner_abs.usda"
    assert "artifacts/usd" in str(path)


@pytest.mark.skipif(not _USD.is_file(), reason="USD import artifacts are not on this machine")
def test_imported_usd_composes_robot_and_physx_payloads() -> None:
    names = {path.name for path in imported_usd_payload_paths()}
    assert "ur5e_pruner_abs.usda" in names
    assert "robot.usda" in names
    assert "physx.usda" in names


@pytest.mark.skipif(not _USD.is_file(), reason="USD import artifacts are not on this machine")
def test_imported_usd_has_ur5e_and_mock_pruner_not_slider() -> None:
    text = load_imported_usd()
    spec = load_ur5e_pruner_spec()
    for joint in spec.arm_joints:
        assert joint.name in text
    assert "mock_pruner__tool0" in text
    assert "mock_pruner__tof0" in text
    assert "linear_slider__joint1" not in text
    for expr in spec.joint_names_expr:
        assert "linear_slider" not in expr
