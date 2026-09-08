"""Isaac-free checks for the GPU recorder's schedule and JSON contract."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch


def _runner():
    path = Path(__file__).resolve().parents[1] / "hpc/inner/render_pruning_workflow.py"
    spec = importlib.util.spec_from_file_location("pruning_workflow_recorder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capture_schedule_approaches_and_returns_without_teleport():
    runner = _runner()
    sequence = [runner.episode_command(index, 140) for index in range(140)]
    phases = {phase for phase, _ in sequence}
    assert phases == {"observe", "approach", "align", "inspect_at_standoff", "retreat"}
    positions = np.asarray([position for _, position in sequence])
    np.testing.assert_allclose(positions[0], 0.0)
    np.testing.assert_allclose(positions[-1], 0.0, atol=1.0e-12)
    assert positions[:, 1].max() == pytest.approx(0.25)
    assert np.linalg.norm(np.diff(positions, axis=0), axis=1).max() < 0.014


@pytest.mark.parametrize("index,count", [(-1, 140), (140, 140), (0, 29)])
def test_capture_schedule_rejects_invalid_indices(index, count):
    with pytest.raises(ValueError):
        _runner().episode_command(index, count)


def test_raw_missing_sensor_samples_are_strict_json_nulls():
    values = {"range_m": np.array([[0.10, np.inf], [np.nan, 0.30]]), "valid": [True, False]}
    encoded = json.dumps(_runner()._json_safe(values), allow_nan=False)
    assert json.loads(encoded) == {"range_m": [[0.10, None], [None, 0.30]], "valid": [True, False]}


def test_exterior_wrist_mount_is_fixed_and_looks_toward_initial_fixture():
    runner = _runner()
    offset = np.asarray(runner.WRIST_POSITION_IN_TOOL_M)
    # The prior optical center at [0, 0, -0.025] was inside the mock-pruner.
    assert abs(offset[1]) >= 0.09
    initial_target = np.array([0.0, 0.0, 0.35])
    rotation = runner.optical_rotation_from_forward(initial_target - offset)
    np.testing.assert_allclose(rotation.T @ rotation, np.eye(3), atol=1.0e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)
    target_optical = rotation.T @ (initial_target - offset)
    np.testing.assert_allclose(target_optical[:2], 0.0, atol=1.0e-12)
    # It stays in the 480x320 image at 0.10 m standoff without retargeting.
    final_optical = rotation.T @ (np.array([0.0, 0.0, 0.10]) - offset)
    pixels = final_optical[:2] / final_optical[2] * 320.0 + [240, 160]
    assert 0 < pixels[0] < 480 and 0 < pixels[1] < 320


@pytest.mark.parametrize("direction", [[0, 0, 0], [0, np.inf, 1], [0, 1, 0], [1, 0]])
def test_camera_mount_rejects_degenerate_directions(direction):
    with pytest.raises(ValueError):
        _runner().optical_rotation_from_forward(direction)


def test_live_tof_stop_gate_records_loss_without_inventing_measurements():
    runner = _runner()
    assert runner.sensor_guard([0.1, 0.3], [True, True], 3) == (None, 0)
    assert runner.sensor_guard([0.04, 0.3], [True, True], 0) == ("tof_minimum_clearance", 0)
    assert runner.sensor_guard([0.04, 0.3], [False, True], 0) == (None, 0)
    assert runner.sensor_guard([np.inf, np.nan], [False, False], 2) == (None, 3)
    assert runner.sensor_guard([np.inf, np.nan], [False, False], 3) == ("both_tof_missing_4_frames", 4)


def test_stopped_capture_measures_held_action_not_continuing_schedule():
    runner = _runner()
    # Execute the recorder's actual telemetry assignment without starting Kit.
    # A translated root with a 90-degree yaw catches both frame and stop errors.
    assignment = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(runner.main)))
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "error_m" for target in node.targets)
    )
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    root = np.array([1.0, -2.0, 0.7])
    held_position_b = np.array([0.2, 0.3, 0.4])
    held_position_w = root + rotation @ held_position_b
    scheduled_offset = np.array([0.0, 0.25, 0.0])
    namespace = {
        "command_tracking_error_m": runner.command_tracking_error_m,
        "tool": np.r_[held_position_w + [0.0004, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]],
        "action": torch.tensor([np.r_[held_position_b, [1.0, 0.0, 0.0, 0.0]].tolist()]),
        "root_pose_now": torch.tensor([np.r_[root, [0.0, 0.0, 2**-0.5, 2**-0.5]].tolist()]),
        "matrix_from_quat": lambda _: torch.from_numpy(rotation).unsqueeze(0),
        # Keep the former expression's inputs so reverting the call produces
        # the incorrect scheduled-path error rather than just a NameError.
        "initial_tool": torch.tensor([np.r_[held_position_w, [1.0, 0.0, 0.0, 0.0]].tolist()]),
        "offset": scheduled_offset,
        "root_rotation_w": rotation,
        "np": np,
    }
    exec(compile(ast.Module(body=[assignment], type_ignores=[]), "<tracking assignment>", "exec"), namespace)
    assert namespace["error_m"] == pytest.approx(0.0004, abs=1.0e-7)


@pytest.mark.parametrize("invalid", [[0.0, np.nan, 0.0], [0.0, 0.0]])
def test_tracking_error_rejects_invalid_positions(invalid):
    with pytest.raises(ValueError):
        _runner().command_tracking_error_m(invalid, [0, 0, 0], [0, 0, 0], np.eye(3))
