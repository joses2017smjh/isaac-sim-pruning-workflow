"""Isaac-free checks for the GPU recorder's schedule and JSON contract."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize(
    "measured,expected,visible", [(0.344, 0.35, True), (0.037, 0.35, False), (np.inf, 0.35, False)]
)
def test_seed_visibility_rejects_tool_occlusion_before_tracking(measured, expected, visible):
    result = _runner().seed_visibility(np.full((320, 480), measured), [240, 160], expected)
    assert result["visible"] is visible


def test_seed_visibility_rejects_out_of_frame_projection():
    assert not _runner().seed_visibility(np.ones((10, 10)), [12, 5], 1.0)["visible"]


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


def _capture_loop():
    return next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(_runner().main)))
        if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == "index"
    )


def _calls(node, name):
    return any(
        isinstance(child, ast.Call)
        and (
            isinstance(child.func, ast.Name)
            and child.func.id == name
            or isinstance(child.func, ast.Attribute)
            and child.func.attr == name
        )
        for child in ast.walk(node)
    )


def _execute_capture_nodes(nodes, namespace):
    exec(compile(ast.Module(body=nodes, type_ignores=[]), "<actual capture loop>", "exec"), namespace)


def _live_gate_fixture(range_m, *, permit_detach=True):
    events = []

    class Demo:
        stopped_reason = None
        cut_step = SimpleNamespace(phase="closing", detach_event=False)

        def stop(self, reason):
            events.append("stop")
            self.stopped_reason = reason

        def observe(self, *args, **kwargs):
            events.append("observe")
            self.cut_step = SimpleNamespace(
                phase="stopped" if self.stopped_reason else "closing",
                stopped_reason=self.stopped_reason,
                detach_event=permit_detach and self.stopped_reason is None,
            )
            return {"cut": {"detached": self.cut_step.detach_event}}

    def guard(*args):
        events.append("range_gate")
        return _runner().sensor_guard(*args)

    def detach():
        events.append("detach")
        return True

    valid = bool(np.isfinite(range_m))
    env = SimpleNamespace(
        tof0=torch.tensor([[range_m]]),
        tof1=torch.tensor([[range_m]]),
        tof0_valid=torch.tensor([[valid]]),
        tof1_valid=torch.tensor([[valid]]),
        blender_scene=SimpleNamespace(detach=detach),
    )
    namespace = {
        "np": np,
        "env": env,
        "demo": Demo(),
        "sensor_guard": guard,
        "missing_sensor_frames": 0,
        "stopped_reason": None,
        "report": {},
        "index": 9,
        "fps": 10,
        "wrist": np.zeros((2, 2, 3), dtype=np.uint8),
        "depth": np.ones((2, 2)),
        "camera_matrix": np.eye(3),
        "optical_transform": lambda *args: np.eye(4),
        "camera_position": np.zeros(3),
        "camera_rotation": np.eye(3),
        "tool": np.array([0, 0, 0, 1, 0, 0, 0]),
        "contact_force_n": 0.0,
        "detach_requested": False,
    }
    return namespace, events


@pytest.mark.parametrize("range_m,expected_detach", [(0.03, False), (0.08, True)])
def test_fresh_capture_tof_gate_precedes_closure_observation_and_detachment(range_m, expected_detach):
    # Execute the actual renderer block, not a second implementation of its
    # ordering. The fake cut gate is ready to detach unless stopped first.
    live_block = next(node for node in _capture_loop().body if isinstance(node, ast.If) and _calls(node, "observe"))
    namespace, events = _live_gate_fixture(range_m)
    _execute_capture_nodes([live_block], namespace)
    assert events[0] == "range_gate"
    assert namespace["detach_requested"] is expected_detach
    if expected_detach:
        assert events == ["range_gate", "observe", "detach"]
    else:
        assert events == ["range_gate", "stop", "observe"]
        assert namespace["stopped_reason"] == "tof_minimum_clearance"
        assert namespace["report"]["sensor_stop_frame"] == 9


def test_missing_tof_count_advances_once_per_live_capture_after_startup():
    loop = _capture_loop()
    pre_guard = next(
        node
        for node in loop.body
        if isinstance(node, ast.If) and _calls(node, "sensor_guard") and not _calls(node, "observe")
    )
    live_block = next(node for node in loop.body if isinstance(node, ast.If) and _calls(node, "observe"))
    namespace, events = _live_gate_fixture(float("inf"), permit_detach=False)
    for index in range(1, 5):
        namespace["index"] = index
        _execute_capture_nodes([pre_guard, live_block], namespace)
        assert namespace["missing_sensor_frames"] == index
        assert events.count("range_gate") == index
        assert namespace["stopped_reason"] == ("both_tof_missing_4_frames" if index == 4 else None)
    assert "detach" not in events


@pytest.mark.parametrize("fail_step", [None, 2])
def test_renderer_acknowledges_applied_vision_only_after_physics_steps(fail_step):
    loop = _capture_loop()
    physics = next(node for node in loop.body if isinstance(node, ast.For) and _calls(node, "step"))
    acknowledge = next(node for node in loop.body if isinstance(node, ast.If) and _calls(node, "command_applied"))
    assert loop.body.index(physics) < loop.body.index(acknowledge)
    events = []

    def step(action):
        events.append("step")
        if len(events) == fail_step:
            raise RuntimeError("physical step failed")

    namespace = {
        "steps_per_frame": 3,
        "action": "issued action",
        "env": SimpleNamespace(step=step),
        "demo": SimpleNamespace(command_applied=lambda *args: events.append("applied")),
        "phase": "vision_approach",
        "vision_decision": {"state": "tracking"},
    }
    if fail_step:
        with pytest.raises(RuntimeError, match="physical step failed"):
            _execute_capture_nodes([physics, acknowledge], namespace)
        assert events == ["step", "step"]
    else:
        _execute_capture_nodes([physics, acknowledge], namespace)
        assert events == ["step", "step", "step", "applied"]


def test_live_return_error_uses_home_before_first_command_not_first_captured_pose():
    assignment = next(
        node
        for node in ast.walk(ast.parse(inspect.getsource(_runner().main)))
        if isinstance(node, ast.Assign)
        and any(ast.unparse(target) == "report['metrics']['retreat_return_error_m']" for target in node.targets)
    )
    home = np.array([0.2, 0.3, 0.8])
    positions = np.array([home + [0.004, 0, 0], home + [0.10, 0, 0], home])
    namespace = {
        "np": np,
        "report": {"metrics": {}},
        "positions": positions,
        "displacements": np.linalg.norm(positions - positions[0], axis=1),
        "initial_tool": torch.tensor([np.r_[home, [1, 0, 0, 0]].tolist()], dtype=torch.float64),
    }
    _execute_capture_nodes([assignment], namespace)
    assert namespace["report"]["metrics"]["retreat_return_error_m"] == pytest.approx(0.0, abs=1e-12)
