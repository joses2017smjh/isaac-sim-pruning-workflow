from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from isaaclab_pruning.baselines.runtime_gate import (
    BaselineEvidenceError,
    validate_scripted_tof_runtime,
)


def _snapshot(frame: int) -> dict:
    sensor = {
        "class": "MultiMeshRayCasterCamera",
        "frame": [frame],
        "consumed_frame": [frame],
        "observation": {
            "shape": [1, 8, 8],
            "finite_fraction": 0.75,
            "valid_fraction": 0.75,
            "min_m": 0.25,
            "median_m": 0.40,
            "max_m": 0.65,
        },
    }
    return {
        "source": "live_multi_mesh_ray_caster",
        "range_limits_m": [0.03, 3.4],
        "sensors": {"tof0": deepcopy(sensor), "tof1": deepcopy(sensor)},
    }


def _validate(before: dict, after: dict, *, command: float = 0.01, motion: float = 0.002):
    return validate_scripted_tof_runtime(
        before,
        after,
        expected_num_envs=1,
        expected_hw=(8, 8),
        max_command_translation_m=[command],
        max_tool_translation_m=[motion],
    )


def test_runtime_gate_accepts_live_valid_advanced_frames_and_motion() -> None:
    result = _validate(_snapshot(2), _snapshot(4))

    assert result["source"] == "live_multi_mesh_ray_caster"
    assert result["sensors"]["tof0"]["frame_advance"] == [2]
    assert result["max_tool_translation_m"] == [0.002]


@pytest.mark.parametrize("phase", ["before", "after"])
def test_runtime_gate_rejects_non_live_sensor_source(phase: str) -> None:
    before, after = _snapshot(2), _snapshot(4)
    target = before if phase == "before" else after
    target["source"] = "external_debug_injection"

    with pytest.raises(BaselineEvidenceError, match="not live ray-caster data"):
        _validate(before, after)


@pytest.mark.parametrize("field", ["finite_fraction", "valid_fraction"])
def test_runtime_gate_rejects_invalid_ranges(field: str) -> None:
    before, after = _snapshot(2), _snapshot(4)
    after["sensors"]["tof1"]["observation"][field] = 0.0

    with pytest.raises(BaselineEvidenceError, match="no (finite observation|valid in-range) pixels"):
        _validate(before, after)


def test_runtime_gate_rejects_stale_frame_even_when_ranges_are_valid() -> None:
    with pytest.raises(BaselineEvidenceError, match="frame did not advance"):
        _validate(_snapshot(2), _snapshot(2))


@pytest.mark.parametrize(
    ("command", "motion", "message"),
    [(0.0, 0.002, "meaningful command"), (0.01, 0.0, "measurable response")],
)
def test_runtime_gate_rejects_noop_controller_or_tool(command: float, motion: float, message: str) -> None:
    with pytest.raises(BaselineEvidenceError, match=message):
        _validate(_snapshot(2), _snapshot(4), command=command, motion=motion)


def test_runner_never_marks_curobo_complete_from_configuration_only() -> None:
    runner = (Path(__file__).resolve().parents[1] / "hpc" / "inner" / "run_baselines.py").read_text(encoding="utf-8")

    assert runner.index('report["tof_before"] = tof_before') < runner.index(
        "runtime_gate = validate_scripted_tof_runtime("
    )
    assert runner.index("runtime_gate = validate_scripted_tof_runtime(") < runner.index('report["scripted_tof"] = True')
    assert '"runtime_exercised": False' in runner
    assert '"oracle_complete": False' in runner
    assert 'report["curobo_oracle"] = False' in runner
    assert 'report["curobo_oracle"] = bool(' not in runner
    assert "no task success rate is measured" in runner


def test_runner_uses_strict_json_and_rejects_nonfinite_runtime_values() -> None:
    runner = (Path(__file__).resolve().parents[1] / "hpc" / "inner" / "run_baselines.py").read_text(encoding="utf-8")

    assert "allow_nan=False" in runner
    assert "Warm-up reward is non-finite" in runner
    assert "Scripted reward is non-finite" in runner
    assert "Scripted ToF command delta is non-finite" in runner
    assert "Maximum measured tool translation is non-finite" in runner
