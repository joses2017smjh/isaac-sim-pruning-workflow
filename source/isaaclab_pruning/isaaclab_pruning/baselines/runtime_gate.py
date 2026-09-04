"""Fail-closed validation for the scripted ToF runtime baseline.

This module deliberately has no Isaac Lab or Torch dependency so the evidence
contract can be unit-tested on a CPU login node.  The GPU runner supplies
JSON-safe snapshots from :meth:`PruningEnv.tof_state` plus measured command and
tool-motion maxima.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

LIVE_TOF_SOURCE = "live_multi_mesh_ray_caster"
REQUIRED_TOF_SENSORS = ("tof0", "tof1")


class BaselineEvidenceError(RuntimeError):
    """Raised when a scripted-baseline runtime claim lacks required evidence."""


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise BaselineEvidenceError(f"{label} must be a finite number, not {value!r}.")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise BaselineEvidenceError(f"{label} must be a finite number, got {value!r}.") from error
    if not math.isfinite(result):
        raise BaselineEvidenceError(f"{label} must be finite, got {value!r}.")
    return result


def _frames(value: Any, label: str, expected_num_envs: int) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BaselineEvidenceError(f"{label} must contain one frame counter per environment.")
    if len(value) != expected_num_envs:
        raise BaselineEvidenceError(f"{label} has {len(value)} counters; expected {expected_num_envs}.")
    frames: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int):
            raise BaselineEvidenceError(f"{label} contains a non-integer counter: {item!r}.")
        frames.append(item)
    return frames


def _motion_values(value: Any, label: str, expected_num_envs: int) -> list[float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise BaselineEvidenceError(f"{label} must contain one value per environment.")
    if len(value) != expected_num_envs:
        raise BaselineEvidenceError(f"{label} has {len(value)} values; expected {expected_num_envs}.")
    return [_number(item, f"{label}[{index}]") for index, item in enumerate(value)]


def _snapshot_sensor(
    state: Mapping[str, Any],
    *,
    phase: str,
    name: str,
    expected_num_envs: int,
    expected_hw: tuple[int, int],
) -> dict[str, Any]:
    sensors = state.get("sensors")
    if not isinstance(sensors, Mapping) or name not in sensors:
        raise BaselineEvidenceError(f"{phase} snapshot is missing required sensor {name!r}.")
    sensor = sensors[name]
    if not isinstance(sensor, Mapping):
        raise BaselineEvidenceError(f"{phase}.{name} must be a sensor record.")
    if sensor.get("class") != "MultiMeshRayCasterCamera":
        raise BaselineEvidenceError(f"{phase}.{name} is not a MultiMeshRayCasterCamera: {sensor.get('class')!r}.")

    observation = sensor.get("observation")
    if not isinstance(observation, Mapping):
        raise BaselineEvidenceError(f"{phase}.{name} is missing observation statistics.")
    expected_shape = [expected_num_envs, *expected_hw]
    if observation.get("shape") != expected_shape:
        raise BaselineEvidenceError(
            f"{phase}.{name} observation shape {observation.get('shape')!r} != {expected_shape}."
        )
    finite_fraction = _number(observation.get("finite_fraction"), f"{phase}.{name}.observation.finite_fraction")
    valid_fraction = _number(observation.get("valid_fraction"), f"{phase}.{name}.observation.valid_fraction")
    if not 0.0 < finite_fraction <= 1.0:
        raise BaselineEvidenceError(f"{phase}.{name} has no finite observation pixels ({finite_fraction}).")
    if not 0.0 < valid_fraction <= finite_fraction:
        raise BaselineEvidenceError(f"{phase}.{name} has no valid in-range pixels ({valid_fraction}).")

    limits = state.get("range_limits_m")
    if not isinstance(limits, Sequence) or isinstance(limits, (str, bytes)) or len(limits) != 2:
        raise BaselineEvidenceError(f"{phase} snapshot has invalid range_limits_m={limits!r}.")
    min_limit = _number(limits[0], f"{phase}.range_limits_m[0]")
    max_limit = _number(limits[1], f"{phase}.range_limits_m[1]")
    if not 0.0 < min_limit < max_limit:
        raise BaselineEvidenceError(f"{phase} snapshot has invalid range limits {limits!r}.")
    min_range = _number(observation.get("min_m"), f"{phase}.{name}.observation.min_m")
    max_range = _number(observation.get("max_m"), f"{phase}.{name}.observation.max_m")
    if min_range < min_limit or max_range > max_limit or min_range > max_range:
        raise BaselineEvidenceError(
            f"{phase}.{name} valid range [{min_range}, {max_range}] is outside [{min_limit}, {max_limit}]."
        )

    frames = _frames(sensor.get("frame"), f"{phase}.{name}.frame", expected_num_envs)
    consumed = _frames(sensor.get("consumed_frame"), f"{phase}.{name}.consumed_frame", expected_num_envs)
    if any(frame < 1 for frame in frames):
        raise BaselineEvidenceError(f"{phase}.{name} did not produce a live frame: {frames}.")
    if consumed != frames:
        raise BaselineEvidenceError(
            f"{phase}.{name} frame was not consumed exactly once: frame={frames}, consumed={consumed}."
        )
    return {
        "frame": frames,
        "finite_fraction": finite_fraction,
        "valid_fraction": valid_fraction,
        "min_m": min_range,
        "max_m": max_range,
    }


def validate_scripted_tof_runtime(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
    *,
    expected_num_envs: int,
    expected_hw: tuple[int, int],
    max_command_translation_m: Sequence[float],
    max_tool_translation_m: Sequence[float],
    min_motion_m: float = 1.0e-4,
) -> dict[str, Any]:
    """Validate and summarize the live scripted-ToF runtime evidence.

    Every environment must receive a meaningful translation command and show a
    measured tool response.  Both 8x8 sensors must provide finite, valid ranges
    and advance their live frame counters after the warm-up snapshot.
    """
    if expected_num_envs < 1:
        raise BaselineEvidenceError("expected_num_envs must be positive.")
    if tuple(expected_hw) != (8, 8):
        raise BaselineEvidenceError(f"The hardware ToF gate requires 8x8 data, got {expected_hw}.")
    threshold = _number(min_motion_m, "min_motion_m")
    if threshold <= 0.0:
        raise BaselineEvidenceError("min_motion_m must be positive.")

    for phase, state in (("before", before), ("after", after)):
        if not isinstance(state, Mapping):
            raise BaselineEvidenceError(f"{phase} snapshot must be a mapping.")
        if state.get("source") != LIVE_TOF_SOURCE:
            raise BaselineEvidenceError(f"{phase} ToF source {state.get('source')!r} is not live ray-caster data.")

    commands = _motion_values(max_command_translation_m, "max_command_translation_m", expected_num_envs)
    responses = _motion_values(max_tool_translation_m, "max_tool_translation_m", expected_num_envs)
    if any(value <= threshold for value in commands):
        raise BaselineEvidenceError(
            f"Scripted controller did not issue a meaningful command in every environment: {commands}."
        )
    if any(value <= threshold for value in responses):
        raise BaselineEvidenceError(f"Tool did not show a measurable response in every environment: {responses}.")

    summary: dict[str, Any] = {
        "source": LIVE_TOF_SOURCE,
        "min_motion_gate_m": threshold,
        "max_command_translation_m": commands,
        "max_tool_translation_m": responses,
        "sensors": {},
    }
    for name in REQUIRED_TOF_SENSORS:
        sensor_before = _snapshot_sensor(
            before,
            phase="before",
            name=name,
            expected_num_envs=expected_num_envs,
            expected_hw=expected_hw,
        )
        sensor_after = _snapshot_sensor(
            after,
            phase="after",
            name=name,
            expected_num_envs=expected_num_envs,
            expected_hw=expected_hw,
        )
        frame_advance = [new - old for old, new in zip(sensor_before["frame"], sensor_after["frame"], strict=True)]
        if any(delta <= 0 for delta in frame_advance):
            raise BaselineEvidenceError(f"{name} frame did not advance in every environment: {frame_advance}.")
        summary["sensors"][name] = {
            "before": sensor_before,
            "after": sensor_after,
            "frame_advance": frame_advance,
        }
    return summary
