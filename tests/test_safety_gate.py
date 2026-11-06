"""The Python reference the C++ safety gate is checked against: its rules, and that the two agree on limits."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from isaaclab_pruning.sim.vision_demo_controller import DEMO_MOUTH_OFFSET_TOOL_M, demo_cut_config
from isaaclab_pruning.task.safety_gate import (
    MINIMUM_CLEARANCE_M,
    GateCycle,
    SafetyGateReference,
    sensor_guard,
)
from isaaclab_pruning.task.simulated_cut import CutObservation

ROOT = Path(__file__).resolve().parents[1]
TARGET = "tree0_SPUR_component_530"


def observation(**changes):
    values = dict(
        time_s=5.0,
        vision_timestamp_s=5.0,
        target_id=TARGET,
        target_semantic="branch",
        vision_valid=True,
        target_position_w=(0.25, 0.63, 0.81),
        target_axis_w=(0.0, 0.0, 1.0),
        target_radius_m=0.005,
        mouth_position_w=(0.252, 0.63, 0.81),
        cutter_closing_axis_w=(1.0, 0.0, 0.0),
    )
    values.update(changes)
    return CutObservation(**values)


def cycle(ranges=0.3, **changes):
    kwargs = {k: changes.pop(k) for k in ("prior_cut_phase", "prior_detached", "operator_hold") if k in changes}
    grid = np.full(128, ranges, dtype=float)
    return GateCycle(tof_ranges=grid, tof_valid=np.isfinite(grid), observation=observation(**changes), **kwargs)


def test_sensor_guard_is_the_recorders():
    runner_source = (ROOT / "hpc/inner/render_pruning_workflow.py").read_text()
    assert "from isaaclab_pruning.task.safety_gate import sensor_guard" in runner_source
    assert sensor_guard([0.04, 0.3], [True, True], 0) == ("tof_minimum_clearance", 0)
    assert sensor_guard([np.inf, np.nan], [False, False], 3) == ("both_tof_missing_4_frames", 4)


def test_tof_stops_before_the_cut_check_and_latches():
    gate = SafetyGateReference(demo_cut_config(TARGET))
    assert gate.step(cycle()).passed
    grid = np.full(128, 0.3)
    grid[5] = 0.05
    verdict = gate.step(
        GateCycle(tof_ranges=grid, tof_valid=np.isfinite(grid), observation=observation(vision_valid=False))
    )
    assert (verdict.passed, verdict.reason) == (False, "tof_minimum_clearance")
    later = gate.step(cycle())
    assert not later.passed and later.latched_reason == "tof_minimum_clearance"


def test_four_cycles_without_a_usable_zone_stop():
    gate = SafetyGateReference(demo_cut_config(TARGET))
    reasons = [gate.step(cycle(ranges=np.nan)).reason for _ in range(4)]
    assert reasons == [None, None, None, "both_tof_missing_4_frames"]


def test_after_detachment_only_contact_stops():
    gate = SafetyGateReference(demo_cut_config(TARGET))
    assert gate.step(cycle(prior_detached=True, vision_valid=False)).passed
    assert gate.step(cycle(prior_detached=True, hazard_contact=True)).reason == "hazard_contact"


def test_losing_readiness_while_closing_stops():
    verdict = SafetyGateReference(demo_cut_config(TARGET)).step(
        cycle(prior_cut_phase="closing", mouth_position_w=(0.27, 0.63, 0.81))
    )
    assert verdict.reason == "gate_lost_during_closure" and not verdict.closure_authorized


@pytest.mark.parametrize(
    "change,reason",
    [
        ({"target_id": "tree1_SPUR_component_19444"}, "wrong_target_id"),
        ({"vision_timestamp_s": 4.7}, "vision_stale_or_future"),
        ({"target_radius_m": 0.02}, "branch_exceeds_demo_cut_limit"),
        ({"target_position_w": (np.nan,) * 3}, "invalid_geometry"),
    ],
)
def test_cut_check_failures_hold_with_their_first_reason(change, reason):
    assert SafetyGateReference(demo_cut_config(TARGET)).step(cycle(**change)).reason == reason


def test_operator_hold_wins():
    assert SafetyGateReference(demo_cut_config(TARGET)).step(cycle(operator_hold=True)).reason == "operator_hold"


def test_cpp_defaults_match_the_demo_limits():
    """The C++ CutConfig and node defaults must be the limits the recorded runs used."""
    config = demo_cut_config(TARGET)
    header = (ROOT / "ros2/pruning_sil_cpp/include/pruning_sil_cpp/safety_gates.hpp").read_text()
    node = (ROOT / "ros2/pruning_sil_cpp/src/safety_gate_node.cpp").read_text()

    def default(name, text):
        match = re.search(rf"{name}\{{([0-9.]+)\}}", text) or re.search(rf'"{name}", ([0-9.]+)\)', text)
        assert match, name
        return float(match.group(1))

    assert default("max_branch_radius_m", header) == config.max_branch_radius_m
    assert default("mouth_position_tolerance_m", header) == config.mouth_position_tolerance_m
    assert default("alignment_tolerance_deg", header) == config.alignment_tolerance_deg
    assert default("vision_max_age_s", header) == config.vision_max_age_s
    assert "kMinimumClearanceM = 0.06" in header and MINIMUM_CLEARANCE_M == 0.06
    assert default("max_branch_radius_m", node) == config.max_branch_radius_m
    assert '"mouth_offset_tool", std::vector<double>{0.0, 0.0, 0.07}' in node
    assert DEMO_MOUTH_OFFSET_TOOL_M == (0.0, 0.0, 0.07)


def test_parity_fixture_is_present_and_has_holds():
    fixture = (ROOT / "ros2/pruning_sil_cpp/test/safety_gate_fixture.txt").read_text().splitlines()
    assert fixture[0] == "pruning_safety_gate_fixture 1"
    expects = [line.split() for line in fixture if line.startswith("expect ")]
    assert len(expects) >= 50
    assert {line[2] for line in expects if line[1] == "0"} >= {"tof_minimum_clearance", "vision_invalid"}
