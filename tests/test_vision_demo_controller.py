"""CPU contracts for the live camera → decision → next physics-step loop."""

from __future__ import annotations

import json
from collections import deque

import numpy as np
import pytest

from isaaclab_pruning.sim.vision_demo_controller import VisionPruningDemo

POSE = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
MOUTH = np.array([1.0, 2.0, 3.07])


def test_demo_uses_more_initial_corners_without_relaxing_tracking_gates():
    controller = VisionPruningDemo("branch_7", (0, 0, 1), 0.004, POSE)
    config = controller.tracker.config
    assert config.feature_quality_level == 0.005
    assert config.replenish_features is True
    assert config.roi_half_size_px == (14, 24)
    assert config.min_features == 4
    assert config.max_roundtrip_error_px == 1.0
    assert config.max_flow_residual_px == 3.0
    assert config.min_patch_correlation == 0.35
    assert config.max_depth_spread_m == 0.025
    assert config.depth_radius_px == 1


def tracking(target):
    return {"state": "tracking", "target_position_world_m": list(target), "confidence": 1.0}


class StubTracker:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.initializations = []
        self.updates = []

    def initialize(self, rgb, pixel, depth):
        self.initializations.append((rgb, pixel, depth))
        return {"state": "initialized", "target_position_world_m": None}

    def update(self, rgb, depth, matrix, transform):
        self.updates.append((rgb, depth, matrix, transform))
        return self.responses.popleft()


def demo(*responses, **kwargs):
    options = dict(target_id="branch_7", branch_axis_w=(0, 0, 1), branch_radius_m=0.004, home_pose_wxyz=POSE)
    options.update(kwargs)
    controller = VisionPruningDemo(**options)
    controller.tracker = StubTracker(responses)
    return controller


def observe(controller, time_s, pose=POSE, **kwargs):
    return controller.observe("live rgb", "live optical-z", np.eye(3), np.eye(4), time_s, pose, **kwargs)


def test_uninitialized_controller_holds_measured_pose_not_home():
    controller = demo()
    measured = POSE.copy()
    measured[:3] += [0.1, -0.2, 0.3]
    command, phase, decision = controller.command(measured, 0.0)
    np.testing.assert_array_equal(command, measured)
    assert phase == "observe"
    assert decision["reason"] == "no_camera_observation"
    assert controller.vision_command_count == 0


def test_seed_pixel_is_used_once_and_does_not_authorize_motion():
    controller = demo(tracking(MOUTH + [0.1, 0, 0]))
    seed = [120, 80]
    initialized = controller.initialize("seed rgb", "seed depth", seed)
    assert initialized["state"] == "initialized"
    assert controller.command(POSE, 0.0)[1] == "observe"
    observe(controller, 0.0)
    assert controller.command(POSE, 0.0)[1] == "vision_approach"
    assert len(controller.tracker.initializations) == 1
    assert controller.tracker.initializations[0][1] == seed
    assert controller.tracker.updates[0][:2] == ("live rgb", "live optical-z")


def test_next_command_uses_latest_camera_world_target_and_keeps_measured_orientation():
    controller = demo(tracking(MOUTH + [0.1, 0, 0]), tracking(MOUTH + [0, 0.1, 0]))
    observe(controller, 0.0)
    command, phase, decision = controller.command(POSE, 0.1)
    np.testing.assert_allclose(command[:3] - POSE[:3], [0.004, 0, 0], atol=1e-12)
    np.testing.assert_array_equal(command[3:], POSE[3:])
    assert phase == "vision_approach" and decision["state"] == "tracking"
    assert controller.vision_command_request_count == 1
    assert controller.vision_command_count == 0
    controller.command_applied(phase, decision)
    assert controller.vision_command_count == 1
    observe(controller, 0.1)
    command, phase, decision = controller.command(POSE, 0.2)
    np.testing.assert_allclose(command[:3] - POSE[:3], [0, 0.004, 0], atol=1e-12)
    assert controller.vision_command_request_count == 2
    assert controller.vision_command_count == 1
    controller.command_applied(phase, decision)
    assert controller.vision_command_count == 2


def test_external_gate_overridden_request_is_not_counted_as_applied_motion():
    controller = demo(tracking(MOUTH + [0.1, 0, 0]))
    observe(controller, 0.0)
    _, phase, decision = controller.command(POSE, 0.0)
    assert phase == "vision_approach" and decision["state"] == "tracking"
    controller.stop("tof_minimum_clearance")
    controller.command_applied("stopped_failure", {"state": "hold", "reason": "tof_minimum_clearance"})
    evidence = controller.evidence()
    assert evidence["vision_command_request_count"] == 1
    assert evidence["vision_command_count"] == 0


@pytest.mark.parametrize(
    "phase,decision",
    [("retreat", {"state": "retreat"}), ("vision_approach", {"state": "hold"}), ("vision_approach", None)],
)
def test_applied_counter_excludes_holds_retreat_and_missing_decisions(phase, decision):
    controller = demo()
    controller.command_applied(phase, decision)
    assert controller.vision_command_count == 0


def test_visual_servo_uses_rotated_mouth_offset_instead_of_tool_origin():
    rotated = POSE.copy()
    rotated[3:] = [2**-0.5, 0, 2**-0.5, 0]
    mouth = rotated[:3] + [0.07, 0, 0]
    controller = demo(tracking(mouth + [0, 0.1, 0]), branch_axis_w=(1, 0, 0))
    observe(controller, 0.0, rotated)
    command, phase, _ = controller.command(rotated, 0.1)
    assert phase == "vision_approach"
    np.testing.assert_allclose(command[:3] - rotated[:3], [0, 0.004, 0], atol=1e-12)
    np.testing.assert_array_equal(command[3:], rotated[3:])


@pytest.mark.parametrize("state", ["tracking_lost", "invalid_depth", "invalid_calibration"])
def test_vision_loss_latches_hold_without_scene_target_or_home_fallback(state):
    controller = demo(
        tracking(MOUTH + [0.1, 0, 0]),
        {"state": state, "target_position_world_m": None},
        tracking(MOUTH + [0, 0.1, 0]),
    )
    observe(controller, 0.0)
    assert controller.command(POSE, 0.0)[1] == "vision_approach"
    moved = POSE.copy()
    moved[:3] += [0.003, 0, 0]
    evidence = observe(controller, 0.1, moved)
    assert evidence["cut"]["stopped_reason"] == "vision_invalid"
    command, phase, _ = controller.command(moved, 0.1)
    np.testing.assert_array_equal(command, moved)
    assert phase == "stopped_failure"
    assert not controller.cut_step.detached
    # Even a later valid measurement cannot bypass the latched cut stop.
    observe(controller, 0.2, moved)
    command, phase, _ = controller.command(moved, 0.2)
    np.testing.assert_array_equal(command, moved)
    assert phase == "stopped_failure"
    assert controller.tracker.initializations == []
    json.dumps(evidence, allow_nan=False)


def test_invalid_measurement_state_cannot_use_leftover_target_payload():
    controller = demo({"state": "invalid_depth", "target_position_world_m": list(MOUTH + [0.1, 0, 0])})
    observe(controller, 0.0)
    command, phase, _ = controller.command(POSE, 0.1)
    np.testing.assert_array_equal(command, POSE)
    assert phase == "stopped_failure"
    assert controller.vision_command_count == 0


@pytest.mark.parametrize("command_time", [-0.1, 0.26, float("nan")])
def test_unobserved_time_cannot_reuse_stale_or_future_camera_target(command_time):
    controller = demo(tracking(MOUTH + [0.1, 0, 0]))
    observe(controller, 0.0)
    command, phase, decision = controller.command(POSE, command_time)
    np.testing.assert_array_equal(command, POSE)
    assert phase == "vision_hold"
    assert decision["reason"] == "vision_stale_or_future"
    assert controller.vision_command_count == 0


@pytest.mark.parametrize("time_s", [0.0, -0.1])
def test_reused_or_regressing_camera_timestamp_is_rejected(time_s):
    controller = demo(tracking(MOUTH), tracking(MOUTH))
    observe(controller, 0.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        observe(controller, time_s)
    assert not controller.cut_step.detached


def test_closure_holds_and_retreat_only_starts_after_verified_detachment():
    controller = demo(*(tracking(MOUTH) for _ in range(6)))
    for time_s in [0.0, 0.1, 0.2, 0.3]:
        observe(controller, time_s)
        command, phase, _ = controller.command(POSE, time_s)
        np.testing.assert_array_equal(command, POSE)
        assert phase in {"align", "simulated_closure"}
    assert controller.cut_step.phase == "closing"
    observe(controller, 0.6)
    assert not controller.cut_step.detach_event
    assert controller.command(POSE, 0.6)[1] == "simulated_closure"
    observe(controller, 0.91)
    assert controller.cut_step.detach_event and controller.cut_step.detached
    displaced = POSE.copy()
    displaced[:3] += [0.01, 0, 0]
    command, phase, decision = controller.command(displaced, 0.91)
    assert phase == "retreat" and decision["state"] == "retreat"
    np.testing.assert_allclose(command[:3] - displaced[:3], [-0.004, 0, 0], atol=1e-12)
    assert controller.command(POSE, 1.0)[1] == "complete"


def test_no_new_camera_frames_cannot_advance_closure_by_elapsed_command_time():
    controller = demo(*(tracking(MOUTH) for _ in range(4)))
    for time_s in [0.0, 0.1, 0.2, 0.3]:
        observe(controller, time_s)
    command, phase, _ = controller.command(POSE, 10.0)
    np.testing.assert_array_equal(command, POSE)
    assert phase == "simulated_closure"
    assert not controller.cut_step.detached and not controller.cut_step.detach_event
    assert controller.cut_step.closure_progress == 0.0


def test_hazard_stops_valid_vision_before_any_approach_command():
    controller = demo(tracking(MOUTH + [0.1, 0, 0]))
    observe(controller, 0.0, hazard_contact=True)
    command, phase, _ = controller.command(POSE, 0.1)
    np.testing.assert_array_equal(command, POSE)
    assert phase == "stopped_failure"
    assert controller.cut_step.stopped_reason == "hazard_contact"


def test_hazard_after_detachment_stops_retreat():
    controller = demo(*(tracking(MOUTH) for _ in range(6)))
    for time_s in [0.0, 0.1, 0.2, 0.3, 0.91]:
        observe(controller, time_s)
    assert controller.cut_step.detached
    observe(controller, 1.0, hazard_contact=True)
    displaced = POSE.copy()
    displaced[:3] += [0.01, 0, 0]
    command, phase, _ = controller.command(displaced, 1.0)
    np.testing.assert_array_equal(command, displaced)
    assert phase == "stopped_failure"


def test_evidence_discloses_ground_truth_depth_metadata_and_surrogate_cut():
    controller = demo(tracking(MOUTH + [0.1, 0, 0]))
    evidence = observe(controller, 0.0)
    assert "metadata" in evidence["target_identity_axis_radius_source"]
    assert "ground truth" in evidence["depth_source"]
    assert "surrogate" in evidence["cut_model"]
    json.dumps(evidence, allow_nan=False)


def test_external_tof_stop_latches_without_fabricating_contact_or_detachment():
    controller = demo(*(tracking(MOUTH) for _ in range(6)))
    for time_s in [0.0, 0.1, 0.2, 0.3]:
        observe(controller, time_s)
    assert controller.cut_step.phase == "closing"
    controller.stop("tof_minimum_clearance")
    displaced = POSE.copy()
    displaced[:3] += [0.01, 0, 0]
    command, phase, decision = controller.command(displaced, 0.4)
    np.testing.assert_array_equal(command, displaced)
    assert phase == "stopped_failure" and decision["reason"] == "tof_minimum_clearance"
    evidence = observe(controller, 1.0)
    assert not evidence["cut"]["detached"]
    assert evidence["cut"]["stopped_reason"] == "tof_minimum_clearance"
    assert "hazard_contact" not in evidence["cut"]["certificate"]["reasons"]
    controller.stop("later_stop")
    assert controller.external_stop_reason == "tof_minimum_clearance"
