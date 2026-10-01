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
    def __init__(self, responses, config):
        self.config = config
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
    controller.tracker = StubTracker(responses, controller.tracker.config)
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


def test_straight_baseline_is_unchanged_and_strategies_leave_every_gate_identical():
    from isaaclab_pruning.sim.vision_demo_controller import ApproachStrategy

    baseline = VisionPruningDemo("branch_7", (0, 0, 1), 0.004, POSE)
    assert baseline.approach == ApproachStrategy() and baseline.approach_phase == "straight"
    for strategy in (
        ApproachStrategy("tool_axis_standoff", 0.06),
        ApproachStrategy("horizontal_standoff", 0.08),
        ApproachStrategy(max_step_m=0.002),
    ):
        variant = VisionPruningDemo("branch_7", (0, 0, 1), 0.004, POSE, approach=strategy)
        assert variant.tracker.config == baseline.tracker.config
        assert variant.cutter.config == baseline.cutter.config
        assert variant.max_step == strategy.max_step_m
    for bad in (("straight", 0.05), ("tool_axis_standoff", 0.0), ("sideways", 0.05)):
        with pytest.raises(ValueError):
            ApproachStrategy(*bad)
    with pytest.raises(ValueError):
        ApproachStrategy(max_step_m=0.02)


def test_tool_axis_standoff_freezes_the_axis_at_first_tracking_then_finishes_along_it():
    from isaaclab_pruning.sim.vision_demo_controller import ApproachStrategy

    # Tool +z is world +z at POSE (identity rotation); mouth is 0.07 above the tool origin.
    target = MOUTH + [0.0, 0.0, 0.10]
    controller = demo(*[tracking(target)] * 40, approach=ApproachStrategy("tool_axis_standoff", 0.06))
    _, phase, _ = controller.command(POSE, 0.05)
    assert phase == "observe" and controller.approach_axis_w is None  # nothing frozen before tracking
    pose = POSE.copy()
    t = 0.1
    observe(controller, t)
    _, phase, decision = controller.command(pose, t + 0.05)
    assert phase == "vision_approach" and decision["approach_phase"] == "standoff"
    np.testing.assert_allclose(controller.approach_axis_w, [0, 0, 1], atol=1e-12)
    # Standoff point is 0.04 above the mouth: ten 4 mm steps reach it, then the final phase begins.
    phases = []
    for step in range(14):
        t += 0.1
        observe(controller, t)
        command, phase, decision = controller.command(pose, t + 0.05)
        phases.append(decision["approach_phase"])
        pose = command
    assert "standoff" in phases and "final" in phases and phases.index("final") >= 9
    assert controller.evidence()["approach_strategy"] == {
        "mode": "tool_axis_standoff",
        "standoff_m": 0.06,
        "max_step_m": 0.004,
        "max_rotation_deg": 1.5,
        "planned_tool_quat_wxyz": None,
    }
    # In the final phase the mouth keeps moving along the same axis toward the target itself.
    assert pose[2] > POSE[2] + 0.04 and np.allclose(pose[:2], POSE[:2])


def test_horizontal_standoff_axis_has_no_vertical_component():
    from isaaclab_pruning.sim.vision_demo_controller import ApproachStrategy

    target = MOUTH + [0.3, 0.4, 0.2]
    controller = demo(tracking(target), tracking(target), approach=ApproachStrategy("horizontal_standoff", 0.08))
    observe(controller, 0.0)
    controller.command(POSE, 0.05)
    np.testing.assert_allclose(controller.approach_axis_w, [0.6, 0.8, 0.0], atol=1e-12)


def test_quaternion_helpers_rotate_at_a_bounded_rate_and_build_a_perpendicular_closing_axis():
    from isaaclab_pruning.sim.vision_demo_controller import (
        closing_axis_tool_at,
        quat_angle_deg,
        quat_to_matrix,
        rotate_toward,
    )

    identity = np.array([1.0, 0.0, 0.0, 0.0])
    quarter = np.array([np.cos(np.pi / 4), 0.0, np.sin(np.pi / 4), 0.0])  # 90 deg about y
    assert quat_angle_deg(identity, quarter) == pytest.approx(90.0)
    step = rotate_toward(identity, quarter, 1.5)
    assert quat_angle_deg(identity, step) == pytest.approx(1.5, abs=1e-6)
    assert np.allclose(rotate_toward(identity, quarter, 120.0), quarter)
    np.testing.assert_allclose(quat_to_matrix(quarter) @ [0, 0, 1], [1, 0, 0], atol=1e-12)
    closing = closing_axis_tool_at(identity, (0.0, 1.0, 0.0))
    np.testing.assert_allclose(closing, [-1.0, 0.0, 0.0], atol=1e-12)  # z cross y
    with pytest.raises(ValueError):
        closing_axis_tool_at(identity, (0.0, 0.0, 1.0))


def test_planned_strategy_validates_its_orientation():
    from isaaclab_pruning.sim.vision_demo_controller import ApproachStrategy

    good = ApproachStrategy("planned_pose_standoff", 0.06, planned_tool_quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    assert good.planned_tool_quat_wxyz == (1.0, 0.0, 0.0, 0.0) and good.max_rotation_deg == 1.5
    with pytest.raises(ValueError):
        ApproachStrategy("planned_pose_standoff", 0.06, planned_tool_quat_wxyz=(2.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        ApproachStrategy("tool_axis_standoff", 0.06, planned_tool_quat_wxyz=(1.0, 0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        ApproachStrategy("planned_pose_standoff", 0.06, max_rotation_deg=10.0)


def test_planned_approach_rotates_moves_to_a_standoff_from_the_tracked_target_then_finishes_and_retreats():
    from isaaclab_pruning.sim.vision_demo_controller import ApproachStrategy, quat_angle_deg, quat_to_matrix

    # Final orientation: 20 degrees about x from home; its tool axis is where the mouth must approach from.
    half = np.radians(20.0) / 2
    planned = (float(np.cos(half)), float(np.sin(half)), 0.0, 0.0)
    axis = quat_to_matrix(planned)[:, 2]
    target = MOUTH + 0.15 * axis + np.array([0.0, 0.0, 0.0])
    strategy = ApproachStrategy("planned_pose_standoff", 0.06, planned_tool_quat_wxyz=planned)
    controller = demo(*[tracking(target)] * 400, approach=strategy)
    pose, t, phases, rotations, steps = POSE.copy(), 0.0, [], [], []
    for _ in range(120):
        observe(controller, t, pose)
        command, phase, decision = controller.command(pose, t + 0.05)
        if phase == "align":  # the unchanged cut gate takes over inside its 8 mm mouth tolerance
            break
        assert phase == "vision_approach", decision
        rotations.append(quat_angle_deg(pose[3:], command[3:]))
        steps.append(float(np.linalg.norm(decision["mouth_delta_world_m"])))
        phases.append(decision["approach_phase"])
        pose, t = command, t + 0.1
    assert phase == "align"
    assert max(rotations) <= 1.5 + 1e-9 and max(steps) <= 0.004 + 1e-9
    assert "standoff" in phases and "final" in phases
    # The standoff point was frozen from the tracked target, 60 mm back along the planned axis.
    np.testing.assert_allclose(controller.standoff_point_w, target - 0.06 * axis, atol=1e-9)
    assert quat_angle_deg(pose[3:], planned) < 1e-6
    assert np.linalg.norm(controller._mouth(pose[:3], pose[3:]) - target) <= 0.008
    # After detachment the retreat goes back out to the standoff point, then home, rotating back.
    controller.cut_step = type("Step", (), {"detached": True, "phase": "retreat"})()
    retreat_phases = []
    for _ in range(200):
        command, phase, decision = controller.command(pose, t)
        retreat_phases.append(decision["retreat_phase"])
        assert quat_angle_deg(pose[3:], command[3:]) <= 1.5 + 1e-9
        pose = command
        if phase == "complete":
            break
    assert retreat_phases[0] == "to_standoff" and retreat_phases[-1] == "to_home"
    np.testing.assert_allclose(pose[:3], POSE[:3], atol=0.0031)
    assert quat_angle_deg(pose[3:], POSE[3:]) <= 0.5


def test_planned_approach_holds_without_valid_tracking_and_identity_plan_uses_the_home_orientation():
    from isaaclab_pruning.sim.vision_demo_controller import ApproachStrategy

    controller = demo(
        tracking(MOUTH + [0, 0, 0.1]),
        tracking(MOUTH + [0, 0, 0.1]),
        approach=ApproachStrategy("planned_pose_standoff", 0.06),
    )
    observe(controller, 0.0)
    # A stale measurement holds exactly as the baseline does, and freezes nothing.
    command, phase, decision = controller.command(POSE, 1.0)
    assert phase == "vision_hold" and np.array_equal(command, POSE) and controller.standoff_point_w is None
    observe(controller, 0.1)
    command, phase, _ = controller.command(POSE, 0.15)
    assert phase == "vision_approach" and np.allclose(command[3:], POSE[3:])
    assert controller.evidence()["planned_tool_quat_wxyz"] == pytest.approx(list(POSE[3:]))


# ---------------------------------------------------------------------------------------------------------------
# Labelled arms: jaw self-mask and closure hold
# ---------------------------------------------------------------------------------------------------------------
UNCHANGED_EVIDENCE_KEYS = {
    "tracker_config",
    "measurement",
    "measurement_time_s",
    "cut",
    "observations",
    "vision_command_count",
    "vision_command_request_count",
    "external_stop_reason",
    "approach_strategy",
    "approach_axis_w",
    "approach_phase",
    "planned_tool_quat_wxyz",
    "standoff_point_w",
    "retreat_phase",
    "mouth_offset_tool_m",
    "closing_axis_tool",
    "target_identity_axis_radius_source",
    "depth_source",
    "cut_model",
}
SHAPE = (320, 480)
CAMERA = np.array([[320.0, 0.0, 240.0], [0.0, 320.0, 160.0], [0.0, 0.0, 1.0]])


def jaw_camera(pose=POSE):
    """Optical axes along world axes, 12 cm in front of the jaw centre (tool-local z 0.07) and looking at it."""
    transform = np.eye(4)
    transform[:3, 3] = np.asarray(pose[:3]) + [0.0, 0.0, -0.05]
    return transform


def occluded():
    return {"state": "jaw_mask_occluded", "reason": "too_few_unmasked_patch_pixels", "target_position_world_m": None}


def latched():
    return {"state": "tracking_lost", "reason": "explicit_initialization_required", "target_position_world_m": None}


def lost(reason, **telemetry):
    return {"state": "tracking_lost", "reason": reason, "target_position_world_m": None, **telemetry}


class MaskRecordingTracker(StubTracker):
    def __init__(self, responses, config):
        super().__init__(responses, config)
        self.masks = []
        self.initial_masks = []

    def initialize(self, rgb, pixel, depth, exclusion_mask=None):
        self.initial_masks.append(exclusion_mask)
        return {"state": "initialized", "target_position_world_m": None}

    def update(self, rgb, depth, matrix, transform, exclusion_mask=None):
        self.masks.append(exclusion_mask)
        return self.responses.popleft()


def masked_demo(*responses, **kwargs):
    options = dict(target_id="branch_7", branch_axis_w=(0, 0, 1), branch_radius_m=0.004, home_pose_wxyz=POSE)
    options.update(kwargs)
    controller = VisionPruningDemo(**options)
    controller.tracker = MaskRecordingTracker(responses, controller.tracker.config)
    return controller


def observe_frame(controller, time_s, pose=POSE, **kwargs):
    rgb = np.zeros((*SHAPE, 3), dtype=np.uint8)
    depth = np.zeros(SHAPE, dtype=np.float32)
    return controller.observe(rgb, depth, CAMERA, jaw_camera(pose), time_s, pose, **kwargs)


def test_flags_off_is_the_unchanged_controller_with_unchanged_calls_and_evidence():
    responses = [tracking(MOUTH), tracking(MOUTH), occluded(), latched(), tracking(MOUTH)]
    default = demo(*responses)
    explicit = demo(*responses, jaw_self_mask=False, closure_hold=False)
    assert explicit.cutter_without_hold is None and explicit.jaw_roll_rad is None
    assert default.initialize("seed rgb", "seed depth", [3, 4]) == explicit.initialize("seed rgb", "seed depth", [3, 4])
    for time_s in (0.0, 0.1, 0.2, 0.3, 0.4):
        first, second = observe(default, time_s), observe(explicit, time_s)
        assert first == second
        assert set(first) == UNCHANGED_EVIDENCE_KEYS
        assert first["cut"]["certificate"]["vision_source"] == "live"
        assert first["cut"]["certificate"]["waived_checks"] == ()
        assert first["cut"]["certificate"]["held_target_age_s"] is None
    # StubTracker's 4-argument update would raise on any extra argument; the loss stopped the cut as before.
    assert all(len(update) == 4 for update in explicit.tracker.updates)
    assert explicit.cut_step.stopped_reason == "vision_invalid"


def test_arms_must_be_booleans():
    for bad in ({"jaw_self_mask": 1}, {"closure_hold": "yes"}):
        with pytest.raises(ValueError, match="bool"):
            VisionPruningDemo("branch_7", (0, 0, 1), 0.004, POSE, **bad)


def test_jaw_mask_uses_the_rendered_pre_update_closure_progress():
    from isaaclab_pruning.perception.jaw_self_mask import jaw_boxes, jaw_mask

    controller = masked_demo(*[tracking(MOUTH)] * 8, jaw_self_mask=True)
    progresses = []
    for time_s in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        rendered = controller.cut_step.closure_progress if controller.cut_step else 0.0
        evidence = observe_frame(controller, time_s)
        mask = controller.tracker.masks[-1]
        assert mask.dtype == bool and mask.shape == SHAPE
        expected = jaw_mask(jaw_boxes(POSE, 0.0, rendered, 0.004), CAMERA, jaw_camera(), SHAPE)
        np.testing.assert_array_equal(mask, expected)
        record = evidence["jaw_self_mask"]
        assert record["closure_progress_used"] == rendered
        assert record["gap_m"] == pytest.approx((1 - rendered) * 0.032 + rendered * 0.008)
        assert record["mask_pixel_count"] == int(expected.sum()) and len(record["corners_px"]) == 16
        advanced = controller.cut_step.closure_progress
        if advanced != rendered:
            # The progress the cutter reaches during this observation is NOT what this frame was rendered with.
            assert not np.array_equal(
                mask, jaw_mask(jaw_boxes(POSE, 0.0, advanced, 0.004), CAMERA, jaw_camera(), SHAPE)
            )
        progresses.append(rendered)
        json.dumps(evidence, allow_nan=False)
    assert progresses[:5] == [0.0] * 5 and progresses[5] > 0 and progresses == sorted(progresses)


def test_jaw_initialization_builds_the_open_preview_mask_and_refuses_without_camera_or_pose():
    from isaaclab_pruning.perception.jaw_self_mask import jaw_boxes, jaw_mask

    controller = masked_demo(jaw_self_mask=True)
    rgb, depth = np.zeros((*SHAPE, 3), dtype=np.uint8), np.zeros(SHAPE, dtype=np.float32)
    with pytest.raises(ValueError, match="jaw_self_mask"):
        controller.initialize(rgb, depth, [240, 160])
    with pytest.raises(ValueError, match="jaw_self_mask"):
        controller.initialize(rgb, depth, [240, 160], camera_matrix=CAMERA, world_from_optical=jaw_camera())
    result = controller.initialize(
        rgb, depth, [240, 160], camera_matrix=CAMERA, world_from_optical=jaw_camera(), tool_pose_wxyz=POSE
    )
    expected = jaw_mask(jaw_boxes(POSE, 0.0, 0.0, 0.004), CAMERA, jaw_camera(), SHAPE)
    np.testing.assert_array_equal(controller.tracker.initial_masks[-1], expected)
    assert result["jaw_self_mask"]["closure_progress_used"] == 0.0 and result["jaw_self_mask"]["gap_m"] == 0.032


def _into_closing(controller, observe_fn=observe):
    for time_s in (0.0, 0.1, 0.2, 0.3):
        observe_fn(controller, time_s)
    assert controller.cut_step.phase == "closing"


def test_closure_hold_carries_the_reference_target_to_detachment_and_records_the_shadow_stop():
    from isaaclab_pruning.task.simulated_cut import CLOSURE_HOLD_WAIVED_CHECKS

    controller = demo(*[tracking(MOUTH)] * 4, occluded(), *[latched()] * 5, closure_hold=True)
    _into_closing(controller)
    reference = controller.cutter.closure_reference
    assert reference.target_position_w == tuple(MOUTH) and reference.vision_timestamp_s == 0.3
    evidence = observe(controller, 0.4)
    hold = evidence["closure_hold"]
    assert hold["eligible"] and hold["stationary"] and hold["explained"] and hold["held"] and hold["latched_by_jaw"]
    assert hold["explanation"] == "too_few_unmasked_patch_pixels" and hold["dt_mm"] == 0.0 and hold["dr_deg"] == 0.0
    certificate = evidence["cut"]["certificate"]
    assert certificate["vision_source"] == "closure_hold"
    assert certificate["waived_checks"] == CLOSURE_HOLD_WAIVED_CHECKS
    assert certificate["held_target_age_s"] == pytest.approx(0.1) == hold["held_target_age_s"]
    assert certificate["mouth_distance_m"] == 0.0 and certificate["ready_to_close"]
    assert evidence["cut"]["phase"] == "closing" and evidence["cut"]["closure_progress"] == pytest.approx(1 / 6)
    # M alone (the shadow cutter fed the honest observation) stops here, as the unchanged gate would.
    assert evidence["cut_without_hold"] == {"phase": "stopped", "stopped_reason": "vision_invalid"}
    for time_s in (0.5, 0.6, 0.7, 0.8):
        evidence = observe(controller, time_s)
        assert (
            evidence["closure_hold"]["held"] and evidence["closure_hold"]["explanation"] == "latched_by_held_jaw_loss"
        )
        assert evidence["cut"]["phase"] == "closing" and not evidence["cut"]["detach_event"]
    evidence = observe(controller, 0.9)
    assert evidence["cut"]["detach_event"] and evidence["cut"]["phase"] == "retreat"
    assert evidence["cut"]["certificate"]["vision_source"] == "closure_hold"
    assert evidence["cut"]["certificate"]["held_target_age_s"] == pytest.approx(0.6)
    assert evidence["cut_without_hold"] == {"phase": "stopped", "stopped_reason": "vision_invalid"}
    displaced = POSE.copy()
    displaced[:3] += [0.01, 0, 0]
    assert controller.command(displaced, 0.9)[1] == "retreat"
    json.dumps(evidence, allow_nan=False)


def test_flow_failure_after_a_mask_drop_is_held():
    flow = lost("optical_flow_failed", jaw_mask_dropped_valid=1, jaw_mask_unmasked_flow_would_pass=True)
    controller = demo(*[tracking(MOUTH)] * 4, flow, latched(), closure_hold=True)
    _into_closing(controller)
    assert observe(controller, 0.4)["closure_hold"]["explanation"] == "flow_failed_only_after_mask_drop"
    assert observe(controller, 0.5)["closure_hold"]["held"]
    assert controller.cut_step.phase == "closing"


@pytest.mark.parametrize(
    "measurement",
    [
        lost("appearance_changed_or_occluded"),
        lost("optical_flow_failed", jaw_mask_dropped_valid=0, jaw_mask_unmasked_flow_would_pass=True),
        lost("optical_flow_failed", jaw_mask_dropped_valid=3, jaw_mask_unmasked_flow_would_pass=False),
        lost("optical_flow_failed"),
        lost("world_target_jump_or_wrong_surface"),
        lost("low_confidence"),
        {"state": "invalid_depth", "reason": "depth_measurement_rejected", "target_position_world_m": None},
        latched(),
    ],
)
def test_losses_the_mask_does_not_explain_still_stop_during_closure(measurement):
    controller = demo(*[tracking(MOUTH)] * 4, measurement, closure_hold=True)
    _into_closing(controller)
    evidence = observe(controller, 0.4)
    assert not evidence["closure_hold"]["held"] and not evidence["closure_hold"]["explained"]
    assert evidence["cut"]["phase"] == "stopped" and evidence["cut"]["stopped_reason"] == "vision_invalid"
    assert evidence["cut"]["certificate"]["vision_source"] == "live"
    assert evidence["cut_without_hold"] == {"phase": "stopped", "stopped_reason": "vision_invalid"}


@pytest.mark.parametrize(
    "delta,stationary",
    [
        ({"translation": [0.0004, 0.0, 0.0]}, True),
        ({"translation": [0.0006, 0.0, 0.0]}, False),
        ({"rotation_deg": 0.2}, True),
        ({"rotation_deg": 0.3}, False),
    ],
)
def test_the_hold_requires_the_tool_still_at_its_closure_start_pose(delta, stationary):
    moved = POSE.copy()
    moved[:3] += delta.get("translation", [0.0, 0.0, 0.0])
    half = np.radians(delta.get("rotation_deg", 0.0)) / 2
    moved[3:] = [np.cos(half), 0.0, 0.0, np.sin(half)]  # about the tool axis: the mouth stays put
    controller = demo(*[tracking(MOUTH)] * 4, occluded(), closure_hold=True)
    _into_closing(controller)
    evidence = observe(controller, 0.4, moved)
    assert evidence["closure_hold"]["stationary"] is stationary
    assert evidence["closure_hold"]["held"] is stationary
    assert evidence["cut"]["phase"] == ("closing" if stationary else "stopped")


def test_no_hold_in_approach_or_align():
    approach = demo(tracking(MOUTH + [0.1, 0, 0]), occluded(), closure_hold=True)
    assert observe(approach, 0.0)["cut"]["phase"] == "approach"
    evidence = observe(approach, 0.1)
    assert not evidence["closure_hold"]["eligible"] and not evidence["closure_hold"]["held"]
    assert evidence["closure_hold"]["explained"]  # a jaw occlusion, but not during closure
    assert evidence["cut"]["stopped_reason"] == "vision_invalid"
    align = demo(tracking(MOUTH), tracking(MOUTH), occluded(), closure_hold=True)
    observe(align, 0.0)
    assert observe(align, 0.1)["cut"]["phase"] == "align"
    evidence = observe(align, 0.2)
    assert not evidence["closure_hold"]["held"] and evidence["cut"]["stopped_reason"] == "vision_invalid"
    assert align.latched_by_jaw is False


def test_external_stops_reach_the_shadow_cutter_too():
    controller = demo(*[tracking(MOUTH)] * 6, closure_hold=True)
    observe(controller, 0.0)
    observe(controller, 0.1)
    controller.stop("tof_minimum_clearance")
    evidence = observe(controller, 0.2)
    assert evidence["cut"]["stopped_reason"] == "tof_minimum_clearance"
    assert evidence["cut_without_hold"] == {"phase": "stopped", "stopped_reason": "tof_minimum_clearance"}


def test_arm_evidence_appears_only_with_its_flag():
    masked = masked_demo(*[tracking(MOUTH)] * 2, jaw_self_mask=True)
    assert masked.evidence()["jaw_self_mask"] is None and "closure_hold" not in masked.evidence()
    evidence = observe_frame(masked, 0.0)
    assert set(evidence) == UNCHANGED_EVIDENCE_KEYS | {"jaw_self_mask"}
    held = demo(tracking(MOUTH), closure_hold=True)
    assert held.evidence()["cut_without_hold"] is None
    assert set(observe(held, 0.0)) == UNCHANGED_EVIDENCE_KEYS | {"closure_hold", "cut_without_hold"}


def _synthetic_frame(progress, radius):
    """A static textured plane through the mouth height, with the jaw surrogate drawn where it is rendered."""
    from isaaclab_pruning.perception.jaw_self_mask import jaw_boxes, jaw_mask

    rgb = np.random.default_rng(5).integers(20, 236, (*SHAPE, 3), dtype=np.uint8)
    depth = np.full(SHAPE, 0.12, dtype=np.float32)
    jaw = jaw_mask(jaw_boxes(POSE, 0.0, progress, radius), CAMERA, jaw_camera(), SHAPE, margin_px=0.0)
    rgb[jaw] = (107, 120, 135)
    depth[jaw] = 0.105
    return rgb, depth


@pytest.mark.parametrize("closure_hold", [True, False])
def test_real_tracker_mask_and_hold_on_a_synthetic_closure(closure_hold):
    # The tracked bark sits 6 mm from the mouth toward one jaw; the closing jaw covers its patch mid-closure.
    pytest.importorskip("cv2")
    radius = 0.002
    controller = VisionPruningDemo("branch_7", (0, 1, 0), radius, POSE, jaw_self_mask=True, closure_hold=closure_hold)
    rgb, depth = _synthetic_frame(0.0, radius)
    seed = [240 + 320 * 0.006 / 0.12, 160.0]
    initialized = controller.initialize(
        rgb, depth, seed, camera_matrix=CAMERA, world_from_optical=jaw_camera(), tool_pose_wxyz=POSE
    )
    assert initialized["state"] == "initialized"
    states, phases = [], []
    for step in range(10):
        progress = controller.cut_step.closure_progress if controller.cut_step else 0.0
        rgb, depth = _synthetic_frame(progress, radius)
        evidence = controller.observe(rgb, depth, CAMERA, jaw_camera(), step / 10, POSE)
        states.append(evidence["measurement"]["state"])
        phases.append(evidence["cut"]["phase"])
        json.dumps(evidence, allow_nan=False)
        if evidence["cut"]["phase"] in ("stopped", "retreat"):
            break
    first_occluded = states.index("jaw_mask_occluded")
    assert first_occluded > phases.index("closing")
    assert all(state == "tracking" for state in states[:first_occluded])
    if closure_hold:
        assert phases[-1] == "retreat" and evidence["cut"]["detach_event"]
        assert evidence["cut"]["certificate"]["vision_source"] == "closure_hold"
        assert evidence["cut_without_hold"] == {"phase": "stopped", "stopped_reason": "vision_invalid"}
    else:
        assert phases[-1] == "stopped" and evidence["cut"]["stopped_reason"] == "vision_invalid"
        assert len(states) == first_occluded + 1


class DepthRecordingTracker(StubTracker):
    """Records the jaw boxes each update received; refuses the self-mask, as the registered variant does."""

    def __init__(self, responses, config):
        super().__init__(responses, config)
        self.boxes = []

    def update(self, rgb, depth, matrix, transform, exclusion_mask=None, *, jaw_boxes=None):
        assert exclusion_mask is None
        self.boxes.append(jaw_boxes)
        return self.responses.popleft()


def test_depth_appearance_runs_the_strict_depth_tracker_alone_with_the_rendered_jaw():
    from isaaclab_pruning.perception.depth_appearance import DepthAppearanceTracker, frame_jaw_boxes
    from isaaclab_pruning.perception.visual_servo import VisualServoTracker

    plain = VisionPruningDemo("branch_7", (0, 0, 1), 0.004, POSE)
    assert type(plain.tracker) is VisualServoTracker and plain.depth_appearance is False
    real = VisionPruningDemo("branch_7", (0, 0, 1), 0.004, POSE, depth_appearance=True)
    assert type(real.tracker) is DepthAppearanceTracker and real.tracker.arm == "strict"
    assert real.tracker.config == plain.tracker.config
    for other in ({"jaw_self_mask": True}, {"closure_hold": True}):
        with pytest.raises(ValueError, match="depth_appearance"):
            VisionPruningDemo("branch_7", (0, 0, 1), 0.004, POSE, depth_appearance=True, **other)
    with pytest.raises(ValueError, match="bool"):
        VisionPruningDemo("branch_7", (0, 0, 1), 0.004, POSE, depth_appearance=1)

    controller = VisionPruningDemo("branch_7", (0, 0, 1), 0.004, POSE, depth_appearance=True)
    controller.tracker = DepthRecordingTracker([tracking(MOUTH)] * 8, controller.tracker.config)
    progresses = []
    for time_s in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7):
        rendered = controller.cut_step.closure_progress if controller.cut_step else 0.0
        evidence = observe_frame(controller, time_s)
        boxes, expected = controller.tracker.boxes[-1], frame_jaw_boxes(POSE, 0.0, rendered, 0.004)
        assert len(boxes) == len(expected) == 2
        for (centre, rotation, half), (centre_e, rotation_e, half_e) in zip(boxes, expected, strict=True):
            np.testing.assert_array_equal(centre, centre_e)
            np.testing.assert_array_equal(rotation, rotation_e)
            np.testing.assert_array_equal(half, half_e)
        assert evidence["depth_appearance"] == {"closure_progress_used": rendered, "jaw_boxes_known": True}
        assert "jaw_self_mask" not in evidence and "closure_hold" not in evidence
        json.dumps(evidence, allow_nan=False)
        progresses.append(rendered)
    assert progresses[:5] == [0.0] * 5 and progresses[5] > 0
