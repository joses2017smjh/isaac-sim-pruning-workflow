from __future__ import annotations

import math
from dataclasses import asdict, replace

import numpy as np
import pytest

from isaaclab_pruning.task.simulated_cut import (
    MODEL_LABEL,
    CutConfig,
    CutObservation,
    SimulatedCutController,
    evaluate_cut_gate,
    tool_mouth_geometry,
)


def observation(time_s: float = 0.0, **kwargs) -> CutObservation:
    base = CutObservation(
        time_s=time_s,
        vision_timestamp_s=time_s,
        target_id="branch_7",
        target_semantic="branch",
        vision_valid=True,
        target_position_w=(1.0, 2.0, 3.0),
        target_axis_w=(0.0, 0.0, 1.0),
        target_radius_m=0.004,
        mouth_position_w=(1.0, 2.0, 3.0),
        cutter_closing_axis_w=(1.0, 0.0, 0.0),
    )
    return replace(base, **kwargs)


def controller(**kwargs) -> SimulatedCutController:
    return SimulatedCutController(CutConfig(selected_target_id="branch_7", **kwargs))


def test_success_requires_stable_live_frames_and_full_closure():
    cut = controller()
    assert cut.update(observation(0.0)).phase == "align"
    assert cut.update(observation(0.1)).stable_count == 2
    start = cut.update(observation(0.2))
    assert start.phase == "closing"
    assert start.closure_started_s == 0.2
    assert not start.detach_event
    halfway = cut.update(observation(0.45))
    assert halfway.closure_progress == pytest.approx(0.5)
    assert not halfway.detach_event
    complete = cut.update(observation(0.71))
    assert complete.phase == "retreat"
    assert complete.detach_event and complete.detached
    assert complete.detached_at_s == 0.71
    assert complete.model == MODEL_LABEL
    assert asdict(complete)["certificate"]["ready_to_close"]
    assert not cut.update(observation(0.8)).detach_event
    # The cut piece can disappear from the camera during retreat.
    assert cut.update(observation(1.0, vision_valid=False)).phase == "retreat"


@pytest.mark.parametrize("start,duration,end", [(7.2, 0.6, 7.8), (0.1, 0.2, 0.3), (12.3, 0.6, 12.9)])
def test_closure_deadline_roundoff_does_not_delay_release(start, duration, end):
    cut = controller(stable_frames=2, closure_duration_s=duration)
    cut.update(observation(start - 0.1))
    assert cut.update(observation(start)).phase == "closing"
    almost = cut.update(observation(end - 1e-6))
    assert not almost.detach_event
    assert almost.closure_progress < 1
    complete = cut.update(observation(end))
    assert complete.detach_event
    assert complete.closure_progress == 1
    assert complete.detached_at_s == end


@pytest.mark.parametrize(
    "change", [{"vision_valid": False}, {"hazard_contact": True}, {"mouth_position_w": (1.1, 2, 3)}]
)
def test_deadline_roundoff_correction_never_bypasses_current_gate(change):
    cut = controller(stable_frames=2, closure_duration_s=0.6)
    cut.update(observation(7.1))
    cut.update(observation(7.2))
    stopped = cut.update(observation(7.8, **change))
    assert stopped.phase == "stopped"
    assert not stopped.detach_event
    assert not stopped.detached


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"target_semantic": "post"}, "target_not_branch"),
        ({"target_semantic": "trunk"}, "target_not_branch"),
        ({"target_id": "another_branch"}, "wrong_target_id"),
        ({"vision_valid": False}, "vision_invalid"),
        ({"vision_timestamp_s": -0.3}, "vision_stale_or_future"),
        ({"vision_timestamp_s": 0.1}, "vision_stale_or_future"),
        ({"vision_timestamp_s": float("nan")}, "vision_stale_or_future"),
        ({"target_radius_m": 0.008}, "branch_exceeds_demo_cut_limit"),
        ({"target_radius_m": 0.0}, "invalid_branch_radius"),
        ({"target_radius_m": float("nan")}, "invalid_branch_radius"),
        ({"hazard_contact": True}, "hazard_contact"),
        ({"target_axis_w": (0.0, 0.0, 0.0)}, "invalid_geometry"),
        ({"mouth_position_w": (0.0, float("inf"), 0.0)}, "invalid_geometry"),
    ],
)
def test_unsafe_target_latches_stop_without_event(changes, reason):
    cut = controller()
    step = cut.update(observation(**changes))
    assert step.phase == "stopped"
    assert step.stopped_reason == reason
    assert not step.detach_event and not step.detached
    assert cut.update(observation(1.0)).phase == "stopped"


def test_far_target_keeps_approaching_and_wrong_alignment_holds():
    cut = controller()
    far = cut.update(observation(mouth_position_w=(1.0, 2.0, 2.9)))
    assert far.phase == "approach"
    assert far.certificate.safe_to_approach
    assert not far.certificate.ready_to_close
    wrong_axis = cut.update(observation(0.1, cutter_closing_axis_w=(0.0, 0.0, 1.0)))
    assert wrong_axis.phase == "align"
    assert wrong_axis.certificate.perpendicularity_error_deg == pytest.approx(90.0)
    assert wrong_axis.stable_count == 0


def test_perpendicularity_is_axis_sign_invariant():
    cfg = CutConfig("branch_7")
    tilted = (math.cos(math.radians(14)), 0.0, math.sin(math.radians(14)))
    forward = evaluate_cut_gate(observation(cutter_closing_axis_w=tilted), cfg)
    reverse = evaluate_cut_gate(observation(cutter_closing_axis_w=tuple(-x for x in tilted)), cfg)
    assert forward.ready_to_close and reverse.ready_to_close
    assert forward.perpendicularity_error_deg == pytest.approx(14.0)


def test_geometry_loss_during_closure_does_not_detach():
    cut = controller(stable_frames=2)
    cut.update(observation(0.0))
    assert cut.update(observation(0.1)).phase == "closing"
    step = cut.update(observation(0.8, mouth_position_w=(1.0, 2.0, 2.95)))
    assert step.phase == "stopped"
    assert step.stopped_reason == "gate_lost_during_closure"
    assert not step.detach_event


def test_vision_loss_during_closure_does_not_detach():
    cut = controller(stable_frames=2)
    cut.update(observation(0.0))
    cut.update(observation(0.1))
    step = cut.update(observation(0.8, vision_timestamp_s=0.1))
    assert step.stopped_reason == "vision_stale_or_future"
    assert not step.detach_event


def test_reused_vision_does_not_count_as_stable_frames():
    cut = controller()
    cut.update(observation(0.0))
    assert cut.update(observation(0.1, vision_timestamp_s=0.0)).stable_count == 0
    assert cut.update(observation(0.2, vision_timestamp_s=0.0)).phase == "align"
    assert cut.update(observation(0.3, vision_timestamp_s=0.0)).phase == "stopped"


def test_fast_motion_cannot_accumulate_stability():
    cut = controller()
    cut.update(observation(0.0))
    assert cut.update(observation(0.1, mouth_position_w=(1.01, 2.0, 3.0))).stable_count == 0
    assert cut.update(observation(0.2)).stable_count == 0
    assert cut.update(observation(0.3)).stable_count == 1


def test_target_jump_resets_stability():
    cut = controller(max_target_shift_m=0.005)
    cut.update(observation(0.0))
    step = cut.update(observation(0.1, target_position_w=(1.006, 2.0, 3.0)))
    assert step.phase == "align"
    assert step.stable_count == 0


def test_radius_change_resets_stability_even_when_both_radii_fit_mouth():
    cut = controller()
    cut.update(observation(0.0))
    step = cut.update(observation(0.1, target_radius_m=0.006))
    assert step.stable_count == 0
    assert not step.detach_event


def test_repeated_frame_cannot_advance_closure_even_while_fresh():
    cut = controller(stable_frames=2, closure_duration_s=0.1)
    cut.update(observation(0.0))
    cut.update(observation(0.1))
    step = cut.update(observation(0.21, vision_timestamp_s=0.1))
    assert step.stopped_reason == "vision_frame_reused_during_closure"
    assert not step.detach_event


@pytest.mark.parametrize(
    "changes",
    [{"mouth_position_w": (1.01, 2, 3)}, {"target_radius_m": 0.006}],
)
def test_motion_or_radius_change_during_closure_stops(changes):
    cut = controller(stable_frames=2)
    cut.update(observation(0.0))
    cut.update(observation(0.1))
    step = cut.update(observation(0.2, **changes))
    assert step.stopped_reason == "unstable_during_closure"
    assert not step.detach_event


def test_slow_cumulative_target_drift_during_closure_cannot_detach():
    cut = controller(stable_frames=2, max_target_shift_m=0.005)
    cut.update(observation(0.0))
    cut.update(observation(0.1))
    assert cut.update(observation(0.2, target_position_w=(1.004, 2, 3))).phase == "closing"
    step = cut.update(observation(0.3, target_position_w=(1.008, 2, 3)))
    assert step.stopped_reason == "unstable_during_closure"
    assert not step.detach_event


def test_reset_clears_stop_and_detachment():
    cut = controller(stable_frames=2)
    cut.update(observation(0.0, hazard_contact=True))
    cut.reset()
    cut.update(observation(0.0))
    cut.update(observation(0.1))
    assert cut.update(observation(0.7)).detach_event
    cut.reset()
    step = cut.update(observation(0.0))
    assert step.phase == "align"
    assert not step.detached
    assert step.detached_at_s is None and step.closure_started_s is None


def test_hazard_still_stops_after_detachment():
    cut = controller(stable_frames=2)
    cut.update(observation(0.0))
    cut.update(observation(0.1))
    cut.update(observation(0.7))
    step = cut.update(observation(0.8, hazard_contact=True))
    assert step.phase == "stopped" and step.detached
    assert not step.detach_event


@pytest.mark.parametrize("time_s", [0.0, -1.0, float("nan"), float("inf")])
def test_update_rejects_bad_time(time_s):
    cut = controller()
    cut.update(observation(0.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        cut.update(observation(time_s))


def test_regressing_camera_timestamp_stops():
    cut = controller()
    cut.update(observation(0.2))
    step = cut.update(observation(0.3, vision_timestamp_s=0.15))
    assert step.stopped_reason == "vision_timestamp_regressed"


def test_tool_mouth_offset_uses_wxyz_world_rotation():
    pose = (10.0, 20.0, 30.0, math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    mouth, closing_axis = tool_mouth_geometry(pose, (0.1, 0.0, 0.0))
    np.testing.assert_allclose(mouth, (10.0, 20.1, 30.0), atol=1e-12)
    np.testing.assert_allclose(closing_axis, (0.0, 1.0, 0.0), atol=1e-12)


def test_tool_mouth_rejects_zero_quaternion():
    with pytest.raises(ValueError, match="quaternion"):
        tool_mouth_geometry((0.0,) * 7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_branch_radius_m": 0.0},
        {"closure_duration_s": float("nan")},
        {"stable_frames": 1},
        {"stable_frames": 3.5},
        {"alignment_tolerance_deg": 90},
        {"allowed_semantics": ("post",)},
    ],
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        controller(**kwargs)
