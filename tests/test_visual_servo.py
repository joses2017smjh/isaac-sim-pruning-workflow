from __future__ import annotations

import json

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from isaaclab_pruning.perception.visual_servo import (
    VisualServoConfig,
    VisualServoTracker,
    bounded_visual_servo_translation,
    project_depth_to_world,
)


def _scene(dx=0, dy=0, *, depth_value=0.6):
    image = np.full((160, 240, 3), 30, dtype=np.uint8)
    patch = np.random.default_rng(7).integers(45, 240, (52, 28, 3), dtype=np.uint8)
    x, y = 120 + dx, 80 + dy
    image[y - 26 : y + 26, x - 14 : x + 14] = patch
    depth = np.full((160, 240), 2.0, dtype=np.float32)
    depth[y - 26 : y + 26, x - 14 : x + 14] = depth_value
    return image, depth


K = np.array([[160.0, 0, 120], [0, 160.0, 80], [0, 0, 1]])
WORLD_FROM_OPTICAL = np.eye(4)


def _tracker():
    image, depth = _scene()
    tracker = VisualServoTracker()
    result = tracker.initialize(image, [120, 80], depth)
    assert result["state"] == "initialized"
    assert result["target_position_world_m"] is None
    return tracker


def test_tracks_consecutive_rgb_translation_and_backprojects_measured_depth():
    tracker = _tracker()
    for dx, dy in [(2, 1), (4, 2), (6, 3)]:
        image, depth = _scene(dx, dy)
        result = tracker.update(image, depth, K, WORLD_FROM_OPTICAL)
        assert result["state"] == "tracking", result
        np.testing.assert_allclose(result["pixel_xy"], [120 + dx, 80 + dy], atol=0.15)
        np.testing.assert_allclose(result["target_position_world_m"], [0.6 * dx / 160, 0.6 * dy / 160, 0.6], atol=0.001)
        assert result["feature_count"] >= tracker.config.min_features
        assert result["confidence"] >= tracker.config.min_confidence
        json.dumps(result, allow_nan=False)


def test_projection_uses_optical_z_not_ray_length_and_respects_world_rotation():
    matrix = np.array([[320, 0, 240], [0, 320, 160], [0, 0, 1]])
    transform = np.array([[0, 0, 1, 0.1], [-1, 0, 0, 0.2], [0, -1, 0, 0.3], [0, 0, 0, 1]])
    np.testing.assert_allclose(project_depth_to_world([400, 240], 2.0, matrix, transform), [2.1, -0.8, -0.2])


@pytest.mark.parametrize("invalid", [np.nan, np.inf, 0.0, -1.0, 100.0])
def test_depth_dropout_never_returns_stale_world_target_and_can_recover(invalid):
    tracker = _tracker()
    image, depth = _scene()
    assert tracker.update(image, depth, K, WORLD_FROM_OPTICAL)["state"] == "tracking"
    result = tracker.update(image, np.full_like(depth, invalid), K, WORLD_FROM_OPTICAL)
    assert result["state"] == "invalid_depth"
    assert result["target_position_world_m"] is None
    assert result["confidence"] == 0
    json.dumps(result, allow_nan=False)
    assert tracker.update(image, depth, K, WORLD_FROM_OPTICAL)["state"] == "tracking"


def test_blank_frame_latches_loss_until_explicit_reinitialization():
    tracker = _tracker()
    image, depth = _scene()
    lost = tracker.update(np.zeros_like(image), depth, K, WORLD_FROM_OPTICAL)
    assert lost["state"] == "tracking_lost"
    assert lost["target_position_world_m"] is None
    assert lost["requires_reinitialize"]
    assert tracker.update(image, depth, K, WORLD_FROM_OPTICAL)["target_position_world_m"] is None
    tracker.initialize(image, [120, 80], depth)
    assert tracker.update(image, depth, K, WORLD_FROM_OPTICAL)["state"] == "tracking"


def test_textured_occlusion_rejected_instead_of_reseeding_scene_target():
    tracker = _tracker()
    image, depth = _scene()
    image[52:108, 104:136] = np.random.default_rng(99).integers(0, 256, (56, 32, 3), dtype=np.uint8)
    result = tracker.update(image, depth, K, WORLD_FROM_OPTICAL)
    assert result["state"] == "tracking_lost"
    assert result["target_position_world_m"] is None


def test_initial_blank_image_cannot_authorize_motion():
    tracker = VisualServoTracker()
    result = tracker.initialize(np.zeros((160, 240, 3), dtype=np.uint8), [120, 80])
    assert result["state"] == "initialization_failed"
    assert result["reason"] == "insufficient_texture"
    assert result["requires_reinitialize"]


def test_identical_weak_texture_frames_fail_closed_without_lowering_lk_gate():
    # goodFeaturesToTrack uses a relative quality threshold, while LK rejects
    # these tiny gradients at its default absolute minEigThreshold=1e-4.
    # Job 21222710 hit this mismatch on an occluding tool surface. Identical
    # frames alone do not prove that an initialized feature is trackable.
    image = np.full((160, 240, 3), 60, dtype=np.uint8)
    image[54:106, 106:134] = np.random.default_rng(7).integers(59, 62, (52, 28, 3), dtype=np.uint8)
    depth = np.full((160, 240), 0.6, dtype=np.float32)
    tracker = VisualServoTracker()
    initialized = tracker.initialize(image, [120, 80], depth)
    assert initialized["target_position_world_m"] is None
    if initialized["state"] == "initialized":
        assert initialized["feature_count"] >= tracker.config.min_features
        measurement = tracker.update(image.copy(), depth, K, WORLD_FROM_OPTICAL)
        assert measurement["state"] == "tracking_lost"
        assert measurement["reason"] == "optical_flow_failed"
        assert measurement["flow_reason"] == "too_few_roundtrip_inliers"
        assert measurement["roundtrip_inlier_count"] < tracker.config.min_features
    else:
        # An initializer may reject these unusable features earlier; neither
        # outcome is permission to move toward a guessed or stale target.
        assert initialized["state"] == "initialization_failed"
        measurement = initialized
    assert measurement["target_position_world_m"] is None
    command = bounded_visual_servo_translation(
        measurement,
        (0.0, 0.0, 0.0),
        approach_axis_world=(0.0, 0.0, 1.0),
        standoff_m=0.1,
        now_s=1.0,
        measurement_time_s=1.0,
    )
    assert command["state"] == "hold"
    np.testing.assert_array_equal(command["delta_world_m"], [0.0, 0.0, 0.0])


def test_initializer_owns_gray_pixels_instead_of_aliasing_input_rgb():
    image, depth = _scene()
    tracker = VisualServoTracker()
    assert tracker.initialize(image, [120, 80], depth)["state"] == "initialized"
    previous_gray = tracker._previous_gray.copy()
    assert not np.shares_memory(image, tracker._previous_gray)
    image[:] = 0
    np.testing.assert_array_equal(tracker._previous_gray, previous_gray)


def test_initial_depth_masks_background_feature_points():
    image, depth = _scene()
    image[:, :106] = np.random.default_rng(99).integers(0, 256, image[:, :106].shape, dtype=np.uint8)
    tracker = VisualServoTracker(VisualServoConfig(roi_half_size_px=(35, 35)))
    initialized = tracker.initialize(image, [120, 80], depth)
    assert initialized["state"] == "initialized"
    assert all(106 <= x < 134 and 54 <= y < 106 for x, y in initialized["feature_pixels_xy"])


def test_mixed_depth_surfaces_and_invalid_center_are_rejected():
    tracker = _tracker()
    image, depth = _scene()
    depth[78:83, 118:120] = 2.0
    result = tracker.update(image, depth, K, WORLD_FROM_OPTICAL)
    assert result["state"] == "invalid_depth"
    assert result["depth_reason"] == "mixed_surfaces"
    _, depth = _scene()
    depth[80, 120] = np.nan
    result = tracker.update(image, depth, K, WORLD_FROM_OPTICAL)
    assert result["depth_reason"] == "invalid_or_inconsistent_center"


def test_large_world_jump_latches_loss_without_moving_to_background():
    tracker = _tracker()
    image, depth = _scene()
    assert tracker.update(image, depth, K, WORLD_FROM_OPTICAL)["state"] == "tracking"
    result = tracker.update(image, depth + 0.2, K, WORLD_FROM_OPTICAL)
    assert result["state"] == "tracking_lost"
    assert result["reason"] == "world_target_jump_or_wrong_surface"
    assert result["target_position_world_m"] is None


def test_bad_calibration_never_yields_position():
    tracker = _tracker()
    image, depth = _scene()
    bad_transform = np.diag([1, -1, 1, 1])
    result = tracker.update(image, depth, K, bad_transform)
    assert result["state"] == "invalid_calibration"
    assert result["target_position_world_m"] is None
    with pytest.raises(ValueError, match="positive focal"):
        project_depth_to_world([120, 80], 1.0, np.zeros((3, 3)), np.eye(4))


def test_mismatched_frame_and_depth_shapes_fail_closed():
    tracker = _tracker()
    image, depth = _scene()
    assert tracker.update(image, depth[:10], K, WORLD_FROM_OPTICAL)["state"] == "invalid_depth"
    assert tracker.update(image[:10], depth, K, WORLD_FROM_OPTICAL)["state"] == "tracking_lost"


@pytest.mark.parametrize(
    "kwargs", [{"min_features": 2}, {"depth_radius_px": -1}, {"min_confidence": float("nan")}, {"max_world_jump_m": 0}]
)
def test_invalid_thresholds_are_rejected(kwargs):
    with pytest.raises(ValueError):
        VisualServoConfig(**kwargs)


def _servo(measurement=None, **kwargs):
    options = dict(approach_axis_world=(0, 0, 2), standoff_m=0.1, now_s=1.0, measurement_time_s=1.0)
    options.update(kwargs)
    if measurement is None:
        measurement = {"state": "tracking", "target_position_world_m": [0, 0, 0.5]}
    return bounded_visual_servo_translation(measurement, (0, 0, 0), **options)


def test_visual_servo_uses_current_measurement_with_unit_axis_and_bounded_step():
    step = _servo()
    assert step["state"] == "tracking"
    assert step["remaining_distance_m"] == pytest.approx(0.4)
    np.testing.assert_allclose(step["command_position_world_m"], [0, 0, 0.004])
    changed = _servo({"state": "tracking", "target_position_world_m": [0.2, 0, 0.1]})
    np.testing.assert_allclose(changed["delta_world_m"], [0.004, 0, 0], atol=1e-12)
    assert np.linalg.norm(changed["delta_world_m"]) <= 0.004
    json.dumps(changed, allow_nan=False)


def test_visual_servo_cannot_overshoot_or_divide_by_zero_at_target():
    close = _servo({"state": "tracking", "target_position_world_m": [0, 0, 0.101]})
    np.testing.assert_allclose(close["delta_world_m"], [0, 0, 0.001])
    reached = _servo({"state": "tracking", "target_position_world_m": [0, 0, 0.1]})
    assert reached["delta_world_m"] == [0, 0, 0]


@pytest.mark.parametrize("state", ["tracking_lost", "invalid_depth", "invalid_calibration", "initialized"])
def test_visual_servo_loss_holds_even_when_caller_leaves_old_target_in_payload(state):
    step = _servo({"state": state, "target_position_world_m": [0, 0, 0.5]})
    assert step["state"] == "hold" and step["reason"] == "vision_not_tracking"
    assert step["delta_world_m"] == [0, 0, 0]
    assert step["target_position_world_m"] is None


@pytest.mark.parametrize("target", [None, [0, 0], [0, float("nan"), 1], "not a point"])
def test_visual_servo_missing_or_malformed_target_holds(target):
    assert _servo({"state": "tracking", "target_position_world_m": target})["reason"] == "invalid_target"


@pytest.mark.parametrize("timestamp", [0.5, 1.1, float("nan"), float("inf")])
def test_visual_servo_stale_or_future_target_holds(timestamp):
    step = _servo(measurement_time_s=timestamp)
    assert step["reason"] == "vision_stale_or_future"
    assert step["delta_world_m"] == [0, 0, 0]


def test_visual_servo_hazard_holds_independently_of_valid_perception():
    assert _servo(hazard_contact=True)["reason"] == "hazard_contact"


@pytest.mark.parametrize(
    "kwargs", [{"approach_axis_world": [0, 0, 0]}, {"standoff_m": -0.1}, {"max_step_m": 0}, {"max_age_s": float("nan")}]
)
def test_visual_servo_invalid_control_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):
        _servo(**kwargs)
