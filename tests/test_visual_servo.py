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
    propagate_pixel,
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


@pytest.mark.parametrize(
    "quality", [0, -0.1, 1.001, np.nan, np.inf, -np.inf, True, False, np.bool_(True), None, "0.005"]
)
def test_feature_quality_level_rejects_invalid_or_boolean_values(quality):
    with pytest.raises(ValueError, match="feature_quality_level"):
        VisualServoConfig(feature_quality_level=quality)


@pytest.mark.parametrize("quality", [0.005, 0.02, 1.0])
def test_initializer_passes_configured_corner_quality_and_records_provenance(monkeypatch, quality):
    original = cv2.goodFeaturesToTrack
    calls = []

    def measured_call(*args, **kwargs):
        calls.append(kwargs.copy())
        return original(*args, **kwargs)

    monkeypatch.setattr(cv2, "goodFeaturesToTrack", measured_call)
    tracker = VisualServoTracker(VisualServoConfig(feature_quality_level=quality))
    image, depth = _scene()
    initialized = tracker.initialize(image, [120, 80], depth)
    assert len(calls) == 1
    assert calls[0]["qualityLevel"] == quality
    assert calls[0]["minDistance"] == 3
    assert calls[0]["blockSize"] == 3
    assert calls[0]["mask"].shape == depth.shape
    assert initialized["initial_feature_config"] == {
        "quality_level": quality,
        "min_distance_px": 3,
        "block_size_px": 3,
        "roi_half_size_px": [14, 24],
    }
    assert tracker.config.min_features == 4
    assert tracker.config.max_roundtrip_error_px == 1.0
    json.dumps(initialized, allow_nan=False)


def test_default_initializer_quality_remains_unchanged():
    assert VisualServoConfig().feature_quality_level == 0.02


@pytest.mark.parametrize("mode", ["auto", "CLAHE", "", True, None])
def test_photometric_mode_rejects_invalid_values(mode):
    with pytest.raises(ValueError, match="photometric_normalization"):
        VisualServoConfig(photometric_normalization=mode)


def test_raw_default_has_identical_measurements_to_explicit_raw():
    default = VisualServoTracker()
    explicit = VisualServoTracker(VisualServoConfig(photometric_normalization="raw"))
    image, depth = _scene()
    assert default.initialize(image, [120, 80], depth) == explicit.initialize(image, [120, 80], depth)
    image, depth = _scene(2, 1)
    assert default.update(image, depth, K, WORLD_FROM_OPTICAL) == explicit.update(image, depth, K, WORLD_FROM_OPTICAL)


@pytest.mark.parametrize("mode", ["raw", "clahe"])
def test_photometric_modes_preserve_inputs_translation_and_fail_closed_gates(mode):
    tracker = VisualServoTracker(VisualServoConfig(photometric_normalization=mode))
    image, depth = _scene()
    original_image, original_depth = image.copy(), depth.copy()
    initialized = tracker.initialize(image, [120, 80], depth)
    assert initialized["state"] == "initialized"
    np.testing.assert_array_equal(image, original_image)
    np.testing.assert_array_equal(depth, original_depth)
    assert initialized["photometric_normalization"]["mode"] == mode
    assert initialized["photometric_normalization"]["clahe_clip_limit"] == (2.0 if mode == "clahe" else None)
    image, depth = _scene(2, 1)
    measured = tracker.update(image, depth, K, WORLD_FROM_OPTICAL)
    assert measured["state"] == "tracking"
    np.testing.assert_allclose(measured["pixel_xy"], [122, 81], atol=0.2)
    dropped = tracker.update(image, np.full_like(depth, np.nan), K, WORLD_FROM_OPTICAL)
    assert dropped["state"] == "invalid_depth"
    assert dropped["target_position_world_m"] is None
    assert tracker.update(image, depth, K, WORLD_FROM_OPTICAL)["state"] == "tracking"
    lost = tracker.update(np.zeros_like(image), depth, K, WORLD_FROM_OPTICAL)
    assert lost["state"] == "tracking_lost"
    assert lost["target_position_world_m"] is None
    assert tracker.update(image, depth, K, WORLD_FROM_OPTICAL)["reason"] == "explicit_initialization_required"
    json.dumps(lost, allow_nan=False)


def test_maintenance_adds_only_same_depth_candidates_after_validated_measurement(monkeypatch):
    image, depth = _scene()
    tracker = VisualServoTracker(VisualServoConfig(replenish_features=True))
    tracker.initialize(image, [120, 80], depth)
    tracker._points = tracker._points[: tracker.config.min_features].copy()
    # Keep confidence valid to isolate maintenance from the independent gate.
    tracker._initial_count = tracker.config.min_features * 2
    result = tracker.update(image, depth, K, WORLD_FROM_OPTICAL)
    assert result["state"] == "tracking"
    assert result["feature_count"] == tracker.config.min_features
    pending = result["pending_feature_pixels_for_next_frame"]
    assert 0 < len(pending) <= result["feature_count"]
    assert len(tracker._points) == result["feature_count"] + len(pending)
    assert all(106 <= x < 134 and 54 <= y < 106 for x, y in pending)
    for x, y in pending:
        assert abs(depth[round(y), round(x)] - result["depth_m"]) < tracker.config.max_depth_spread_m
    # Merely adding corners cannot rescue failed validation on the next image.
    monkeypatch.setattr(tracker, "_track_points", lambda gray: (None, None, {"flow_reason": "test_loss"}))
    lost = tracker.update(image, depth, K, WORLD_FROM_OPTICAL)
    assert lost["state"] == "tracking_lost"
    assert lost["target_position_world_m"] is None
    assert "pending_feature_pixels_for_next_frame" not in lost
    assert tracker.update(image, depth, K, WORLD_FROM_OPTICAL)["reason"] == "explicit_initialization_required"


@pytest.mark.parametrize("failure", ["depth", "confidence", "appearance"])
def test_invalid_measurements_cannot_add_maintenance_features(monkeypatch, failure):
    image, depth = _scene()
    tracker = VisualServoTracker(VisualServoConfig(replenish_features=True))
    tracker.initialize(image, [120, 80], depth)
    if failure == "depth":
        depth[:] = np.nan
    elif failure == "confidence":
        tracker._initial_count = 10000
    else:
        image[:] = 0
    monkeypatch.setattr(tracker, "_replenish_for_next_frame", lambda *args: pytest.fail("invalid maintenance"))
    assert tracker.update(image, depth, K, WORLD_FROM_OPTICAL)["state"] != "tracking"


def test_feature_maintenance_is_opt_in_and_requires_boolean():
    assert VisualServoConfig().replenish_features is False
    with pytest.raises(ValueError, match="replenish_features"):
        VisualServoConfig(replenish_features="yes")


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


def test_motion_model_defaults_to_translation_and_rejects_unknown_values():
    assert VisualServoConfig().motion_model == "translation"
    with pytest.raises(ValueError):
        VisualServoConfig(motion_model="affine")
    # Every gate is identical whatever the motion model.
    base, similar = VisualServoConfig(), VisualServoConfig(motion_model="similarity")
    for name in (
        "min_features",
        "max_roundtrip_error_px",
        "max_lk_error",
        "max_flow_residual_px",
        "min_patch_correlation",
        "min_confidence",
        "depth_radius_px",
        "max_depth_spread_m",
        "max_world_jump_m",
    ):
        assert getattr(base, name) == getattr(similar, name)


def test_similarity_follows_a_zoom_that_a_median_flow_drifts_under():
    # Features sit to one side of the tracked pixel while the image scales by
    # 1.2 about a point: the median flow carries their mean displacement onto
    # the pixel, the similarity fit maps the pixel exactly.
    rng = np.random.default_rng(3)
    centre = np.array([200.0, 120.0])
    old = centre + np.column_stack([rng.uniform(4, 12, 20), rng.uniform(-10, 10, 20)])
    pixel = np.array([198.0, 120.0])
    new = centre + 1.2 * (old - centre) + np.array([1.5, -0.5])
    truth = centre + 1.2 * (pixel - centre) + np.array([1.5, -0.5])
    moved, info = propagate_pixel(pixel, old, new, "translation")
    assert info == {"motion_model": "translation"}
    assert np.linalg.norm(moved - truth) > 1.0
    fitted, info = propagate_pixel(pixel, old, new, "similarity")
    assert info["motion_model"] == "similarity" and not info["similarity_fallback"]
    assert info["similarity_scale"] == pytest.approx(1.2, abs=1e-3)
    np.testing.assert_allclose(fitted, truth, atol=1e-3)
    with pytest.raises(ValueError):
        propagate_pixel(pixel, old, new[:3], "similarity")
    with pytest.raises(ValueError):
        propagate_pixel(pixel, old, new, "homography")


def test_similarity_tracker_matches_translation_on_a_pure_translation_and_keeps_every_gate():
    results = {}
    for model in ("translation", "similarity"):
        tracker = VisualServoTracker(VisualServoConfig(motion_model=model))
        image, depth = _scene()
        assert tracker.initialize(image, (120, 80), depth)["state"] == "initialized"
        for dx, dy in [(2, 1), (4, 2), (6, 3)]:
            image, depth = _scene(dx, dy)
            results[model] = tracker.update(image, depth, K, WORLD_FROM_OPTICAL)
            assert results[model]["state"] == "tracking", results[model]
    np.testing.assert_allclose(results["similarity"]["pixel_xy"], results["translation"]["pixel_xy"], atol=0.2)
    assert results["similarity"]["motion_model"] == "similarity"
    # A mixed-depth window still fails closed under the similarity model.
    tracker = VisualServoTracker(VisualServoConfig(motion_model="similarity"))
    image, depth = _scene()
    tracker.initialize(image, (120, 80), depth)
    image, depth = _scene(2, 1)
    depth[79:83, 121:124] = 1.5
    assert tracker.update(image, depth, K, WORLD_FROM_OPTICAL)["state"] != "tracking"


# ---------------------------------------------------------------------------------------------------------------
# Robot self-pixel exclusion mask (the jaw self-mask arm)
# ---------------------------------------------------------------------------------------------------------------
def _no_mask_keys(result):
    return {key: value for key, value in result.items() if not key.startswith("jaw_mask_")}


def _loss_sequence():
    """Tracking, depth dropout, recovery, motion, a blank frame, then the latched loss."""
    sequence = []
    for dx, dy in [(0, 0), (2, 1), (2, 1), (4, 2), (6, 3)]:
        sequence.append(_scene(dx, dy))
    image, depth = _scene(2, 1)
    sequence.insert(2, (image, np.full_like(depth, np.nan)))
    sequence.append((np.zeros_like(image), depth))
    sequence.append(_scene(6, 3))
    return sequence


@pytest.mark.parametrize(
    "config",
    [
        VisualServoConfig(),
        VisualServoConfig(depth_radius_px=1, feature_quality_level=0.005, replenish_features=True),
        VisualServoConfig(replenish_features=True, motion_model="similarity"),
    ],
)
def test_exclusion_mask_none_and_all_false_give_identical_measurements(config):
    plain, masked = VisualServoTracker(config), VisualServoTracker(config)
    image, depth = _scene()
    empty = np.zeros(depth.shape, bool)
    first = plain.initialize(image, [120, 80], depth)
    assert "jaw_mask_roi_pixels" not in first
    assert _no_mask_keys(masked.initialize(image, [120, 80], depth, exclusion_mask=empty)) == first
    for image, depth in _loss_sequence():
        expected = plain.update(image, depth, K, WORLD_FROM_OPTICAL)
        assert not any(key.startswith("jaw_mask_") for key in expected)
        result = masked.update(image, depth, K, WORLD_FROM_OPTICAL, exclusion_mask=empty)
        assert _no_mask_keys(result) == expected
        json.dumps(result, allow_nan=False)


def _jaw_scene(shift):
    """Static textured branch everywhere, plus two textured 'jaw' bands that move +2 px per frame in x.

    The bands cover rows 56-70 and 90-104 of the 29 x 49 ROI around (120, 80), well over half of it, and never
    the 13 x 13 appearance patch (rows 74-86). Depth is one surface, so every corner is a candidate feature.
    """
    rng = np.random.default_rng(11)
    image = rng.integers(20, 236, (160, 240, 3), dtype=np.uint8)
    jaw = np.random.default_rng(23).integers(0, 256, (49, 90, 3), dtype=np.uint8)
    mask = np.zeros((160, 240), bool)
    for top in (56, 90):
        x0 = 75 + shift
        image[top : top + 15, x0 : x0 + 90] = jaw[top - 56 : top - 56 + 15]
        mask[top : top + 15, x0 : x0 + 90] = True
    return image, np.full((160, 240), 0.6, dtype=np.float32), mask


def test_features_on_a_moving_masked_region_never_vote():
    results = {}
    for name in ("plain", "masked"):
        tracker = VisualServoTracker()
        image, depth, _ = _jaw_scene(0)
        assert tracker.initialize(image, [120, 80], depth)["state"] == "initialized"
        assert (
            sum(56 <= y <= 70 or 90 <= y <= 104 for _, y in tracker._points.reshape(-1, 2)) > len(tracker._points) / 2
        )
        trail = []
        for step in (1, 2, 3):
            image, depth, mask = _jaw_scene(2 * step)
            if name == "masked":
                # The previous frame's mask came from the frame before (the initial frame had no mask here).
                result = tracker.update(image, depth, K, WORLD_FROM_OPTICAL, exclusion_mask=mask)
            else:
                result = tracker.update(image, depth, K, WORLD_FROM_OPTICAL)
            trail.append(result)
        results[name] = trail
    masked = results["masked"]
    assert all(result["state"] == "tracking" for result in masked), [r["state"] for r in masked]
    for result in masked:
        np.testing.assert_allclose(result["pixel_xy"], [120, 80], atol=0.05)
        assert result["jaw_mask_patch_unmasked"] == 169
    assert masked[0]["jaw_mask_dropped_valid"] > 0
    # Without the mask the moving bands carry the median flow: the pixel leaves the static texture or is lost.
    plain = results["plain"]
    assert any(
        result["state"] != "tracking" or np.linalg.norm(np.subtract(result["pixel_xy"], [120, 80])) > 1.0
        for result in plain
    )


def _patch_tracker(masked_elements, config=None):
    """A tracker initialized at the integer pixel (120, 80) with ``masked_elements`` of its 13 x 13 patch masked."""
    image, depth = _scene()
    mask = np.zeros(depth.shape, bool)
    rows, cols = np.unravel_index(np.arange(masked_elements), (13, 13))
    mask[74 + rows, 114 + cols] = True
    tracker = VisualServoTracker(config)
    assert tracker.initialize(image, [120, 80], depth, exclusion_mask=mask)["state"] == "initialized"
    return tracker, image, depth, mask


def test_correlation_uses_only_elements_unmasked_in_both_patches():
    cv2 = pytest.importorskip("cv2")
    tracker, image, depth, mask = _patch_tracker(26)
    tracker._mask_current = np.zeros(depth.shape, bool)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    changed = gray.copy()
    changed[74:76, 114:127] = np.where(changed[74:76, 114:127] > 128, 0, 255)  # only the masked rows change
    correlation, info = tracker._masked_appearance(changed, np.float32([120.0, 80.0]))
    assert info["jaw_mask_patch_unmasked"] == 169 - 26 and info["jaw_mask_patch_masked_prev"] == 26
    assert correlation == pytest.approx(1.0, abs=1e-6)
    keep = np.ones((13, 13), bool)
    keep[:2, :] = False
    previous = gray[74:87, 114:127].astype(np.float32)[keep]
    shifted = np.roll(gray, 1, axis=1)
    current = shifted[74:87, 114:127].astype(np.float32)[keep]
    expected = np.mean((previous - previous.mean()) * (current - current.mean())) / (previous.std() * current.std())
    assert tracker._masked_appearance(shifted, np.float32([120.0, 80.0]))[0] == pytest.approx(expected, abs=1e-6)
    # The full-patch correlation of the changed frame is what the unmasked tracker would judge.
    assert tracker._appearance(changed, np.float32([120.0, 80.0])) < correlation - 0.2
    # Nothing masked: the full-patch formula, bit for bit.
    tracker._mask_previous = np.zeros(depth.shape, bool)
    assert tracker._masked_appearance(shifted, np.float32([120.0, 80.0]))[0] == tracker._appearance(
        shifted, np.float32([120.0, 80.0])
    )


def test_a_change_confined_to_masked_elements_keeps_tracking_and_only_lowers_the_unmasked_correlation():
    tracker, image, depth, mask = _patch_tracker(26)
    changed = image.copy()
    changed[74:76, 106:135] = 255 - changed[74:76, 106:135]
    result = tracker.update(changed, depth, K, WORLD_FROM_OPTICAL, exclusion_mask=mask)
    assert result["state"] == "tracking", result.get("reason")
    assert result["jaw_mask_patch_unmasked"] >= 140
    plain = VisualServoTracker()
    plain.initialize(image, [120, 80], depth)
    assert plain.update(changed, depth, K, WORLD_FROM_OPTICAL)["patch_correlation"] < result["patch_correlation"]


@pytest.mark.parametrize("masked_elements,occluded", [(30, True), (29, False)])
def test_fewer_than_140_unmasked_elements_is_a_latched_jaw_mask_occluded_state(masked_elements, occluded):
    tracker, image, depth, _ = _patch_tracker(masked_elements)
    empty = np.zeros(depth.shape, bool)
    result = tracker.update(image, depth, K, WORLD_FROM_OPTICAL, exclusion_mask=empty)
    assert result["jaw_mask_patch_unmasked"] == 169 - masked_elements
    if not occluded:
        assert result["state"] == "tracking"
        return
    assert result["state"] == "jaw_mask_occluded"
    assert result["reason"] == "too_few_unmasked_patch_pixels"
    assert result["target_position_world_m"] is None and result["requires_reinitialize"] is True
    assert "patch_correlation" not in result and result["jaw_mask_patch_correlation_kept"] == pytest.approx(1.0)
    assert "pending_feature_pixels_for_next_frame" not in result
    json.dumps(result, allow_nan=False)
    following = tracker.update(image, depth, K, WORLD_FROM_OPTICAL, exclusion_mask=empty)
    assert (following["state"], following["reason"]) == ("tracking_lost", "explicit_initialization_required")


@pytest.mark.parametrize("masked_elements,expected", [(40, "jaw_mask_occluded"), (29, "tracking_lost")])
def test_occluded_is_checked_before_the_correlation_threshold(masked_elements, expected):
    # Invert the lower rows of the patch: the kept-element correlation fails the 0.35 threshold (the 29-element
    # control shows it), yet with fewer than 140 kept elements the explicit occluded state is reported first.
    tracker, image, depth, _ = _patch_tracker(masked_elements)
    changed = image.copy()
    changed[77:87, 106:135] = 255 - changed[77:87, 106:135]
    result = tracker.update(changed, depth, K, WORLD_FROM_OPTICAL, exclusion_mask=np.zeros(depth.shape, bool))
    assert result["state"] == expected
    if expected == "tracking_lost":
        assert result["reason"] == "appearance_changed_or_occluded"
        assert result["patch_correlation"] < tracker.config.min_patch_correlation


def test_minimum_unmasked_elements_is_read_only_with_a_mask():
    config = VisualServoConfig(min_unmasked_patch_elements=169)
    tracker, image, depth, _ = _patch_tracker(1, config)
    assert tracker.update(image, depth, K, WORLD_FROM_OPTICAL, exclusion_mask=np.zeros(depth.shape, bool))["state"] == (
        "jaw_mask_occluded"
    )
    plain = VisualServoTracker(config)
    plain.initialize(image, [120, 80], depth)
    assert plain.update(image, depth, K, WORLD_FROM_OPTICAL)["state"] == "tracking"


def test_initial_and_replenished_corners_avoid_the_mask():
    image, depth = _scene()
    mask = np.zeros(depth.shape, bool)
    mask[54:106, 106:113] = True  # the left part of the ROI, clear of the appearance patch (columns 114-126)
    tracker = VisualServoTracker(VisualServoConfig(replenish_features=True))
    initialized = tracker.initialize(image, [120, 80], depth, exclusion_mask=mask)
    assert initialized["state"] == "initialized" and initialized["jaw_mask_roi_pixels"] > 0
    assert not any(mask[round(y), round(x)] for x, y in initialized["feature_pixels_xy"])
    plain = VisualServoTracker(VisualServoConfig(replenish_features=True))
    assert any(mask[round(y), round(x)] for x, y in plain.initialize(image, [120, 80], depth)["feature_pixels_xy"])
    # Force replenishment capacity with four anchors in the middle rows, then mask the top band of the ROI in the
    # next frame (away from the anchors and the appearance patch): the new corners must avoid that frame's mask.
    middle = [point for point in tracker._points.reshape(-1, 2) if 72 <= point[1] <= 88]
    assert len(middle) >= tracker.config.min_features
    tracker._points = np.asarray(middle[: tracker.config.min_features], dtype=np.float32).reshape(-1, 1, 2)
    tracker._initial_count = tracker.config.min_features * 4
    moved = np.zeros(depth.shape, bool)
    moved[54:71, 106:135] = True
    result = tracker.update(image, depth, K, WORLD_FROM_OPTICAL, exclusion_mask=moved)
    assert result["state"] == "tracking", result.get("reason")
    pending = result["pending_feature_pixels_for_next_frame"]
    assert pending and not any(moved[round(y), round(x)] for x, y in pending)
    unmasked = VisualServoTracker(VisualServoConfig(replenish_features=True))
    unmasked.initialize(image, [120, 80], depth)
    unmasked._points = tracker._points.copy()
    unmasked._initial_count = tracker._initial_count
    candidates = unmasked.update(image, depth, K, WORLD_FROM_OPTICAL)["pending_feature_pixels_for_next_frame"]
    assert any(moved[round(y), round(x)] for x, y in candidates)


def test_flow_failure_caused_only_by_the_mask_is_attributed():
    image, depth = _scene()
    tracker = VisualServoTracker()
    tracker.initialize(image, [120, 80], depth)
    everything = np.ones(depth.shape, bool)
    result = tracker.update(image, depth, K, WORLD_FROM_OPTICAL, exclusion_mask=everything)
    assert (result["state"], result["reason"]) == ("tracking_lost", "optical_flow_failed")
    assert result["flow_reason"] == "too_few_roundtrip_inliers" and result["roundtrip_inlier_count"] == 0
    assert result["jaw_mask_dropped_valid"] == result["jaw_mask_valid_before_mask"] > 0
    assert result["jaw_mask_unmasked_flow_would_pass"] is True
    assert result["jaw_mask_unmasked_accepted_count"] >= tracker.config.min_features


@pytest.mark.parametrize(
    "mask", [np.zeros((10, 10), bool), np.zeros((160, 240), np.uint8), np.zeros((160, 240, 1), bool), [[True]]]
)
def test_a_mask_that_does_not_cover_the_image_fails_closed(mask):
    image, depth = _scene()
    with pytest.raises(ValueError, match="exclusion_mask"):
        VisualServoTracker().initialize(image, [120, 80], depth, exclusion_mask=mask)
    tracker = _tracker()
    frame_index = tracker._frame_index
    with pytest.raises(ValueError, match="exclusion_mask"):
        tracker.update(image, depth, K, WORLD_FROM_OPTICAL, exclusion_mask=mask)
    assert tracker._frame_index == frame_index
    assert tracker.update(image, depth, K, WORLD_FROM_OPTICAL)["state"] == "tracking"


@pytest.mark.parametrize("value", [0, 170, -1, True, 1.5, "140", None])
def test_minimum_unmasked_patch_elements_is_validated(value):
    with pytest.raises(ValueError, match="min_unmasked_patch_elements"):
        VisualServoConfig(min_unmasked_patch_elements=value)


def test_minimum_unmasked_patch_elements_defaults_to_the_registration():
    from isaaclab_pruning.perception.jaw_self_mask import MIN_UNMASKED_PATCH_ELEMENTS

    assert VisualServoConfig().min_unmasked_patch_elements == MIN_UNMASKED_PATCH_ELEMENTS == 140
    assert VisualServoConfig(min_unmasked_patch_elements=1).min_unmasked_patch_elements == 1
    assert VisualServoConfig(min_unmasked_patch_elements=169).min_unmasked_patch_elements == 169
