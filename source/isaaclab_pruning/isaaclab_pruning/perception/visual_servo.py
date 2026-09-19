"""Classical, seeded RGB tracking with measured depth for a visual servo loop.

This is not a branch detector or a learned policy. A person or scene metadata
selects ONE initial pixel. Subsequent measurements use only consecutive RGB
images, optical-Z depth, and the current camera calibration. A lost track never
silently re-seeds from scene coordinates or returns a stale world position.

Camera coordinates follow OpenCV: +X right, +Y down, +Z forward. ``depth`` must
be ``distance_to_image_plane`` in metres, NOT distance along the viewing ray.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VisualServoConfig:
    roi_half_size_px: tuple[int, int] = (14, 24)
    max_features: int = 80
    min_features: int = 4
    feature_quality_level: float = 0.02
    replenish_features: bool = False
    photometric_normalization: str = "raw"
    max_roundtrip_error_px: float = 1.0
    max_lk_error: float = 40.0
    max_flow_residual_px: float = 3.0
    min_patch_std: float = 3.0
    min_patch_correlation: float = 0.35
    min_confidence: float = 0.15
    depth_radius_px: int = 2
    min_depth_valid_fraction: float = 0.75
    min_depth_m: float = 0.02
    max_depth_m: float = 5.0
    max_depth_spread_m: float = 0.025
    max_world_jump_m: float = 0.06

    def __post_init__(self):
        if self.photometric_normalization not in ("raw", "clahe"):
            raise ValueError("photometric_normalization must be raw or clahe")
        if not isinstance(self.replenish_features, bool):
            raise ValueError("replenish_features must be bool")
        if (
            isinstance(self.feature_quality_level, (bool, np.bool_))
            or not isinstance(self.feature_quality_level, (int, float, np.integer, np.floating))
            or not np.isfinite(self.feature_quality_level)
            or not 0 < self.feature_quality_level <= 1
        ):
            raise ValueError("feature_quality_level must be finite and in (0, 1], not bool")
        if self.min_features < 3 or self.max_features < self.min_features:
            raise ValueError("Require 3 <= min_features <= max_features")
        if len(self.roi_half_size_px) != 2 or min(self.roi_half_size_px) < 3:
            raise ValueError("roi_half_size_px must contain two radii >= 3")
        if self.depth_radius_px < 0:
            raise ValueError("depth_radius_px must be nonnegative")
        for name in ("min_confidence", "min_patch_correlation", "min_depth_valid_fraction"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        for name in (
            "max_roundtrip_error_px",
            "max_lk_error",
            "max_flow_residual_px",
            "min_patch_std",
            "min_depth_m",
            "max_depth_spread_m",
            "max_world_jump_m",
        ):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if not np.isfinite(self.max_depth_m) or self.max_depth_m <= self.min_depth_m:
            raise ValueError("max_depth_m must exceed min_depth_m")


def _gray(rgb):
    import cv2

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] not in (3, 4) or rgb.dtype != np.uint8:
        raise ValueError("rgb must be an H x W x 3 (or 4) uint8 RGB image")
    return cv2.cvtColor(np.ascontiguousarray(rgb[:, :, :3]), cv2.COLOR_RGB2GRAY)


def project_depth_to_world(pixel_xy, depth_m, camera_matrix, world_from_optical):
    """Back-project one optical-Z sample, validating the rigid camera transform."""
    pixel = np.asarray(pixel_xy, dtype=float)
    matrix = np.asarray(camera_matrix, dtype=float)
    transform = np.asarray(world_from_optical, dtype=float)
    if pixel.shape != (2,) or not np.isfinite(pixel).all() or not np.isfinite(depth_m) or depth_m <= 0:
        raise ValueError("pixel and positive depth must be finite")
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera_matrix must be finite 3 x 3")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0 or not np.allclose(matrix[2], [0, 0, 1]):
        raise ValueError("camera_matrix must be a pinhole calibration with positive focal lengths")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("world_from_optical must be finite 4 x 4")
    rotation = transform[:3, :3]
    if (
        not np.allclose(transform[3], [0, 0, 0, 1])
        or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5)
        or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-5)
    ):
        raise ValueError("world_from_optical must be a rigid proper rotation, not a reflection or scale")
    ray = np.linalg.solve(matrix, [*pixel, 1.0])
    optical = ray * (float(depth_m) / ray[2])
    return rotation @ optical + transform[:3, 3]


def bounded_visual_servo_translation(
    measurement,
    current_position_world_m,
    *,
    approach_axis_world,
    standoff_m,
    now_s,
    measurement_time_s,
    max_step_m=0.004,
    max_age_s=0.25,
    hazard_contact=False,
):
    """Move a measured mouth point toward a fresh RGB-D target, or hold.

    The desired mouth point is ``target - standoff * approach_axis_world``.
    The axis points from the robot toward the branch; it must not be recomputed
    from an oracle target after initialization. The caller adds ``delta_world_m``
    to the measured tool position and retains the measured orientation. This is
    a capped position increment, not collision checking or trajectory planning.
    Timestamp freshness and hazard checks do not substitute for those systems.
    No prior target is cached or returned on a missing perception measurement.
    """
    current = np.asarray(current_position_world_m, dtype=float)
    axis = np.asarray(approach_axis_world, dtype=float)
    if current.shape != (3,) or not np.isfinite(current).all():
        raise ValueError("current_position_world_m must contain three finite values")
    if axis.shape != (3,) or not np.isfinite(axis).all() or np.linalg.norm(axis) <= 1e-12:
        raise ValueError("approach_axis_world must contain three finite, nonzero values")
    if not np.isfinite(standoff_m) or standoff_m < 0:
        raise ValueError("standoff_m must be finite and nonnegative")
    if not np.isfinite(max_step_m) or max_step_m <= 0 or not np.isfinite(max_age_s) or max_age_s <= 0:
        raise ValueError("max_step_m and max_age_s must be finite and positive")
    result = {
        "state": "hold",
        "reason": None,
        "command_position_world_m": current.tolist(),
        "delta_world_m": [0.0, 0.0, 0.0],
        "remaining_distance_m": None,
        "target_position_world_m": None,
    }
    if hazard_contact:
        return {**result, "reason": "hazard_contact"}
    if not isinstance(measurement, dict) or measurement.get("state") != "tracking":
        return {**result, "reason": "vision_not_tracking"}
    if (
        not np.isfinite(now_s)
        or not np.isfinite(measurement_time_s)
        or not 0 <= now_s - measurement_time_s <= max_age_s
    ):
        return {**result, "reason": "vision_stale_or_future"}
    try:
        target = np.asarray(measurement.get("target_position_world_m"), dtype=float)
    except (ValueError, TypeError):
        return {**result, "reason": "invalid_target"}
    if target.shape != (3,) or not np.isfinite(target).all():
        return {**result, "reason": "invalid_target"}
    error = target - standoff_m * axis / np.linalg.norm(axis) - current
    distance = float(np.linalg.norm(error))
    delta = error * min(1.0, max_step_m / distance) if distance > 0 else np.zeros(3)
    return {
        **result,
        "state": "tracking",
        "command_position_world_m": (current + delta).tolist(),
        "delta_world_m": delta.tolist(),
        "remaining_distance_m": distance,
        "target_position_world_m": target.tolist(),
    }


class VisualServoTracker:
    """Forward/backward pyramidal LK, local appearance checks, and depth gates.

    ``initialize`` may use scene metadata ONCE to choose the initial pixel; record
    that provenance in the caller. Supply initial depth to exclude feature points
    on a different surface. Use a narrow ROI for a thin branch. A textured branch
    is required: a uniform cylinder is not reliably trackable with optical flow.
    ``update`` returns a JSON-safe dict; only state ``tracking`` permits motion.
    Every other state returns ``target_position_world_m=None``. Tracking loss is
    latched until explicit initialization; invalid depth can recover on the next
    frame, but never authorizes motion in the missing-data interval.
    """

    def __init__(self, config: VisualServoConfig | None = None):
        self.config = config or VisualServoConfig()
        self._previous_gray = None
        self._points = None
        self._pixel = None
        self._initial_count = 0
        self._last_world = None
        self._lost = True
        self._frame_index = 0
        # Fixed, explicitly experimental preprocessing; depth and all gates are
        # unchanged. CLAHE can amplify noise and is not an automatic fallback.
        self._clahe = None
        if self.config.photometric_normalization == "clahe":
            import cv2

            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _prepare_gray(self, rgb):
        gray = _gray(rgb)
        return gray if self._clahe is None else self._clahe.apply(gray)

    def _result(self, state, reason=None, **details):
        return {
            "state": state,
            "reason": reason,
            "target_position_world_m": None,
            "pixel_xy": None if self._pixel is None else self._pixel.astype(float).tolist(),
            "feature_pixels_xy": [] if self._points is None else self._points.reshape(-1, 2).astype(float).tolist(),
            "feature_count": 0 if self._points is None else len(self._points),
            "initial_feature_count": self._initial_count,
            "initial_feature_config": {
                "quality_level": float(self.config.feature_quality_level),
                "min_distance_px": 3,
                "block_size_px": 3,
                "roi_half_size_px": list(self.config.roi_half_size_px),
            },
            "confidence": 0.0,
            "frame_index": self._frame_index,
            "requires_reinitialize": self._lost,
            "method": "seeded_forward_backward_lk_optical_z",
            "feature_maintenance_enabled": self.config.replenish_features,
            "photometric_normalization": {
                "mode": self.config.photometric_normalization,
                "clahe_clip_limit": None if self._clahe is None else 2.0,
                "clahe_tile_grid_size": None if self._clahe is None else [8, 8],
                "input": "uint8_RGB_to_gray",
            },
            **details,
        }

    def _lose(self, reason, **details):
        self._lost = True
        return self._result("tracking_lost", reason, **details)

    def _depth_sample(self, depth, pixel):
        depth = np.asarray(depth)
        if depth.ndim == 3 and depth.shape[2] == 1:
            depth = depth[:, :, 0]
        if depth.shape != self._previous_gray.shape:
            return None, {"depth_reason": "shape_mismatch", "depth_valid_fraction": 0.0}
        x, y = np.rint(pixel).astype(int)
        radius = self.config.depth_radius_px
        if x - radius < 0 or y - radius < 0 or x + radius >= depth.shape[1] or y + radius >= depth.shape[0]:
            return None, {"depth_reason": "window_outside_image", "depth_valid_fraction": 0.0}
        values = depth[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(float)
        valid = np.isfinite(values) & (values >= self.config.min_depth_m) & (values <= self.config.max_depth_m)
        fraction = float(np.mean(valid))
        details = {"depth_valid_fraction": fraction, "depth_window_size_px": int(values.size)}
        if fraction < self.config.min_depth_valid_fraction:
            return None, {**details, "depth_reason": "insufficient_valid_samples"}
        selected = values[valid]
        spread = float(np.percentile(selected, 90) - np.percentile(selected, 10))
        details["depth_spread_m"] = spread
        if spread > self.config.max_depth_spread_m:
            return None, {**details, "depth_reason": "mixed_surfaces"}
        # The central target pixel must agree with the window: a background
        # majority must not replace an invalid or nearer narrow branch pixel.
        center = float(values[radius, radius])
        median = float(np.median(selected))
        if not valid[radius, radius] or abs(center - median) > self.config.max_depth_spread_m:
            return None, {**details, "depth_reason": "invalid_or_inconsistent_center"}
        return median, {**details, "depth_reason": None, "depth_m": median}

    def initialize(self, rgb, pixel_xy, depth=None):
        """Select an initial target once; does not itself authorize any motion."""
        import cv2

        gray = self._prepare_gray(rgb)
        pixel = np.asarray(pixel_xy, dtype=np.float32)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("pixel_xy must contain two finite coordinates")
        height, width = gray.shape
        if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
            raise ValueError("initial pixel is outside image")
        self._previous_gray, self._pixel = gray, pixel
        self._points, self._last_world = None, None
        self._initial_count, self._frame_index, self._lost = 0, 0, True
        half_x, half_y = self.config.roi_half_size_px
        x, y = np.rint(pixel).astype(int)
        mask = np.zeros_like(gray)
        mask[max(0, y - half_y) : min(height, y + half_y + 1), max(0, x - half_x) : min(width, x + half_x + 1)] = 255
        if depth is not None:
            initial_z, details = self._depth_sample(depth, pixel)
            if initial_z is None:
                return self._result("initialization_failed", "invalid_initial_depth", **details)
            depth_array = np.asarray(depth).reshape(gray.shape)
            same_surface = np.isfinite(depth_array) & (
                np.abs(depth_array - initial_z) <= self.config.max_depth_spread_m
            )
            mask[~same_surface] = 0
        self._points = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.config.max_features,
            qualityLevel=self.config.feature_quality_level,
            minDistance=3,
            mask=mask,
            blockSize=3,
        )
        self._initial_count = 0 if self._points is None else len(self._points)
        if self._initial_count < self.config.min_features:
            return self._result("initialization_failed", "insufficient_texture")
        self._lost = False
        return self._result("initialized", "awaiting_rgb_depth_measurement")

    def _track_points(self, gray):
        import cv2

        options = {
            "winSize": (15, 15),
            "maxLevel": 3,
            "criteria": (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        }
        forward, status, errors = cv2.calcOpticalFlowPyrLK(self._previous_gray, gray, self._points, None, **options)
        if forward is None or status is None or errors is None:
            return None, None, {"flow_reason": "forward_lk_failed"}
        backward, back_status, _ = cv2.calcOpticalFlowPyrLK(gray, self._previous_gray, forward, None, **options)
        if backward is None or back_status is None:
            return None, None, {"flow_reason": "backward_lk_failed"}
        old, new = self._points.reshape(-1, 2), forward.reshape(-1, 2)
        roundtrip = np.linalg.norm(old - backward.reshape(-1, 2), axis=1)
        height, width = gray.shape
        valid = (
            status.reshape(-1).astype(bool)
            & back_status.reshape(-1).astype(bool)
            & np.isfinite(new).all(axis=1)
            & np.isfinite(roundtrip)
            & (roundtrip <= self.config.max_roundtrip_error_px)
            & np.isfinite(errors.reshape(-1))
            & (errors.reshape(-1) <= self.config.max_lk_error)
            & (new[:, 0] >= 0)
            & (new[:, 0] < width)
            & (new[:, 1] >= 0)
            & (new[:, 1] < height)
        )
        if np.count_nonzero(valid) < self.config.min_features:
            return None, None, {"flow_reason": "too_few_roundtrip_inliers", "roundtrip_inlier_count": int(valid.sum())}
        flow = new - old
        median_flow = np.median(flow[valid], axis=0)
        valid &= np.linalg.norm(flow - median_flow, axis=1) <= self.config.max_flow_residual_px
        details = {"roundtrip_inlier_count": int(valid.sum())}
        if np.count_nonzero(valid) < self.config.min_features:
            return None, None, {**details, "flow_reason": "incoherent_motion"}
        details["median_roundtrip_error_px"] = float(np.median(roundtrip[valid]))
        return new[valid].reshape(-1, 1, 2), np.median(flow[valid], axis=0), details

    def _appearance(self, gray, next_pixel):
        import cv2

        previous = cv2.getRectSubPix(self._previous_gray.astype(np.float32), (13, 13), tuple(self._pixel))
        current = cv2.getRectSubPix(gray.astype(np.float32), (13, 13), tuple(next_pixel))
        old_std, new_std = float(previous.std()), float(current.std())
        if min(old_std, new_std) < self.config.min_patch_std:
            return 0.0
        correlation = np.mean((previous - previous.mean()) * (current - current.mean())) / (old_std * new_std)
        return float(np.clip(correlation, -1.0, 1.0))

    def _replenish_for_next_frame(self, depth, depth_m):
        """Add local same-depth corners AFTER a valid measurement, never re-seed.

        Current motion/confidence cannot use these new points. They must pass
        the normal forward/backward, coherence and appearance gates next frame.
        Existing matched anchors always comprise at least half the next set.
        """
        import cv2

        capacity = min(self._initial_count, self.config.max_features) - len(self._points)
        capacity = min(capacity, len(self._points))
        if capacity <= 0:
            return []
        gray = self._previous_gray
        height, width = gray.shape
        x, y = np.rint(self._pixel).astype(int)
        half_x, half_y = self.config.roi_half_size_px
        mask = np.zeros_like(gray)
        mask[max(0, y - half_y) : min(height, y + half_y + 1), max(0, x - half_x) : min(width, x + half_x + 1)] = 255
        values = np.asarray(depth).reshape(gray.shape)
        mask[~np.isfinite(values) | (np.abs(values - depth_m) > self.config.max_depth_spread_m)] = 0
        for point in self._points.reshape(-1, 2):
            cv2.circle(mask, tuple(np.rint(point).astype(int)), 3, 0, -1)
        candidates = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=capacity,
            qualityLevel=self.config.feature_quality_level,
            minDistance=3,
            mask=mask,
            blockSize=3,
        )
        if candidates is None:
            return []
        self._points = np.concatenate((self._points, candidates))
        return candidates.reshape(-1, 2).astype(float).tolist()

    def update(self, rgb, depth, camera_matrix, world_from_optical):
        """Measure the tracked target; non-tracking results require a hold/stop."""
        self._frame_index += 1
        if self._lost or self._previous_gray is None:
            return self._result("tracking_lost", "explicit_initialization_required")
        gray = self._prepare_gray(rgb)
        if gray.shape != self._previous_gray.shape:
            return self._lose("image_shape_changed")
        points, flow, details = self._track_points(gray)
        if points is None:
            return self._lose("optical_flow_failed", **details)
        next_pixel = self._pixel + flow
        if not (0 <= next_pixel[0] < gray.shape[1] and 0 <= next_pixel[1] < gray.shape[0]):
            return self._lose("target_outside_image", **details)
        correlation = self._appearance(gray, next_pixel)
        details["patch_correlation"] = correlation
        if correlation < self.config.min_patch_correlation:
            return self._lose("appearance_changed_or_occluded", **details)
        self._previous_gray, self._points, self._pixel = gray, points, next_pixel
        depth_m, depth_details = self._depth_sample(depth, self._pixel)
        details.update(depth_details)
        if depth_m is None:
            return self._result("invalid_depth", "depth_measurement_rejected", **details)
        try:
            world = project_depth_to_world(self._pixel, depth_m, camera_matrix, world_from_optical)
        except (ValueError, np.linalg.LinAlgError) as exc:
            return self._result("invalid_calibration", str(exc), **details)
        if self._last_world is not None:
            details["world_jump_m"] = float(np.linalg.norm(world - self._last_world))
            if details["world_jump_m"] > self.config.max_world_jump_m:
                return self._lose("world_target_jump_or_wrong_surface", **details)
        confidence = float(
            min(1.0, len(points) / self._initial_count) * max(0.0, correlation) * details["depth_valid_fraction"]
        )
        if confidence < self.config.min_confidence:
            return self._lose("low_confidence", measured_confidence=confidence, **details)
        self._last_world = world
        result = self._result("tracking", target_position_world_m=world.tolist(), confidence=confidence, **details)
        if self.config.replenish_features:
            # Snapshot the current validated points before adding candidates:
            # telemetry must not count unvalidated corners as tracking evidence.
            result["pending_feature_pixels_for_next_frame"] = self._replenish_for_next_frame(depth, depth_m)
        return result
