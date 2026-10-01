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

MOTION_MODELS = ("translation", "similarity")
#: Side of the square appearance patch compared by normalized cross-correlation.
PATCH_SIZE_PX = 13
PATCH_ELEMENTS = PATCH_SIZE_PX * PATCH_SIZE_PX
#: Latched state when too few appearance-patch elements lie outside the robot's own self-pixel mask.
JAW_MASK_OCCLUDED_STATE = "jaw_mask_occluded"
JAW_MASK_OCCLUDED_REASON = "too_few_unmasked_patch_pixels"


def propagate_pixel(pixel_xy, old_points, new_points, motion_model="translation"):
    """Next tracked pixel from accepted feature pairs, and what was applied.

    ``translation`` adds the median flow. ``similarity`` fits a 4-DOF
    similarity (``cv2.estimateAffinePartial2D`` with LMedS) to the same pairs
    and maps the pixel through it, falling back to the median flow when no
    finite model is returned. Both use only features that already passed the
    tracker's unchanged roundtrip, LK-error and flow-residual checks.
    """
    import cv2

    pixel = np.asarray(pixel_xy, dtype=np.float64).reshape(2)
    old = np.asarray(old_points, dtype=np.float64).reshape(-1, 2)
    new = np.asarray(new_points, dtype=np.float64).reshape(-1, 2)
    if old.shape != new.shape or len(old) == 0:
        raise ValueError("feature pairs must be non-empty and matched")
    median_flow = np.median(new - old, axis=0)
    if motion_model == "translation":
        return pixel + median_flow, {"motion_model": "translation"}
    if motion_model != "similarity":
        raise ValueError(f"motion_model must be one of {MOTION_MODELS}")
    model, _ = cv2.estimateAffinePartial2D(old.astype(np.float32), new.astype(np.float32), method=cv2.LMEDS)
    if model is None or not np.isfinite(model).all():
        return pixel + median_flow, {"motion_model": "similarity", "similarity_fallback": True}
    scale = float(np.sqrt(abs(np.linalg.det(model[:, :2]))))
    return model[:, :2] @ pixel + model[:, 2], {
        "motion_model": "similarity",
        "similarity_fallback": False,
        "similarity_scale": scale,
    }


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
    # How the tracked pixel follows its accepted features. "translation" (the
    # default, unchanged) moves it by their median flow; "similarity" fits
    # scale, rotation and translation to the same accepted features, which
    # removes the sideways drift a median flow shows under approach zoom when
    # the features sit to one side of the tracked point. No gate reads it.
    motion_model: str = "translation"
    # Read only when the caller supplies a robot self-pixel exclusion mask: the
    # fewest of the 169 appearance-patch elements that must be unmasked in both
    # patches before the correlation may be judged (else jaw_mask_occluded).
    min_unmasked_patch_elements: int = 140

    def __post_init__(self):
        if self.photometric_normalization not in ("raw", "clahe"):
            raise ValueError("photometric_normalization must be raw or clahe")
        if (
            isinstance(self.min_unmasked_patch_elements, (bool, np.bool_))
            or not isinstance(self.min_unmasked_patch_elements, (int, np.integer))
            or not 1 <= self.min_unmasked_patch_elements <= PATCH_ELEMENTS
        ):
            raise ValueError(f"min_unmasked_patch_elements must be an integer in 1..{PATCH_ELEMENTS}, not bool")
        if self.motion_model not in MOTION_MODELS:
            raise ValueError(f"motion_model must be one of {MOTION_MODELS}")
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


def _points_in_mask(mask, points):
    """Whether each point's nearest pixel (``np.rint``) lies inside the bool mask; off-image points do not."""
    pts = np.asarray(points, float).reshape(-1, 2)
    out = np.zeros(len(pts), bool)
    if mask is None or len(pts) == 0:
        return out
    finite = np.isfinite(pts).all(axis=1)
    xy = np.full((len(pts), 2), -1, int)
    xy[finite] = np.rint(pts[finite]).astype(int)
    height, width = mask.shape
    inside = finite & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    out[inside] = mask[xy[inside, 1], xy[inside, 0]]
    return out


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

    ``exclusion_mask`` (optional, bool H x W, True = the robot's own pixels) marks
    pixels that are never evidence: an LK feature whose tracked position (rounded)
    lies in this frame's mask, or whose previous position lay in the previous
    frame's mask, is dropped before the feature count, median flow and residual
    filter; masked pixels are excluded from initial and replenished corners; and
    the appearance correlation uses only patch elements unmasked in both patches,
    with the latched state ``jaw_mask_occluded`` when fewer than
    ``min_unmasked_patch_elements`` remain (checked before the correlation
    threshold). ``jaw_mask_*`` keys are attribution telemetry, never a decision.
    Without a mask every output is exactly what it was before masks existed.
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
        self._accepted_pairs = None
        # Exclusion masks of the frame held in _previous_gray and of the frame being measured.
        self._mask_previous = None
        self._mask_current = None
        # Fixed, explicitly experimental preprocessing; depth and all gates are
        # unchanged. CLAHE can amplify noise and is not an automatic fallback.
        self._clahe = None
        if self.config.photometric_normalization == "clahe":
            import cv2

            self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def _prepare_gray(self, rgb):
        gray = _gray(rgb)
        return gray if self._clahe is None else self._clahe.apply(gray)

    @staticmethod
    def _exclusion_mask(mask, shape):
        """None, or the caller's bool mask after checking it covers exactly the image (fail closed)."""
        if mask is None:
            return None
        mask = np.asarray(mask)
        if mask.dtype != bool or mask.ndim != 2:
            raise ValueError("exclusion_mask must be a boolean H x W array (True = robot self-pixel)")
        if mask.shape != tuple(shape):
            raise ValueError(f"exclusion_mask shape {mask.shape} differs from the image {tuple(shape)}")
        return mask

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

    def initialize(self, rgb, pixel_xy, depth=None, exclusion_mask=None):
        """Select an initial target once; does not itself authorize any motion.

        ``exclusion_mask`` (bool H x W, True = robot self-pixel) keeps initial
        corners off the robot's own pixels; None keeps the original behaviour.
        """
        import cv2

        gray = self._prepare_gray(rgb)
        excluded = self._exclusion_mask(exclusion_mask, gray.shape)
        pixel = np.asarray(pixel_xy, dtype=np.float32)
        if pixel.shape != (2,) or not np.isfinite(pixel).all():
            raise ValueError("pixel_xy must contain two finite coordinates")
        height, width = gray.shape
        if not (0 <= pixel[0] < width and 0 <= pixel[1] < height):
            raise ValueError("initial pixel is outside image")
        self._previous_gray, self._pixel = gray, pixel
        self._points, self._last_world = None, None
        self._initial_count, self._frame_index, self._lost = 0, 0, True
        self._mask_previous, self._mask_current = excluded, None
        half_x, half_y = self.config.roi_half_size_px
        x, y = np.rint(pixel).astype(int)
        mask = np.zeros_like(gray)
        mask[max(0, y - half_y) : min(height, y + half_y + 1), max(0, x - half_x) : min(width, x + half_x + 1)] = 255
        mask_details = {}
        if excluded is not None:
            mask_details["jaw_mask_roi_pixels"] = int(np.count_nonzero((mask > 0) & excluded))
        if depth is not None:
            initial_z, details = self._depth_sample(depth, pixel)
            if initial_z is None:
                return self._result("initialization_failed", "invalid_initial_depth", **details, **mask_details)
            depth_array = np.asarray(depth).reshape(gray.shape)
            same_surface = np.isfinite(depth_array) & (
                np.abs(depth_array - initial_z) <= self.config.max_depth_spread_m
            )
            mask[~same_surface] = 0
        if excluded is not None:
            mask[excluded] = 0
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
            return self._result("initialization_failed", "insufficient_texture", **mask_details)
        self._lost = False
        return self._result("initialized", "awaiting_rgb_depth_measurement", **mask_details)

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
        mask_details = {}
        if self._mask_current is not None:
            # A feature on the robot's own pixels (now, or in the previous frame) never votes on the flow.
            in_mask = _points_in_mask(self._mask_current, new) | _points_in_mask(self._mask_previous, old)
            would_pass, would_accept = self._unmasked_flow_outcome(valid, old, new)
            mask_details = {
                "jaw_mask_features_in": int(len(old)),
                "jaw_mask_valid_before_mask": int(valid.sum()),
                "jaw_mask_dropped_valid": int((valid & in_mask).sum()),
                "jaw_mask_in_mask_any": int(in_mask.sum()),
                "jaw_mask_unmasked_flow_would_pass": would_pass,
                "jaw_mask_unmasked_accepted_count": would_accept,
            }
            valid = valid & ~in_mask
        if np.count_nonzero(valid) < self.config.min_features:
            return (
                None,
                None,
                {
                    "flow_reason": "too_few_roundtrip_inliers",
                    "roundtrip_inlier_count": int(valid.sum()),
                    **mask_details,
                },
            )
        flow = new - old
        median_flow = np.median(flow[valid], axis=0)
        valid &= np.linalg.norm(flow - median_flow, axis=1) <= self.config.max_flow_residual_px
        details = {"roundtrip_inlier_count": int(valid.sum()), **mask_details}
        if np.count_nonzero(valid) < self.config.min_features:
            return None, None, {**details, "flow_reason": "incoherent_motion"}
        details["median_roundtrip_error_px"] = float(np.median(roundtrip[valid]))
        self._accepted_pairs = (old[valid], new[valid])
        return new[valid].reshape(-1, 1, 2), np.median(flow[valid], axis=0), details

    def _unmasked_flow_outcome(self, valid, old, new):
        """Attribution only: whether the same LK output without the mask passes the flow checks, and its count."""
        if np.count_nonzero(valid) < self.config.min_features:
            return False, int(valid.sum())
        flow = new - old
        median_flow = np.median(flow[valid], axis=0)
        accepted = valid & (np.linalg.norm(flow - median_flow, axis=1) <= self.config.max_flow_residual_px)
        return bool(np.count_nonzero(accepted) >= self.config.min_features), int(accepted.sum())

    def _appearance(self, gray, next_pixel):
        import cv2

        previous = cv2.getRectSubPix(self._previous_gray.astype(np.float32), (13, 13), tuple(self._pixel))
        current = cv2.getRectSubPix(gray.astype(np.float32), (13, 13), tuple(next_pixel))
        old_std, new_std = float(previous.std()), float(current.std())
        if min(old_std, new_std) < self.config.min_patch_std:
            return 0.0
        correlation = np.mean((previous - previous.mean()) * (current - current.mean())) / (old_std * new_std)
        return float(np.clip(correlation, -1.0, 1.0))

    def _masked_appearance(self, gray, next_pixel):
        """Correlation over patch elements unmasked in both patches (None when none remain), and telemetry.

        The masks are sampled with the patch's own ``getRectSubPix`` footprint and
        any nonzero weight counts as masked. ``min_patch_std`` applies to the kept
        elements; with nothing masked the full-patch formula runs unchanged.
        """
        import cv2

        size = (PATCH_SIZE_PX, PATCH_SIZE_PX)
        previous_masked = cv2.getRectSubPix(self._mask_previous.astype(np.float32), size, tuple(self._pixel)) > 0
        current_masked = cv2.getRectSubPix(self._mask_current.astype(np.float32), size, tuple(next_pixel)) > 0
        keep = ~(previous_masked | current_masked)
        kept = int(keep.sum())
        info = {
            "jaw_mask_patch_unmasked": kept,
            "jaw_mask_patch_masked_prev": int(previous_masked.sum()),
            "jaw_mask_patch_masked_cur": int(current_masked.sum()),
        }
        if kept == PATCH_ELEMENTS:
            return self._appearance(gray, next_pixel), info
        if kept == 0:
            return None, info
        previous = cv2.getRectSubPix(self._previous_gray.astype(np.float32), size, tuple(self._pixel))[keep]
        current = cv2.getRectSubPix(gray.astype(np.float32), size, tuple(next_pixel))[keep]
        old_std, new_std = float(previous.std()), float(current.std())
        info["jaw_mask_kept_std"] = [old_std, new_std]
        if min(old_std, new_std) < self.config.min_patch_std:
            return 0.0, info
        correlation = np.mean((previous - previous.mean()) * (current - current.mean())) / (old_std * new_std)
        return float(np.clip(correlation, -1.0, 1.0)), info

    def _mask_pixel_flags(self, pixel, mask):
        """Telemetry: whether a pixel, or the depth window around it, touches the mask."""
        x, y = np.rint(pixel).astype(int)
        radius = self.config.depth_radius_px
        height, width = mask.shape
        window = mask[max(0, y - radius) : min(height, y + radius + 1), max(0, x - radius) : min(width, x + radius + 1)]
        inside = 0 <= x < width and 0 <= y < height and bool(mask[y, x])
        return {"jaw_mask_pixel_in_mask": inside, "jaw_mask_depth_window_in_mask": bool(window.any())}

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
        if self._mask_previous is not None:
            mask[self._mask_previous] = 0  # this frame's self-pixels (previous_gray is this frame here)
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

    def update(self, rgb, depth, camera_matrix, world_from_optical, exclusion_mask=None):
        """Measure the tracked target; non-tracking results require a hold/stop.

        ``exclusion_mask`` is this frame's robot self-pixel mask (see the class
        docstring); None gives exactly the unmasked tracker.
        """
        excluded = self._exclusion_mask(exclusion_mask, np.shape(rgb)[:2]) if exclusion_mask is not None else None
        self._frame_index += 1
        if self._lost or self._previous_gray is None:
            return self._result("tracking_lost", "explicit_initialization_required")
        gray = self._prepare_gray(rgb)
        if gray.shape != self._previous_gray.shape:
            return self._lose("image_shape_changed")
        self._mask_current = excluded
        if excluded is not None and self._mask_previous is None:
            self._mask_previous = np.zeros(gray.shape, bool)
        points, flow, details = self._track_points(gray)
        if points is None:
            return self._lose("optical_flow_failed", **details)
        if self.config.motion_model == "translation":
            next_pixel = self._pixel + flow
        else:
            next_pixel, applied = propagate_pixel(self._pixel, *self._accepted_pairs, self.config.motion_model)
            next_pixel = np.asarray(next_pixel, dtype=np.float32)
            details.update(applied)
        if not (0 <= next_pixel[0] < gray.shape[1] and 0 <= next_pixel[1] < gray.shape[0]):
            return self._lose("target_outside_image", **details)
        if excluded is None:
            correlation = self._appearance(gray, next_pixel)
        else:
            correlation, patch_details = self._masked_appearance(gray, next_pixel)
            details.update(patch_details)
            details.update(
                {f"{key}_propagated": value for key, value in self._mask_pixel_flags(next_pixel, excluded).items()}
            )
            if patch_details["jaw_mask_patch_unmasked"] < self.config.min_unmasked_patch_elements:
                # Too little of the patch is the branch: no appearance judgement, no target, latched.
                self._lost = True
                details["jaw_mask_patch_correlation_kept"] = correlation
                return self._result(JAW_MASK_OCCLUDED_STATE, JAW_MASK_OCCLUDED_REASON, **details)
        details["patch_correlation"] = correlation
        if correlation < self.config.min_patch_correlation:
            return self._lose("appearance_changed_or_occluded", **details)
        self._previous_gray, self._points, self._pixel = gray, points, next_pixel
        self._mask_previous = excluded
        if excluded is not None:
            details.update(self._mask_pixel_flags(self._pixel, excluded))
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
        if excluded is not None and "jaw_mask_unmasked_accepted_count" in details:
            # Attribution only: the same confidence with the feature count the unmasked flow would have accepted.
            details["jaw_mask_confidence_with_unmasked_count"] = float(
                min(1.0, details["jaw_mask_unmasked_accepted_count"] / self._initial_count)
                * max(0.0, correlation)
                * details["depth_valid_fraction"]
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
