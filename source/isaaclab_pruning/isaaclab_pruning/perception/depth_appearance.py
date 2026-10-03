"""Depth-aware appearance check D_strict + J: a static-world depth test in front of the tracker's 0.35 gate.

Registered in docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_HELDOUT_2026-10-01.md (held-out offline replay, H1-H4
supported) and run in closed loop under docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_CLOSED_LOOP_2026-10-01.md.

``DepthAppearanceTracker`` is the repository ``VisualServoTracker`` with one change. When the grey NCC
appearance check fails (below ``min_patch_correlation``), it asks whether the previous 13 x 13 patch's surface
stayed where a static world puts it in the current frame, and whether the robot's own jaw silhouette touches
the patch. If both hold, the 0.35 gate passes; in the strict arm the raw NCC stays in the unchanged confidence
``min(1, n/n0) * NCC * depth_valid_fraction >= 0.15``. Every other check is the repository code, called
unchanged. While the grey check passes, every result is exactly the base tracker's.

The test (registered thresholds; code constants, nothing reachable from the environment):
  * back-project the previous patch rint(p) + [-6..6]^2 at the previous frame's optical-Z and pose (pixel
    index u has its centre at u + 0.5) and reproject it with the current pose; both depths come from the
    3 x 3 median rule (median of the valid samples, invalid below 75% valid; no spread or centre check);
  * keep the same-surface elements, within 25 mm of the tracked depth (the median of the tracker's own depth
    window at p, previous frame);
  * accept only if (1) >= 75% of the same-surface elements are verified (in the image with a valid current
    depth), (2) >= 20 same-surface elements remain, (3) <= 5% of the verified elements are nearer than
    predicted by more than 10 mm, (4) their median |measured - predicted| is <= 3 mm, (5) the propagated pixel
    is within 3 px of the static-world prediction of p, and (J) the committed 2.0 px jaw silhouette
    (``jaw_self_mask.jaw_mask``) touches none of the previous, reprojected or propagated patches. J fails
    closed when the silhouette is undefined or no jaw pose is known.

Label: simulator depth (RTX optical-Z ground truth, noise-free, perfectly registered, exact camera motion; not a
sensor). The variant changes the 0.35 appearance gate's rule, so its results are never pooled with
unchanged-gate results.
"""

from __future__ import annotations

import math

import numpy as np

from isaaclab_pruning.perception.jaw_self_mask import MASK_MARGIN_PX, jaw_boxes, jaw_mask, jaw_signed_distance_px
from isaaclab_pruning.perception.visual_servo import PATCH_SIZE_PX, VisualServoConfig, VisualServoTracker

ARMS = ("strict", "agreement")
PATCH_HALF = PATCH_SIZE_PX // 2
#: Pixel index u is the ray through continuous image coordinate u + 0.5 of K (the jaw module's convention).
PIXEL_CENTRE = 0.5

# Registered thresholds of the depth test (EVAL_PROTOCOL_DEPTH_APPEARANCE_HELDOUT_2026-10-01.md).
MIN_VERIFIED_FRACTION = 0.75
MIN_SAME_SURFACE_PX = 20
NEAR_TOLERANCE_M = 0.010
MAX_NEAR_FRACTION = 0.05
MAX_MEDIAN_ABS_M = 0.003
MAX_PIXEL_DISAGREEMENT_PX = 3.0
SAME_SURFACE_M = 0.025
MEDIAN_WINDOW_HALF = 1  # the 3 x 3 median rule

SIMULATOR_DEPTH_LABEL = (
    "Simulator depth: RTX optical-Z ground truth, noise-free, perfectly registered, exact camera motion; not a "
    "sensor. The variant changes the 0.35 appearance gate's rule."
)
ARM_LABELS = {
    "strict": "D_strict + J: on acceptance the 0.35 gate passes; the raw NCC stays in the unchanged confidence.",
    "agreement": (
        "agreement (+ J): on acceptance the 0.35 gate passes and the depth-agreement fraction replaces the NCC in "
        "the confidence, which changes the confidence gate's input. Reported separately, never pooled."
    ),
}
#: The failure reason of each registered condition.
CONDITION_REASONS = {
    "1_verified_fraction": "unverifiable_depth",
    "2_same_surface_pixels": "too_few_same_surface_pixels",
    "3_near_fraction": "occluder_nearer_than_predicted",
    "4_median_abs": "surface_moved_or_wrong_surface",
    "5_pixel_disagreement": "propagated_pixel_off_static_prediction",
    "J_jaw_silhouette": "jaw_silhouette_touches_patch",
}


# ------------------------------------------------------------------------------------------------ geometry
def backproject(pixels_index, depth_z, camera_matrix, world_from_optical):
    """World points of image-index pixels (N x 2) at optical-Z depths (N,)."""
    pixels = np.asarray(pixels_index, float).reshape(-1, 2)
    homogeneous = np.column_stack([pixels + PIXEL_CENTRE, np.ones(len(pixels))])
    rays = homogeneous @ np.linalg.inv(np.asarray(camera_matrix, float)).T
    optical = rays * (np.asarray(depth_z, float).reshape(-1, 1) / rays[:, 2:3])
    transform = np.asarray(world_from_optical, float)
    return optical @ transform[:3, :3].T + transform[:3, 3]


def project(points_w, camera_matrix, world_from_optical):
    """Image-index coordinates (N x 2) and optical Z (N,) of world points."""
    transform = np.asarray(world_from_optical, float)
    local = (np.asarray(points_w, float).reshape(-1, 3) - transform[:3, 3]) @ transform[:3, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        uvw = local @ np.asarray(camera_matrix, float).T
        uv = uvw[:, :2] / uvw[:, 2:3] - PIXEL_CENTRE
    return uv, local[:, 2]


def median3_depth(depth, rows, cols, min_m, max_m, min_valid_fraction):
    """The tracker's 3 x 3 median rule at integer index pixels; NaN outside the image or when invalid."""
    depth = np.asarray(depth)
    height, width = depth.shape
    rows, cols = np.asarray(rows, int), np.asarray(cols, int)
    out = np.full(rows.shape, np.nan)
    half = MEDIAN_WINDOW_HALF
    inside = (rows >= half) & (rows < height - half) & (cols >= half) & (cols < width - half)
    r, c = rows[inside], cols[inside]
    offsets = range(-half, half + 1)
    window = np.stack([depth[r + dr, c + dc] for dr in offsets for dc in offsets], axis=-1).astype(float)
    valid = np.isfinite(window) & (window >= min_m) & (window <= max_m)
    window[~valid] = np.nan
    enough = valid.mean(axis=-1) >= min_valid_fraction
    median = np.full(len(r), np.nan)
    if enough.any():
        median[enough] = np.nanmedian(window[enough], axis=-1)
    out[inside] = median
    return out


def tracked_depth(depth, pixel, radius, min_m, max_m):
    """Median of the valid samples in the tracker's own depth window at ``pixel`` (NaN if none or outside)."""
    x, y = np.rint(np.asarray(pixel, float)).astype(int)
    height, width = depth.shape
    if x - radius < 0 or y - radius < 0 or x + radius >= width or y + radius >= height:
        return float("nan")
    window = depth[y - radius : y + radius + 1, x - radius : x + radius + 1].astype(float)
    valid = np.isfinite(window) & (window >= min_m) & (window <= max_m)
    return float(np.median(window[valid])) if valid.any() else float("nan")


def _as_depth(depth):
    depth = np.asarray(depth)
    return depth[..., 0] if depth.ndim == 3 else depth


# ------------------------------------------------------------------------------------------------ the depth test
def depth_test(prev_pixel, next_pixel, prev_depth, prev_pose, cur_depth, cur_pose, camera_matrix, config):
    """Static-world consistency statistics of the previous 13 x 13 patch in the current frame (conditions 1-5).

    Returns (stats, uv_same, conditions); ``uv_same`` are the reprojected index coordinates of the same-surface
    elements (for J). ``conditions`` maps each condition to {value, limit, passed}; a missing value fails.
    """
    min_m, max_m, fraction = config.min_depth_m, config.max_depth_m, config.min_depth_valid_fraction
    prev_depth, cur_depth = _as_depth(prev_depth), _as_depth(cur_depth)
    p = np.asarray(prev_pixel, float).reshape(2)
    q = np.asarray(next_pixel, float).reshape(2)
    stats = {"previous_pixel": p.tolist(), "propagated_pixel": q.tolist()}
    z_track = tracked_depth(prev_depth, p, int(config.depth_radius_px), min_m, max_m)
    stats["tracked_depth_prev_m"] = z_track
    offsets = np.arange(-PATCH_HALF, PATCH_HALF + 1)
    cx, cy = np.rint(p).astype(int)
    cols, rows = np.meshgrid(cx + offsets, cy + offsets)
    z0 = median3_depth(prev_depth, rows, cols, min_m, max_m, fraction)
    valid0 = np.isfinite(z0)
    same = valid0 & (np.abs(z0 - z_track) <= SAME_SURFACE_M) if math.isfinite(z_track) else np.zeros_like(valid0)
    n_same = int(same.sum())
    stats.update({"n_patch": int(z0.size), "n_valid_prev": int(valid0.sum()), "n_same_surface": n_same})
    disagreement = None
    if math.isfinite(z_track):
        p_world = backproject(p[None], [z_track], camera_matrix, prev_pose)
        p_pred, p_pred_z = project(p_world, camera_matrix, cur_pose)
        disagreement = float(np.linalg.norm(q - p_pred[0]))
        stats.update(
            {
                "predicted_pixel": p_pred[0].tolist(),
                "predicted_depth_at_pixel_m": float(p_pred_z[0]),
            }
        )
    stats["pixel_disagreement_px"] = disagreement
    uv_same = np.zeros((0, 2))
    n_ver = 0
    if n_same:
        world = backproject(np.column_stack([cols[same], rows[same]]), z0[same], camera_matrix, prev_pose)
        uv_same, z_pred = project(world, camera_matrix, cur_pose)
        ui, vi = np.rint(uv_same[:, 0]), np.rint(uv_same[:, 1])
        ok = np.isfinite(ui) & np.isfinite(vi) & (z_pred > 0)
        ui = np.where(ok, ui, -1).astype(int)
        vi = np.where(ok, vi, -1).astype(int)
        d1 = median3_depth(cur_depth, vi, ui, min_m, max_m, fraction)
        d1[~ok] = np.nan
        verified = np.isfinite(d1)
        n_ver = int(verified.sum())
        if n_ver:
            diff = d1[verified] - z_pred[verified]  # negative: nearer than the static-world prediction
            absd = np.abs(diff)
            stats.update(
                {
                    "median_abs_m": float(np.median(absd)),
                    "median_signed_m": float(np.median(diff)),
                    "p90_abs_m": float(np.percentile(absd, 90)),
                    "min_signed_m": float(diff.min()),
                    "max_signed_m": float(diff.max()),
                    "near_count": int(np.sum(diff < -NEAR_TOLERANCE_M)),
                    "near_fraction": float(np.mean(diff < -NEAR_TOLERANCE_M)),
                    "far_fraction": float(np.mean(diff > NEAR_TOLERANCE_M)),
                    "agreement_fraction": float(np.mean(absd <= NEAR_TOLERANCE_M)),
                }
            )
    stats["n_verified"] = n_ver
    stats["verified_fraction"] = n_ver / n_same if n_same else 0.0

    def condition(value, limit, passes):
        evaluated = value is not None
        return {"value": value, "limit": limit, "evaluated": evaluated, "passed": bool(evaluated and passes(value))}

    conditions = {
        "1_verified_fraction": condition(
            stats["verified_fraction"], MIN_VERIFIED_FRACTION, lambda v: v >= MIN_VERIFIED_FRACTION
        ),
        "2_same_surface_pixels": condition(n_same, MIN_SAME_SURFACE_PX, lambda v: v >= MIN_SAME_SURFACE_PX),
        "3_near_fraction": condition(stats.get("near_fraction"), MAX_NEAR_FRACTION, lambda v: v <= MAX_NEAR_FRACTION),
        "4_median_abs": condition(stats.get("median_abs_m"), MAX_MEDIAN_ABS_M, lambda v: v <= MAX_MEDIAN_ABS_M),
        "5_pixel_disagreement": condition(
            disagreement, MAX_PIXEL_DISAGREEMENT_PX, lambda v: v <= MAX_PIXEL_DISAGREEMENT_PX
        ),
    }
    return stats, uv_same, conditions


def _footprint_masked(mask, pixel):
    """Mask pixels with nonzero weight in the 13 x 13 ``getRectSubPix`` footprint at ``pixel`` (the tracker's rule)."""
    import cv2

    sub = cv2.getRectSubPix(mask.astype(np.float32), (PATCH_SIZE_PX, PATCH_SIZE_PX), tuple(float(v) for v in pixel))
    return int((sub > 0).sum())


def _points_masked(mask, uv):
    uv = np.asarray(uv, float).reshape(-1, 2)
    finite = np.isfinite(uv).all(axis=1)
    xy = np.full((len(uv), 2), -1, int)
    xy[finite] = np.rint(uv[finite]).astype(int)
    height, width = mask.shape
    inside = finite & (xy[:, 0] >= 0) & (xy[:, 0] < width) & (xy[:, 1] >= 0) & (xy[:, 1] < height)
    return int(mask[xy[inside, 1], xy[inside, 0]].sum())


def jaw_contact(prev_boxes, prev_pose, cur_boxes, cur_pose, camera_matrix, shape, prev_pixel, next_pixel, uv_same):
    """Condition J: whether the committed 2.0 px jaw silhouette touches the previous, reprojected or propagated patch.

    ``*_boxes`` are ``jaw_self_mask.jaw_boxes`` lists (``[]`` = no jaw in view); ``None`` fails closed.
    """
    out = {"margin_px": MASK_MARGIN_PX}
    if prev_boxes is None or cur_boxes is None:
        out.update({"touches": True, "reason": "jaw_silhouette_unavailable"})
        return out
    try:
        prev_mask = jaw_mask(prev_boxes, camera_matrix, prev_pose, shape)
        cur_mask = jaw_mask(cur_boxes, camera_matrix, cur_pose, shape)
    except ValueError as error:  # a corner at or behind the camera: the silhouette is undefined
        out.update({"touches": True, "reason": "jaw_silhouette_undefined", "error": str(error)})
        return out
    out.update(
        {
            "mask_pixels_previous_frame": int(prev_mask.sum()),
            "mask_pixels_current_frame": int(cur_mask.sum()),
            "previous_patch_masked": _footprint_masked(prev_mask, prev_pixel),
            "reprojected_elements_masked": _points_masked(cur_mask, uv_same),
            "reprojected_elements": len(uv_same),
            "propagated_patch_masked": _footprint_masked(cur_mask, next_pixel),
        }
    )
    if cur_boxes:
        out["propagated_pixel_signed_distance_px"] = float(
            jaw_signed_distance_px(cur_boxes, camera_matrix, cur_pose, np.asarray(next_pixel, float))[0]
        )
    if prev_boxes:
        out["previous_pixel_signed_distance_px"] = float(
            jaw_signed_distance_px(prev_boxes, camera_matrix, prev_pose, np.asarray(prev_pixel, float))[0]
        )
    total = out["previous_patch_masked"] + out["reprojected_elements_masked"] + out["propagated_patch_masked"]
    out["touches"] = bool(total > 0)
    out["reason"] = CONDITION_REASONS["J_jaw_silhouette"] if total else None
    return out


# ------------------------------------------------------------------------------------------------ the tracker
class DepthAppearanceTracker(VisualServoTracker):
    """The repository tracker with D_strict + J (``arm="strict"``) or the agreement arm in front of its 0.35 gate.

    ``update`` takes the frame's jaw boxes (``jaw_self_mask.jaw_boxes``; ``[]`` when no jaw is in view, ``None``
    fails J closed). The tracker's own exclusion mask is not part of the registered variant and is refused.
    While the grey check passes, every result is exactly the base tracker's.
    """

    def __init__(self, config: VisualServoConfig | None = None, arm: str = "strict"):
        super().__init__(config)
        if arm not in ARMS:
            raise ValueError(f"arm must be one of {ARMS}")
        self.arm = arm
        self._previous_frame = None  # (depth, world_from_optical, jaw boxes) of the frame held in _previous_gray
        self._current_frame = None
        self._last_world_before_update = None
        self.last_event = None

    def initialize(self, rgb, pixel_xy, depth=None, exclusion_mask=None):
        if exclusion_mask is not None:
            raise ValueError("the registered variant runs without the tracker's exclusion mask")
        result = super().initialize(rgb, pixel_xy, depth)
        # The seed call carries no pose (as the live API): a grey failure at the next update cannot be verified.
        self._previous_frame = (None if depth is None else _as_depth(depth), None, None)
        return result

    def _appearance(self, gray, next_pixel):
        ncc = super()._appearance(gray, next_pixel)
        threshold = self.config.min_patch_correlation
        if ncc >= threshold:
            return ncc
        event = self._depth_event(ncc, next_pixel)
        self.last_event = event
        if not event["accept"]:
            return ncc
        if self.arm == "agreement":
            term = float(event["stats"]["agreement_fraction"])
            event["appearance_gate_value"] = term
            if term < threshold:  # unreachable: condition 4 implies an agreement fraction of at least 0.5
                event["accept"] = False
                event["reasons"].append("agreement_below_gate")
                return ncc
            return term
        event["appearance_gate_value"] = threshold
        return threshold

    def _depth_event(self, ncc, next_pixel):
        prev_depth, prev_pose, prev_boxes = self._previous_frame or (None, None, None)
        cur_depth, camera_matrix, cur_pose, cur_boxes = self._current_frame
        event = {"ncc": float(ncc), "reasons": [], "accept": False}
        if prev_depth is None or prev_pose is None:
            event["reasons"].append("no_previous_depth_or_pose")
            return event
        if np.ndim(cur_depth) != 2 or np.shape(cur_depth) != np.shape(prev_depth):
            event["reasons"].append("no_current_depth")
            return event
        stats, uv_same, conditions = depth_test(
            self._pixel, next_pixel, prev_depth, prev_pose, cur_depth, cur_pose, camera_matrix, self.config
        )
        jaw = jaw_contact(
            prev_boxes,
            prev_pose,
            cur_boxes,
            cur_pose,
            camera_matrix,
            cur_depth.shape,
            self._pixel,
            next_pixel,
            uv_same,
        )
        conditions["J_jaw_silhouette"] = {
            "value": jaw.get("reason"),
            "limit": None,
            "evaluated": True,
            "passed": not jaw["touches"],
        }
        # A condition without a value (no tracked depth, nothing verified) cannot pass; it is reported as not
        # evaluated rather than as a failure with a reason it never measured. Acceptance needs every condition.
        not_evaluated = [name for name, item in conditions.items() if not item["evaluated"]]
        failed = [name for name, item in conditions.items() if item["evaluated"] and not item["passed"]]
        reasons = [] if math.isfinite(stats["tracked_depth_prev_m"]) else ["no_tracked_depth"]
        for name in failed:
            reason = jaw["reason"] if name == "J_jaw_silhouette" else CONDITION_REASONS[name]
            if reason not in reasons:
                reasons.append(reason)
        event.update(
            {
                "stats": stats,
                "conditions": conditions,
                "failed_conditions": failed,
                "not_evaluated_conditions": not_evaluated,
                "jaw": jaw,
                "reasons": reasons,
                "accept": not failed and not not_evaluated,
            }
        )
        return event

    def update(self, rgb, depth, camera_matrix, world_from_optical, exclusion_mask=None, *, jaw_boxes=None):
        if exclusion_mask is not None:
            raise ValueError("the registered variant runs without the tracker's exclusion mask")
        depth_array = _as_depth(depth)
        self._current_frame = (
            depth_array,
            np.asarray(camera_matrix, float),
            np.asarray(world_from_optical, float),
            jaw_boxes,
        )
        self.last_event = None
        before = self._previous_gray
        self._last_world_before_update = self._last_world
        result = super().update(rgb, depth, camera_matrix, world_from_optical)
        if self.last_event is not None:
            self._finish_event(result, self.last_event)
        if self._previous_gray is not before:
            self._previous_frame = (depth_array, np.asarray(world_from_optical, float), jaw_boxes)
        return result

    def _finish_event(self, result, event):
        """Restore the raw NCC in the result, compute both arms' confidences and apply the strict confidence."""
        result["patch_correlation"] = event["ncc"]
        result["depth_appearance"] = event
        event["arm"] = self.arm
        if not event["accept"]:
            event["decision"] = "loss_stands"
            return
        result["appearance_gate_value"] = event["appearance_gate_value"]
        reached_confidence = result["state"] == "tracking" or result.get("reason") == "low_confidence"
        # Every path after an accepted appearance check samples the depth window, so the confidence inputs exist
        # even when a later gate (depth window, calibration, world jump) ends the update.
        if "depth_valid_fraction" in result:
            ratio = min(1.0, result["feature_count"] / max(self._initial_count, 1))
            dvf = float(result["depth_valid_fraction"])
            event["feature_ratio"] = ratio
            event["initial_feature_count"] = int(self._initial_count)
            event["depth_valid_fraction"] = dvf
            event["strict_confidence"] = float(ratio * max(0.0, event["ncc"]) * dvf)
            event["agreement_confidence"] = float(ratio * event["stats"]["agreement_fraction"] * dvf)
        event["reached_confidence_gate"] = reached_confidence
        event["ended_by"] = None if result["state"] == "tracking" else [result["state"], result["reason"]]
        if self.arm == "strict" and reached_confidence:
            strict = event["strict_confidence"]
            if result["state"] == "tracking":
                # The registered confidence keeps the raw NCC; the base computed it with the gate value.
                result["confidence_with_gate_value"] = result["confidence"]
                result["confidence"] = strict
            if result["state"] == "tracking" and strict < self.config.min_confidence:
                # As the repository's own low-confidence return: no stored world target, no replenished corners.
                self._lost = True
                self._last_world = self._last_world_before_update
                self._points = self._points[: result["feature_count"]]
                result.pop("pending_feature_pixels_for_next_frame", None)
                result.update(
                    {
                        "state": "tracking_lost",
                        "reason": "low_confidence",
                        "target_position_world_m": None,
                        "requires_reinitialize": True,
                        "confidence": 0.0,
                    }
                )
            if result.get("reason") == "low_confidence":
                result["measured_confidence"] = strict
        event["decision"] = "continues" if result["state"] == "tracking" else "stops_downstream"
        event["result_state"], event["result_reason"] = result["state"], result["reason"]
        event["ended_by"] = None if result["state"] == "tracking" else [result["state"], result["reason"]]


def frame_jaw_boxes(tool_pose_wxyz, roll_rad, closure_progress, radius_m):
    """The frame's jaw boxes for J, or None (J fails closed) when the pose or progress is unusable."""
    try:
        return jaw_boxes(tool_pose_wxyz, roll_rad, float(closure_progress), radius_m)
    except (TypeError, ValueError):
        return None


def registered_depth_appearance(arm="strict"):
    """The registered constants of one arm, JSON-native; a capture's report must show exactly these."""
    if arm not in ARMS:
        raise ValueError(f"arm must be one of {ARMS}")
    return {
        "arm": arm,
        "min_verified_fraction": MIN_VERIFIED_FRACTION,
        "min_same_surface_px": MIN_SAME_SURFACE_PX,
        "near_tolerance_m": NEAR_TOLERANCE_M,
        "max_near_fraction": MAX_NEAR_FRACTION,
        "max_median_abs_m": MAX_MEDIAN_ABS_M,
        "max_pixel_disagreement_px": MAX_PIXEL_DISAGREEMENT_PX,
        "same_surface_m": SAME_SURFACE_M,
        "median_window_half": MEDIAN_WINDOW_HALF,
        "pixel_centre": PIXEL_CENTRE,
        "jaw_mask_margin_px": MASK_MARGIN_PX,
    }
