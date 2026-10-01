#!/usr/bin/env python3
"""Offline replay of the depth-aware appearance check ``D_strict + J`` and its labelled ``agreement`` arm.

Implements exactly the variant registered in
docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_HELDOUT_2026-10-01.md (H1-H4). CPU only.

Scope. A subclass of the repository ``VisualServoTracker`` that differs from it
only when the grey NCC appearance check fails (below ``min_patch_correlation``,
0.35). It then runs a static-world depth test on the previous 13 x 13 patch and
the jaw guard J. On acceptance the 0.35 gate passes. In the primary arm,
``strict``, the raw NCC stays in the unchanged confidence
``min(1, n/n0) * NCC * depth_valid_fraction >= 0.15``. The secondary arm,
``agreement``, replaces the NCC by the depth-agreement fraction in that
confidence, which changes the confidence gate's input; it is reported
separately and never pooled with ``strict``. Every other check (optical flow,
depth window, world jump, replenishment) is the repository code, called
unchanged through ``super()``.

The depth test (registered thresholds; module constants, not options):
  * back-project the previous patch rint(p) + [-6..6]^2 with the previous
    frame's optical-Z and pose (pixel index u has its centre at u + 0.5);
  * reproject it with the current pose and sample the current depth with the
    3 x 3 median rule (median of the valid samples, invalid unless at least
    ``min_depth_valid_fraction`` of the window is valid; unlike the tracker's
    own depth window it has no spread or centre check). As in the study's
    median3 path (depthcheck.py ``sample_depth``, run as D_median3 at the 8
    earlier grey-check failures; its 129-run regression used nearest-pixel
    sampling), the same rule also gives the previous patch's optical-Z;
  * keep the elements within 25 mm of the tracked depth (the median of the
    tracker's own depth window at p in the previous frame): the same-surface
    elements;
  * accept only if (1) >= 75% of the same-surface elements are verified (land
    in the image with a valid current depth), (2) >= 20 same-surface elements
    remain, (3) <= 5% of the verified elements are nearer than predicted by
    more than 10 mm, (4) the median |measured - predicted| over the verified
    elements is <= 3 mm, (5) the propagated pixel is within 3 px of the
    static-world prediction of p, and
    (J) the committed 2.0 px jaw silhouette
    (``perception/jaw_self_mask.jaw_mask``) touches none of the previous,
    reprojected or propagated patches. "Touches" is any mask pixel in the
    patch footprint: the ``getRectSubPix`` footprint (any nonzero weight, as the
    committed tracker samples a mask) for the previous patch (previous frame's
    mask) and the propagated patch (current mask), and the rounded reprojected
    same-surface elements in the current mask. J fails closed when the
    silhouette is undefined (a jaw corner at or behind the camera) or no jaw
    pose was supplied.

Label. Simulator depth: RTX optical-Z ground truth, noise-free, perfectly
registered, exact camera motion; not a sensor. The variant changes the 0.35
appearance gate's rule.

Replay recipe (tools/replay_visual_tracking.py, tools/replay_jaw_self_mask.py):
the tracker config is frames[0].live_vision.tracker_config; the seed
initialization uses the preview image and depth (skipped when the recording
rejected it); the preview update uses the camera pose rebuilt from the initial
tool pose; every frame update uses the recorded wrist pose. Jaw boxes use the
recorded tool pose, the attachment roll ``proxy_roll_rad(closing_axis_tool)``,
the closure progress rendered into the frame and the scene's branch radius.

Offline-only limits. Only decisions up to each run's recorded stop (the first
frame whose recorded cut phase is ``stopped``; else the last frame) are
replayed: frames after a stop show a stopped robot. A divergence shows that the
recorded stop would not have happened at that frame, not what follows. No
sensor noise, pose error or time-sync error is modelled, and same-depth
occluders were never recorded. Nothing here is a live outcome or a grade; only
tools/validate_vision_sequence.py grades live runs. Inputs are read only; the
output must be a new file outside the repository (and artifacts/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source/isaaclab_pruning"))
from isaaclab_pruning.perception.jaw_self_mask import (
    MASK_MARGIN_PX,
    jaw_boxes,
    jaw_mask,
    jaw_signed_distance_px,
    proxy_roll_rad,
)
from isaaclab_pruning.perception.visual_servo import PATCH_SIZE_PX, VisualServoConfig, VisualServoTracker
from isaaclab_pruning.task.simulated_cut import tool_mouth_geometry

VISION_ROBUSTNESS = ROOT / "artifacts/vision_robustness"
HELDOUT_BATCH_GLOB = "jaw-shadow-*-20260930"
#: Batches recorded after the protocol (the held-out counterfactual and the live jaw-in-view runs).
REGRESSION_EXCLUDED_PREFIXES = ("jaw-shadow-", "jaw-hold-")
#: The registered data sets: 12 held-out recordings and the 129 earlier runs with frames.
EXPECTED_RUNS = {"heldout": 12, "regression": 129}
CORRELATION_TOLERANCE = 1e-6
ARMS = ("strict", "agreement")
PROTOCOL = "docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_HELDOUT_2026-10-01.md"
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
SOURCE_FILES = (
    "source/isaaclab_pruning/isaaclab_pruning/perception/visual_servo.py",
    "source/isaaclab_pruning/isaaclab_pruning/perception/jaw_self_mask.py",
    "source/isaaclab_pruning/isaaclab_pruning/task/simulated_cut.py",
    "tools/replay_depth_appearance.py",
)


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


# ------------------------------------------------------------------------------------------------ recorded runs
def _transform(position, rotation):
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(rotation, dtype=float)
    transform[:3, 3] = np.asarray(position, dtype=float)
    return transform


def preview_transform(report):
    """world_from_optical of the preview frame, rebuilt from the initial tool pose (replay_visual_tracking.py)."""
    camera = report["camera"]
    pose = report["initial_tool_pose_wxyz"][0]
    rotation = np.column_stack([tool_mouth_geometry(pose, closing_axis_tool=axis)[1] for axis in np.eye(3)])
    return _transform(
        np.asarray(pose[:3]) + rotation @ camera["wrist_position_in_tool_m"],
        rotation @ np.asarray(camera["wrist_rotation_in_tool_ros"]),
    )


def tracker_config(frames):
    config = dict(frames[0]["live_vision"]["tracker_config"])
    config["roi_half_size_px"] = tuple(config["roi_half_size_px"])
    return VisualServoConfig(**config)


def _has_frames(path):
    return (
        (path / "report.json").is_file()
        and (path / "frames.json").is_file()
        and (path / "frames").is_dir()
        and any((path / "frames").glob("wrist_*.png"))
    )


def discover_heldout(base=VISION_ROBUSTNESS):
    """The jaw-shadow counterfactual's run directories (the registered held-out data)."""
    return [path for path in sorted(Path(base).glob(f"{HELDOUT_BATCH_GLOB}/run_*")) if _has_frames(path)]


def discover_regression(base=VISION_ROBUSTNESS):
    """Every earlier recorded run with frames (outside the batches recorded after the protocol)."""
    return [
        path
        for path in sorted(Path(base).glob("*/run_*"))
        if not path.parent.name.startswith(REGRESSION_EXCLUDED_PREFIXES) and _has_frames(path)
    ]


def output_refusal(path):
    """Why the output path is not allowed (inside the repository or artifacts/, or existing), else None."""
    resolved = Path(path).expanduser().resolve()
    for forbidden in (ROOT.resolve(), (ROOT / "artifacts").resolve(), Path.home().resolve()):
        if resolved == forbidden or forbidden in resolved.parents:
            return f"refusing to write inside {forbidden}: {resolved}"
    if resolved.exists():
        return f"refusing to overwrite an existing file: {resolved}"
    if not resolved.parent.is_dir():
        return f"the output directory does not exist: {resolved.parent}"
    return None


def same_measurement(replayed, recorded):
    """Differing fields: state, reason and pixel_xy exactly, patch_correlation within the tolerance."""
    fields = [key for key in ("state", "reason", "pixel_xy") if replayed.get(key) != recorded.get(key)]
    a, b = replayed.get("patch_correlation"), recorded.get("patch_correlation")
    if (a is None) != (b is None) or (a is not None and abs(a - b) > CORRELATION_TOLERANCE):
        fields.append("patch_correlation")
    return fields


def _brief(measurement):
    measurement = measurement or {}
    keys = ("state", "reason", "pixel_xy", "patch_correlation", "feature_count", "confidence", "depth_reason")
    out = {key: measurement.get(key) for key in keys}
    for key in (
        "measured_confidence",
        "confidence_with_gate_value",
        "appearance_gate_value",
        "depth_valid_fraction",
        "world_jump_m",
        "target_position_world_m",
    ):
        if key in measurement:
            out[key] = measurement[key]
    return out


def _rgb(path):
    return np.array(Image.open(path).convert("RGB"))


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _frame_jaw_boxes(tool_pose, roll, progress, radius):
    try:
        return jaw_boxes(tool_pose, roll, float(progress), radius)
    except (TypeError, ValueError):
        return None  # J fails closed


def _event_record(index, event):
    record = {"frame": index, **{key: value for key, value in event.items() if key != "stats"}}
    record.update(event.get("stats") or {})
    return record


def replay_run(path):
    """Replay one recorded run through the base tracker and both arms in lockstep, up to its recorded stop."""
    path = Path(path)
    report = json.loads((path / "report.json").read_text())
    if report.get("stage") != "complete":
        raise ValueError(f"{path}: the recording is not complete (stage {report.get('stage')!r})")
    if any((report.get(arm) or {}).get("enabled") for arm in ("jaw_self_mask", "closure_hold")):
        raise ValueError(f"{path}: recorded with the jaw self-mask or closure hold, which this replay does not model")
    frames = json.loads((path / "frames.json").read_text())["frames"]
    if [frame["index"] for frame in frames] != list(range(len(frames))):
        raise ValueError(f"{path}: frame indexes are not 0..N-1")
    camera_matrix = np.asarray(report["camera"]["wrist_intrinsics"], dtype=float)
    config = tracker_config(frames)
    trackers = {"base": VisualServoTracker(config)}
    trackers.update({arm: DepthAppearanceTracker(config, arm=arm) for arm in ARMS})
    initial_live = report.get("initial_live_vision") or {}
    closing_axis = initial_live.get("closing_axis_tool") or frames[0]["live_vision"]["closing_axis_tool"]
    roll = proxy_roll_rad(closing_axis)
    radius = float(report["blender_scene"]["target"]["radius_m"])
    recorded_stop = next((f["index"] for f in frames if f["live_vision"]["cut"]["phase"] == "stopped"), None)
    detach = next((f["index"] for f in frames if f["live_vision"]["cut"].get("detach_event")), None)
    last = recorded_stop if recorded_stop is not None else len(frames) - 1

    recorded_preview = initial_live.get("measurement") or {}
    preview_rgb = _rgb(path / "preview_wrist.png")
    preview_depth = np.load(path / "preview_depth.npy", allow_pickle=False)
    preview_pose = preview_transform(report)
    recorded_init = (report.get("vision_initialization") or {}).get("tracker") or {}
    rejected = recorded_init.get("state") == "initialization_rejected"
    seed = report["vision_initialization"]["pixel_xy"]
    init = {name: None if rejected else t.initialize(preview_rgb, seed, preview_depth) for name, t in trackers.items()}
    preview_boxes = _frame_jaw_boxes(report["initial_tool_pose_wxyz"][0], roll, 0.0, radius)
    preview = {
        name: tracker.update(preview_rgb, preview_depth, camera_matrix, preview_pose)
        if name == "base"
        else tracker.update(preview_rgb, preview_depth, camera_matrix, preview_pose, jaw_boxes=preview_boxes)
        for name, tracker in trackers.items()
    }

    per_arm = {name: {"mismatches": [], "max_corr": 0.0} for name in trackers}
    events = {arm: [] for arm in ARMS}
    for name, tracker in trackers.items():
        if name != "base" and tracker.last_event is not None:
            record = _event_record("preview", tracker.last_event)
            record.update({"recorded": _brief(recorded_preview), "replayed": _brief(preview[name])})
            events[name].append(record)
    at_stop = {}
    for frame in frames[: last + 1]:
        index = frame["index"]
        rgb = _rgb(path / f"frames/wrist_{index:05d}.png")
        depth = np.load(path / f"frames/depth_{index:05d}.npy", allow_pickle=False)
        pose = _transform(frame["wrist_position_w_m"], frame["wrist_rotation_w_ros"])
        boxes = _frame_jaw_boxes(frame["tool_pose_wxyz"], roll, frame.get("visual_jaw_closure_progress") or 0.0, radius)
        recorded = frame["live_vision"]["measurement"]
        for name, tracker in trackers.items():
            if name == "base":
                result = tracker.update(rgb, depth, camera_matrix, pose)
            else:
                result = tracker.update(rgb, depth, camera_matrix, pose, jaw_boxes=boxes)
                if tracker.last_event is not None:
                    record = _event_record(index, tracker.last_event)
                    record["after_recorded_detach"] = detach is not None and index > detach
                    record["recorded"] = _brief(recorded)
                    record["recorded_cut_phase"] = frame["live_vision"]["cut"]["phase"]
                    record["replayed"] = _brief(result)
                    events[name].append(record)
            fields = same_measurement(result, recorded)
            a, b = result.get("patch_correlation"), recorded.get("patch_correlation")
            if a is not None and b is not None:
                per_arm[name]["max_corr"] = max(per_arm[name]["max_corr"], abs(a - b))
            if fields:
                per_arm[name]["mismatches"].append(
                    {
                        "frame": index,
                        "fields": fields,
                        "recorded": _brief(recorded),
                        "replayed": _brief(result),
                        "after_recorded_detach": detach is not None and index > detach,
                        "event_frame": name != "base" and tracker.last_event is not None,
                    }
                )
            if index == recorded_stop:
                at_stop[name] = _brief(result)

    divergence = {}
    for name in trackers:
        mismatches = per_arm[name]["mismatches"]
        preview_fields = same_measurement(preview[name], recorded_preview)
        divergence[name] = {
            "preview_mismatch_fields": preview_fields,
            "mismatch_frames": [item["frame"] for item in mismatches],
            "first_divergence_frame": mismatches[0]["frame"] if mismatches else None,
            "divergences": mismatches,
            "max_abs_patch_correlation_difference": per_arm[name]["max_corr"],
            "equals_recording": not mismatches and not preview_fields,
        }
    base_init = init["base"]
    init_check = {
        "recorded": [recorded_init.get("state"), recorded_init.get("reason"), recorded_init.get("feature_count", 0)],
        "replayed": None
        if base_init is None
        else [base_init["state"], base_init["reason"], base_init["feature_count"]],
    }
    init_check["matches"] = rejected or init_check["replayed"] == init_check["recorded"]
    base_exact = init_check["matches"] and divergence["base"]["equals_recording"]
    stop_measurement = frames[recorded_stop]["live_vision"]["measurement"] if recorded_stop is not None else None
    joined = _join_events(events)
    return {
        "run": f"{path.parent.name}/{path.name}",
        "path": str(path),
        "frames": len(frames),
        "source_sha256": {name: _sha256(path / name) for name in ("report.json", "frames.json")},
        "recorded": {
            "task_outcome": report.get("task_outcome"),
            "stop_frame": recorded_stop,
            "stop_reason": None
            if recorded_stop is None
            else frames[recorded_stop]["live_vision"]["cut"]["stopped_reason"],
            "stop_measurement": _brief(stop_measurement) if stop_measurement else None,
            "detach_frame": detach,
            "initialization_state": recorded_init.get("state"),
        },
        "replayed_through_frame": last,
        "initialization": init_check,
        "base_reproduces_recording": base_exact,
        "variant_results_interpretable": base_exact,
        "measurement_at_recorded_stop": at_stop,
        "divergence": divergence,
        "events": events,
        "events_joined": joined,
        "grey_failures": sorted(
            {event["frame"] for arm in ARMS for event in events[arm]}, key=lambda f: -1 if f == "preview" else f
        ),
    }


def _join_events(events):
    """Per frame: the depth test and J (from the strict arm when both arms saw the frame) and both arms' decisions."""
    by_frame = {}
    for arm in ARMS:
        for event in events[arm]:
            by_frame.setdefault(str(event["frame"]), {})[arm] = event
    joined = []
    for frame, arms in by_frame.items():
        source = arms.get("strict") or arms.get("agreement")
        same_input = (
            "strict" in arms
            and "agreement" in arms
            and arms["strict"].get("previous_pixel") == arms["agreement"].get("previous_pixel")
            and arms["strict"].get("propagated_pixel") == arms["agreement"].get("propagated_pixel")
        )
        row = {
            "frame": source["frame"],
            "arms_saw_the_same_update": same_input,
            "ncc": source["ncc"],
            "accept": source["accept"],
            "reasons": source["reasons"],
            "failed_conditions": source.get("failed_conditions"),
            "not_evaluated_conditions": source.get("not_evaluated_conditions"),
            "conditions": source.get("conditions"),
            "jaw": source.get("jaw"),
            "strict_confidence": source.get("strict_confidence"),
            "agreement_confidence": source.get("agreement_confidence"),
            "feature_ratio": source.get("feature_ratio"),
            "depth_valid_fraction": source.get("depth_valid_fraction"),
            "initial_feature_count": source.get("initial_feature_count"),
            "reached_confidence_gate": source.get("reached_confidence_gate"),
            "after_recorded_detach": source.get("after_recorded_detach"),
        }
        for key in (
            "tracked_depth_prev_m",
            "n_same_surface",
            "n_verified",
            "verified_fraction",
            "median_abs_m",
            "median_signed_m",
            "near_count",
            "near_fraction",
            "agreement_fraction",
            "pixel_disagreement_px",
        ):
            row[key] = source.get(key)
        for arm in ARMS:
            event = arms.get(arm)
            row[f"{arm}_decision"] = (
                None
                if event is None
                else {
                    "state": (event.get("replayed") or {}).get("state"),
                    "reason": (event.get("replayed") or {}).get("reason"),
                    "continues": (event.get("replayed") or {}).get("state") == "tracking",
                    "accept": event["accept"],
                    "ended_by": event.get("ended_by"),
                }
            )
        joined.append(row)
    return joined


# ------------------------------------------------------------------------------------------------ output
def _json_safe(value):
    """Strict JSON: non-finite floats become null, numpy values become Python."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(value) else None
    return value


def _provenance():
    hashes = {name: _sha256(ROOT / name) for name in SOURCE_FILES}
    try:
        revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "--", *SOURCE_FILES], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"code_revision": revision, "source_files_status_porcelain": dirty, "source_sha256": hashes}


def _summary(results):
    ok = [r for r in results if "error" not in r]
    summary = {
        "runs": len(results),
        "errors": [r["run"] for r in results if "error" in r],
        "base_reproduces_recording": sum(r["base_reproduces_recording"] for r in ok),
        "base_does_not_reproduce": [r["run"] for r in ok if not r["base_reproduces_recording"]],
        "updates_replayed": sum(r["replayed_through_frame"] + 1 for r in ok),
        "runs_with_grey_failures": {r["run"]: r["grey_failures"] for r in ok if r["grey_failures"]},
    }
    for arm in ARMS:
        summary[arm] = {
            "runs_equal_to_recording": sum(r["divergence"][arm]["equals_recording"] for r in ok),
            "diverging_runs": {
                r["run"]: {
                    "frames": r["divergence"][arm]["mismatch_frames"],
                    "preview_fields": r["divergence"][arm]["preview_mismatch_fields"],
                    "states": [
                        [d["frame"], d["recorded"]["state"], d["replayed"]["state"], d["replayed"]["reason"]]
                        for d in r["divergence"][arm]["divergences"]
                    ],
                }
                for r in ok
                if not r["divergence"][arm]["equals_recording"]
            },
        }
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--run-dir", type=Path, action="append", help="A recorded run directory (repeatable)")
    selection.add_argument("--heldout", action="store_true", help=f"The {HELDOUT_BATCH_GLOB} run directories")
    selection.add_argument("--regression", action="store_true", help="The 129 earlier runs with frames")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file outside the repository")
    args = parser.parse_args(argv)
    progress = args.output.with_name(args.output.name + ".progress.jsonl")
    for path in (args.output, progress):
        refusal = output_refusal(path)
        if refusal:
            parser.error(refusal)
    import cv2

    cv2.setNumThreads(1)
    if args.heldout:
        selection_name, runs = "heldout", discover_heldout()
    elif args.regression:
        selection_name, runs = "regression", discover_regression()
    else:
        selection_name, runs = "run_dir", [Path(path) for path in args.run_dir]
    if selection_name in EXPECTED_RUNS and len(runs) != EXPECTED_RUNS[selection_name]:
        parser.error(f"{selection_name}: found {len(runs)} runs, registered {EXPECTED_RUNS[selection_name]}")
    header = {
        "header": True,
        "tool": "tools/replay_depth_appearance.py",
        "protocol": PROTOCOL,
        "protocol_sha256": _sha256(ROOT / PROTOCOL),
        "selection": selection_name,
        "runs": [f"{Path(r).parent.name}/{Path(r).name}" for r in runs],
        **_provenance(),
    }
    with progress.open("a") as stream:
        stream.write(json.dumps(_json_safe(header), allow_nan=False) + "\n")
    results = []
    for number, path in enumerate(runs, 1):
        started = time.time()
        try:
            result = replay_run(path)
        except Exception as error:  # noqa: BLE001 - a failed replay is recorded, never dropped
            result = {"run": f"{Path(path).parent.name}/{Path(path).name}", "path": str(path), "error": repr(error)}
        result["replay_seconds"] = round(time.time() - started, 2)
        results.append(result)
        with progress.open("a") as stream:
            stream.write(json.dumps(_json_safe(result), allow_nan=False) + "\n")
        if "error" in result:
            print(f"[{number}/{len(runs)}] {result['run']}: ERROR {result['error']}", flush=True)
            continue
        print(
            f"[{number}/{len(runs)}] {result['run']}: through={result['replayed_through_frame']} "
            f"base_exact={result['base_reproduces_recording']} grey_failures={result['grey_failures']} "
            + " ".join(f"{arm}_div={result['divergence'][arm]['mismatch_frames']}" for arm in ARMS)
            + f" t={result['replay_seconds']}s",
            flush=True,
        )
    document = {
        "schema_version": 1,
        "tool": "tools/replay_depth_appearance.py",
        "protocol": PROTOCOL,
        "protocol_sha256": _sha256(ROOT / PROTOCOL),
        "selection": selection_name,
        "label": SIMULATOR_DEPTH_LABEL,
        "arms": ARM_LABELS,
        **_provenance(),
        "thresholds": {
            "min_patch_correlation_gate": "the tracker config's min_patch_correlation (0.35)",
            "min_confidence": "the tracker config's min_confidence (0.15)",
            "min_verified_fraction": MIN_VERIFIED_FRACTION,
            "min_same_surface_px": MIN_SAME_SURFACE_PX,
            "near_tolerance_m": NEAR_TOLERANCE_M,
            "max_near_fraction": MAX_NEAR_FRACTION,
            "max_median_abs_m": MAX_MEDIAN_ABS_M,
            "max_pixel_disagreement_px": MAX_PIXEL_DISAGREEMENT_PX,
            "same_surface_m": SAME_SURFACE_M,
            "verified_fraction_denominator": "same-surface elements (within 25 mm of the tracked depth)",
            "near_fraction_denominator": "verified same-surface elements",
            "agreement_fraction": "verified elements with |measured - predicted| <= 10 mm, over the verified elements",
            "depth_sampling": (
                "3 x 3 median rule on both the previous and the current depth, as depthcheck.py median3: the median "
                "of the valid samples, invalid unless at least min_depth_valid_fraction (0.75) of the window is "
                "valid; no spread or centre check"
            ),
            "tracked_depth": "median of the valid samples in the tracker's own depth window at p, previous frame",
            "jaw_mask_margin_px": MASK_MARGIN_PX,
            "jaw_touch_rule": (
                "any mask pixel in the getRectSubPix footprint of the previous patch (previous frame's mask) or the "
                "propagated patch (current mask), or at a rounded reprojected same-surface element (current mask); "
                "fails closed when the silhouette is undefined or no jaw pose is known"
            ),
        },
        "criterion": (
            "Per run, through its recorded stop (else every frame): the replayed measurement against the recorded "
            "live_vision.measurement, state, reason and pixel_xy exactly and patch_correlation within 1e-6; plus the "
            "seed initialization and the preview update. Variant results count only where the base tracker "
            "reproduces the recording."
        ),
        "scope": (
            "Offline replay of recorded images only. A divergence shows that the recorded stop would not have "
            "happened at that frame, not what follows; no sensor noise, pose error or time-sync error is modelled."
        ),
        "summary": _summary(results),
        "runs": results,
    }
    with args.output.open("x") as stream:
        json.dump(_json_safe(document), stream, allow_nan=False)
        stream.write("\n")
    print(json.dumps(_json_safe(document["summary"])), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
