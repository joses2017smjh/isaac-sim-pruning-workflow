"""The robot's own visual jaw surrogate, projected into the wrist image as a self-pixel mask.

The renderer poses two jaw boxes from the tool pose, a fixed attachment roll and
the closure progress the cut controller commanded (``BlenderDemoScene``'s
``set_proxy_closing_axis_tool`` and ``update_tool_proxy``). The controller holds
every one of those inputs and the rendering camera's calibration, so it can
predict where its own jaw lies in each frame before the tracker sees the frame.
That prediction is a robot self-model, not a measurement: the mask marks pixels
the tracker must never use as evidence for the branch.

The model is simulator-exact: the two-box visual surrogate, commanded closure,
the scene's branch radius for the closed gap and an ideal pinhole. A real jaw's
CAD, calibration error and latency would need a re-measured margin. Everything
here is a code constant; nothing is reachable from the environment. No pxr.
"""

from __future__ import annotations

import itertools
import math

import numpy as np

from isaaclab_pruning.perception.visual_servo import JAW_MASK_OCCLUDED_REASON, JAW_MASK_OCCLUDED_STATE

#: Half extents of each jaw box in its own frame (the cube scaled to 6 x 25 x 30 mm).
JAW_HALF_EXTENTS_M = (0.003, 0.0125, 0.015)
#: Inner gap between the open jaws.
JAW_OPEN_GAP_M = 0.032
#: Jaw centre height along tool-local +Z.
JAW_HEIGHT_M = 0.070
#: Euclidean margin around each box's projected silhouette, at pixel centres.
MASK_MARGIN_PX = 2.0
#: Pixel index u has its centre at continuous image coordinate u + 0.5.
PIXEL_CENTRE = 0.5
#: Fewest of the 13 x 13 appearance-patch elements that must remain unmasked in both patches.
MIN_UNMASKED_PATCH_ELEMENTS = 140
#: Closure hold: the tool must stay this close to its pose at the closure-start step.
HOLD_MAX_TRANSLATION_M = 0.0005
HOLD_MAX_ROTATION_DEG = 0.25
#: A box corner this close to (or behind) the camera makes the hull silhouette undefined.
MIN_CORNER_DEPTH_M = 0.01

MODEL_LABEL = "two-box visual surrogate, simulator-exact"

#: The losses a closure hold may cover, and only while closing and stationary.
CLOSURE_HOLD_EXPLAINED_BRANCHES = (
    "jaw_mask_occluded: fewer than the minimum unmasked appearance-patch elements",
    "optical_flow_failed after the mask removed at least one otherwise-valid feature, where the same LK output "
    "without the mask would have passed the flow checks",
    "explicit_initialization_required latched by a held loss of either kind within this closure",
)


def registered_jaw_self_mask():
    """The registered self-mask constants, JSON-native; a capture's report must show exactly these."""
    return {
        "jaw_half_extents_m": list(JAW_HALF_EXTENTS_M),
        "jaw_open_gap_m": JAW_OPEN_GAP_M,
        "jaw_height_m": JAW_HEIGHT_M,
        "mask_margin_px": MASK_MARGIN_PX,
        "pixel_centre": PIXEL_CENTRE,
        "min_unmasked_patch_elements": MIN_UNMASKED_PATCH_ELEMENTS,
        "min_corner_depth_m": MIN_CORNER_DEPTH_M,
    }


def registered_closure_hold():
    """The registered closure-hold thresholds, JSON-native."""
    return {"max_translation_m": HOLD_MAX_TRANSLATION_M, "max_rotation_deg": HOLD_MAX_ROTATION_DEG}


def _quat_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz, float) / np.linalg.norm(wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _rot_z(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def proxy_roll_rad(closing_axis_tool):
    """Attachment roll of the jaw surrogate, exactly as ``set_proxy_closing_axis_tool`` derives it."""
    axis = np.array(closing_axis_tool, dtype=float)  # a copy: never normalize the caller's array in place
    if axis.shape != (3,) or not np.isfinite(axis).all() or abs(axis[2]) > 1e-6 or np.linalg.norm(axis) < 1e-8:
        raise ValueError("Proxy closing axis must be nonzero and in the tool XY plane")
    axis /= np.linalg.norm(axis)
    return float(np.arctan2(axis[1], axis[0]))


def jaw_gap_m(closure_progress, radius_m):
    """Inner jaw gap for a commanded closure progress, as ``update_tool_proxy`` renders it."""
    p = float(closure_progress)
    return (1 - p) * JAW_OPEN_GAP_M + p * 2 * float(radius_m)


def jaw_boxes(tool_pose_wxyz, roll_rad, closure_progress, radius_m):
    """Both jaw boxes as ``(centre_w, rotation_w_from_box, half_extents)``, posed as ``update_tool_proxy`` does."""
    pose = np.asarray(tool_pose_wxyz, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("Tool pose must be finite xyz + wxyz")
    if not math.isfinite(closure_progress) or not 0 <= closure_progress <= 1:
        raise ValueError("Closure progress must be in [0,1]")
    if np.linalg.norm(pose[3:]) <= 1e-10:
        raise ValueError("Tool orientation is degenerate")
    rotation = _quat_matrix(pose[3:]) @ _rot_z(roll_rad)
    gap = jaw_gap_m(closure_progress, radius_m)
    half = np.asarray(JAW_HALF_EXTENTS_M, dtype=float)
    boxes = []
    for sign in (-1, 1):
        centre_local = np.array([sign * (gap + 2 * half[0]) / 2, 0.0, JAW_HEIGHT_M])
        boxes.append((pose[:3] + rotation @ centre_local, rotation, half.copy()))
    return boxes


def box_corners(box):
    """The 8 world corners of one box, in a fixed order."""
    centre, rotation, half = box
    corners = np.array(list(itertools.product((-1, 1), repeat=3)), float) * half
    return centre + corners @ rotation.T


def project_points(points_w, camera_matrix, world_from_optical, pixel_centre=PIXEL_CENTRE):
    """Pixel index coordinates (pixel u has its centre at continuous u + ``pixel_centre``) and optical z."""
    transform = np.asarray(world_from_optical, dtype=float)
    rotation, t = transform[:3, :3], transform[:3, 3]
    local = (np.asarray(points_w, float) - t) @ rotation
    uvw = local @ np.asarray(camera_matrix, dtype=float).T
    return uvw[:, :2] / uvw[:, 2:3] - pixel_centre, local[:, 2]


def _hull(points_px):
    import cv2

    # The float32 hull is part of the registered rasterization.
    return cv2.convexHull(np.asarray(points_px, np.float32)).reshape(-1, 2).astype(float)


def polygon_signed_distance(hull, us, vs):
    """Euclidean distance of points (us, vs) to a convex polygon's boundary; negative inside."""
    px, py = np.asarray(us, float), np.asarray(vs, float)
    n = len(hull)
    centroid = hull.mean(axis=0)
    dmin = np.full(px.shape, np.inf)
    inside = np.ones(px.shape, bool)
    for k in range(n):
        a, b = hull[k], hull[(k + 1) % n]
        e = b - a
        length2 = float(e @ e)
        if length2 < 1e-18:
            continue
        t = np.clip(((px - a[0]) * e[0] + (py - a[1]) * e[1]) / length2, 0.0, 1.0)
        dmin = np.minimum(dmin, np.hypot(px - (a[0] + t * e[0]), py - (a[1] + t * e[1])))
        normal = np.array([e[1], -e[0]]) / math.sqrt(length2)
        if normal @ (centroid - a) > 0:
            normal = -normal
        inside &= (px - a[0]) * normal[0] + (py - a[1]) * normal[1] <= 0
    return np.where(inside, -dmin, dmin)


def _validated_camera(camera_matrix, world_from_optical):
    matrix = np.asarray(camera_matrix, dtype=float)
    transform = np.asarray(world_from_optical, dtype=float)
    if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
        raise ValueError("camera_matrix must be finite 3 x 3")
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError("world_from_optical must be finite 4 x 4")
    return matrix, transform


def jaw_projection(boxes, camera_matrix, world_from_optical, pixel_centre=PIXEL_CENTRE):
    """Per box: projected corners (index coordinates), their optical z and the corners' convex hull."""
    matrix, transform = _validated_camera(camera_matrix, world_from_optical)
    out = []
    for box in boxes:
        corners_px, z = project_points(box_corners(box), matrix, transform, pixel_centre)
        out.append({"corners_px": corners_px, "corners_z": z, "hull": _hull(corners_px)})
    return out


def jaw_mask(boxes, camera_matrix, world_from_optical, shape, margin_px=MASK_MARGIN_PX, pixel_centre=PIXEL_CENTRE):
    """Bool H x W mask: pixel centres within ``margin_px`` (signed distance) of either box's projected hull.

    Raises ValueError when any corner is behind or within ``MIN_CORNER_DEPTH_M``
    of the camera, where a hull silhouette is undefined (fail closed).
    """
    height, width = (int(v) for v in shape)
    projections = jaw_projection(boxes, camera_matrix, world_from_optical, pixel_centre)
    if any(not (proj["corners_z"] > MIN_CORNER_DEPTH_M).all() for proj in projections):
        raise ValueError("a jaw corner is behind or at the camera; the hull silhouette is undefined")
    mask = np.zeros((height, width), bool)
    for proj in projections:
        hull = proj["hull"]
        x0 = max(0, int(math.floor(hull[:, 0].min() - margin_px - 1)))
        x1 = min(width - 1, int(math.ceil(hull[:, 0].max() + margin_px + 1)))
        y0 = max(0, int(math.floor(hull[:, 1].min() - margin_px - 1)))
        y1 = min(height - 1, int(math.ceil(hull[:, 1].max() + margin_px + 1)))
        if x0 > x1 or y0 > y1:
            continue
        us, vs = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        mask[y0 : y1 + 1, x0 : x1 + 1] |= polygon_signed_distance(hull, us, vs) <= margin_px
    return mask


def jaw_signed_distance_px(boxes, camera_matrix, world_from_optical, points_px, pixel_centre=PIXEL_CENTRE):
    """Signed distance (px) of pixel positions to the nearest undilated jaw silhouette; negative inside."""
    points = np.atleast_2d(np.asarray(points_px, float))
    best = np.full(len(points), np.inf)
    for proj in jaw_projection(boxes, camera_matrix, world_from_optical, pixel_centre):
        best = np.minimum(best, polygon_signed_distance(proj["hull"], points[:, 0], points[:, 1]))
    return best


def jaw_mask_summary(boxes, mask, camera_matrix, world_from_optical, pixel_centre=PIXEL_CENTRE):
    """Telemetry: mask pixel count, inclusive bbox [x0, y0, x1, y1] (None when empty) and the 16 corners."""
    mask = np.asarray(mask, bool)
    corners = np.concatenate(
        [proj["corners_px"] for proj in jaw_projection(boxes, camera_matrix, world_from_optical, pixel_centre)]
    )
    ys, xs = np.nonzero(mask)
    bbox = None if xs.size == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    return {
        "mask_pixel_count": int(mask.sum()),
        "mask_bbox_xyxy": bbox,
        "corners_px": corners.astype(float).tolist(),
    }


def rotation_angle_deg(q0_wxyz, q1_wxyz):
    """Angle of the relative rotation between two orientations, in degrees."""
    r_a, r_b = _quat_matrix(q0_wxyz), _quat_matrix(q1_wxyz)
    cosine = (np.trace(r_a.T @ r_b) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def pose_change(reference_pose_wxyz, pose_wxyz):
    """Translation (m) and rotation (deg) of a tool pose relative to a reference pose."""
    reference, pose = np.asarray(reference_pose_wxyz, float), np.asarray(pose_wxyz, float)
    return float(np.linalg.norm(pose[:3] - reference[:3])), rotation_angle_deg(reference[3:], pose[3:])


def closure_hold_explanation(measurement, latched_by_jaw):
    """Whether a tracker measurement is a loss the jaw mask alone explains, and which branch.

    Only the three registered branches (``CLOSURE_HOLD_EXPLAINED_BRANCHES``) count;
    every other state, including any ``tracking`` measurement, is unexplained.
    """
    measurement = measurement if isinstance(measurement, dict) else {}
    state, reason = measurement.get("state"), measurement.get("reason")
    if state == JAW_MASK_OCCLUDED_STATE:
        return True, JAW_MASK_OCCLUDED_REASON
    if state == "tracking_lost" and reason == "optical_flow_failed":
        dropped = measurement.get("jaw_mask_dropped_valid")
        if (
            isinstance(dropped, int)
            and not isinstance(dropped, bool)
            and dropped > 0
            and measurement.get("jaw_mask_unmasked_flow_would_pass") is True
        ):
            return True, "flow_failed_only_after_mask_drop"
        return False, None
    if state == "tracking_lost" and reason == "explicit_initialization_required" and latched_by_jaw:
        return True, "latched_by_held_jaw_loss"
    return False, None
