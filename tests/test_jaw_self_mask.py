"""CPU contracts for the jaw self-mask: the controller's model of its own rendered jaw, never evidence."""

from __future__ import annotations

import inspect
import json
import math
import sys
import types

import numpy as np
import pytest

pytest.importorskip("cv2")

from isaaclab_pruning.perception import jaw_self_mask as jsm
from isaaclab_pruning.sim.blender_demo_scene import BlenderDemoScene

K = np.array([[320.0, 0.0, 240.0], [0.0, 320.0, 160.0], [0.0, 0.0, 1.0]])
SHAPE = (320, 480)


def test_registered_constants_are_the_registration_and_json_native():
    assert jsm.JAW_HALF_EXTENTS_M == (0.003, 0.0125, 0.015)
    assert jsm.JAW_OPEN_GAP_M == 0.032 and jsm.JAW_HEIGHT_M == 0.070
    assert jsm.MASK_MARGIN_PX == 2.0 and jsm.PIXEL_CENTRE == 0.5
    assert jsm.MIN_UNMASKED_PATCH_ELEMENTS == 140
    assert jsm.HOLD_MAX_TRANSLATION_M == 0.0005 and jsm.HOLD_MAX_ROTATION_DEG == 0.25
    for registered in (jsm.registered_jaw_self_mask(), jsm.registered_closure_hold()):
        assert json.loads(json.dumps(registered)) == registered
    assert jsm.registered_jaw_self_mask()["min_unmasked_patch_elements"] == 140
    assert jsm.registered_closure_hold() == {"max_translation_m": 0.0005, "max_rotation_deg": 0.25}


def test_a_known_pose_projects_to_known_pixels_under_index_plus_half():
    # Camera at the world origin looking along +Z: a point on the optical axis lands on the principal point
    # (240, 160) in continuous coordinates, i.e. pixel index (239.5, 159.5).
    points = np.array([[0.0, 0.0, 1.0], [0.1, -0.05, 0.5]])
    pixels, z = jsm.project_points(points, K, np.eye(4))
    np.testing.assert_allclose(pixels, [[239.5, 159.5], [303.5, 127.5]], atol=1e-12)
    np.testing.assert_allclose(z, [1.0, 0.5])
    # The camera pose is applied as x_cam = R^T (X - t).
    transform = np.eye(4)
    transform[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]  # optical +X is world +Y
    transform[:3, 3] = [1.0, 2.0, 3.0]
    pixels, z = jsm.project_points([[1.0, 2.1, 3.5]], K, transform)
    np.testing.assert_allclose(pixels, [[240 + 320 * 0.2 - 0.5, 159.5]], atol=1e-12)
    np.testing.assert_allclose(z, [0.5])


def _square_box(z=0.5, half=0.01, depth_half=0.001):
    return [(np.array([0.0, 0.0, z]), np.eye(3), np.array([half, half, depth_half]))]


def test_mask_is_pixel_centres_within_two_px_euclidean_signed_distance():
    # The near face (z = 0.499 m) silhouettes the box: a square of half width a = 5.8 px about index (239.5, 159.5),
    # so pixel centres sit 0.7, 1.7, 2.7 px outside it along each axis, never near the 2 px line.
    a = 5.8
    boxes = _square_box(half=a * 0.499 / 320.0)
    mask = jsm.jaw_mask(boxes, K, np.eye(4), SHAPE)
    us, vs = np.meshgrid(np.arange(SHAPE[1]), np.arange(SHAPE[0]))
    dx = np.maximum(np.abs(us - 239.5) - a, 0.0)
    dy = np.maximum(np.abs(vs - 159.5) - a, 0.0)
    expected = np.hypot(dx, dy) <= 2.0
    np.testing.assert_array_equal(mask, expected)
    # Euclidean, not a square dilation: (247, 167) is 1.7 px outside on both axes, 2.4 px away, so it is excluded;
    # (246, 167) is 0.7 and 1.7 px outside, 1.84 px away, so it is masked.
    assert not mask[167, 247] and mask[167, 246] and mask[167, 239]
    # A zero margin is the bare silhouette.
    assert jsm.jaw_mask(boxes, K, np.eye(4), SHAPE, margin_px=0.0).sum() == int(((dx == 0) & (dy == 0)).sum())
    summary = jsm.jaw_mask_summary(boxes, mask, K, np.eye(4))
    assert summary["mask_pixel_count"] == int(expected.sum())
    ys, xs = np.nonzero(expected)
    assert summary["mask_bbox_xyxy"] == [xs.min(), ys.min(), xs.max(), ys.max()]
    assert np.asarray(summary["corners_px"]).shape == (8, 2)
    json.dumps(summary)


def test_mask_signed_distance_helper_matches_the_rasterization():
    boxes = _square_box()
    a = 0.01 / 0.499 * 320.0
    distances = jsm.jaw_signed_distance_px(boxes, K, np.eye(4), [[239.5, 159.5], [239.5 + a + 1.0, 159.5]])
    np.testing.assert_allclose(distances, [-a, 1.0], atol=1e-4)


@pytest.mark.parametrize("z", [0.005, 0.0105, -0.2])
def test_a_corner_behind_or_at_the_camera_fails_closed(z):
    # depth_half 0.001: a box centred at 0.0105 has a corner at 0.0095 m, inside the 0.01 m guard.
    with pytest.raises(ValueError, match="behind or at the camera"):
        jsm.jaw_mask(_square_box(z=z), K, np.eye(4), SHAPE)


def test_invalid_camera_inputs_are_refused():
    with pytest.raises(ValueError):
        jsm.jaw_mask(_square_box(), np.eye(2), np.eye(4), SHAPE)
    with pytest.raises(ValueError):
        jsm.jaw_mask(_square_box(), K, np.full((4, 4), np.nan), SHAPE)


def test_roll_matches_set_proxy_closing_axis_tool():
    for axis in ([1.0, 0.0, 0.0], [0.3, -0.8, 0.0], [-2.0, 0.5, 1e-9], (0.5849625723371389, 0.8110602868866887, 0.0)):
        scene = object.__new__(BlenderDemoScene)
        scene.evidence = {}
        scene.set_proxy_closing_axis_tool(np.array(axis, dtype=float))
        assert jsm.proxy_roll_rad(axis) == scene._proxy_roll_rad
    caller = np.array([2.0, 0.0, 0.0])
    jsm.proxy_roll_rad(caller)
    np.testing.assert_array_equal(caller, [2.0, 0.0, 0.0])  # never normalized in place
    for bad in ([0.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, np.nan, 0.0], [1.0, 0.0]):
        with pytest.raises(ValueError):
            jsm.proxy_roll_rad(bad)


class _Quat:
    def __init__(self, real, imaginary):
        self.real, self.imaginary = float(real), np.asarray(imaginary, dtype=float)

    def __mul__(self, other):
        w1, v1, w2, v2 = self.real, self.imaginary, other.real, other.imaginary
        return _Quat(w1 * w2 - v1 @ v2, w1 * v2 + w2 * v1 + np.cross(v1, v2))


class _Op:
    def __init__(self):
        self.value = None

    def Set(self, value):  # noqa: N802 - mirrors the USD attribute API
        self.value = value


def _fake_gf():
    gf = types.SimpleNamespace(
        Vec3d=lambda *xyz: np.asarray(xyz, dtype=float),
        Vec3f=lambda *xyz: np.asarray(xyz, dtype=float),
        Quatf=_Quat,
    )
    return types.SimpleNamespace(Gf=gf)


@pytest.mark.parametrize("progress", [0.0, 0.1666, 0.5, 1.0])
def test_boxes_mirror_update_tool_proxy_pose_gap_and_offsets(monkeypatch, progress):
    monkeypatch.setitem(sys.modules, "pxr", _fake_gf())
    scene = object.__new__(BlenderDemoScene)
    scene.evidence = {}
    scene.target_radius_m = 0.0049202724425022526
    scene._proxy_position, scene._proxy_rotation = _Op(), _Op()
    scene._jaw_translations = [_Op(), _Op()]
    scene.set_proxy_closing_axis_tool([0.5849625723371389, 0.8110602868866887, 0.0])
    quat = np.array([-0.0037, -0.3524, 0.3494, 0.8682])
    pose = np.concatenate(([0.25, 0.63, 0.81], quat / np.linalg.norm(quat)))
    rendered = scene.update_tool_proxy(pose, progress)
    boxes = jsm.jaw_boxes(
        pose, jsm.proxy_roll_rad([0.5849625723371389, 0.8110602868866887, 0.0]), progress, 0.0049202724425022526
    )
    assert rendered["visual_gap_m"] == jsm.jaw_gap_m(progress, 0.0049202724425022526)
    q = scene._proxy_rotation.value
    w, (x, y, z) = q.real, q.imaginary
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    for (centre, box_rotation, half), op in zip(boxes, scene._jaw_translations, strict=True):
        np.testing.assert_allclose(box_rotation, rotation, atol=1e-12)
        np.testing.assert_allclose(centre, scene._proxy_position.value + rotation @ op.value, atol=1e-12)
        np.testing.assert_array_equal(half, jsm.JAW_HALF_EXTENTS_M)


def test_half_extents_and_gap_constants_match_the_renderer_source():
    source = inspect.getsource(BlenderDemoScene)
    assert "AddScaleOp().Set((0.006, 0.025, 0.030))" in source
    assert "gap = (1 - closure_progress) * 0.032 + closure_progress * 2 * self.target_radius_m" in source
    assert "op.Set(Gf.Vec3d(sign * (gap + 0.006) / 2, 0, 0.070))" in source
    assert tuple(2 * h for h in jsm.JAW_HALF_EXTENTS_M) == (0.006, 0.025, 0.030)


@pytest.mark.parametrize(
    "bad",
    [
        {"tool_pose_wxyz": [0, 0, 0, 1, 0, 0]},
        {"tool_pose_wxyz": [0, 0, np.nan, 1, 0, 0, 0]},
        {"tool_pose_wxyz": [0, 0, 0, 0, 0, 0, 0]},
        {"closure_progress": 1.5},
        {"closure_progress": float("nan")},
    ],
)
def test_boxes_refuse_what_the_renderer_refuses(bad):
    kwargs = {"tool_pose_wxyz": [0, 0, 0, 1, 0, 0, 0], "roll_rad": 0.0, "closure_progress": 0.0, "radius_m": 0.005}
    kwargs.update(bad)
    with pytest.raises(ValueError):
        jsm.jaw_boxes(**kwargs)


def test_pose_change_measures_translation_and_rotation():
    reference = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    half = math.radians(0.3) / 2
    moved = [0.0003, 0.0004, 0.0, math.cos(half), 0.0, 0.0, math.sin(half)]
    translation, rotation = jsm.pose_change(reference, moved)
    assert translation == pytest.approx(0.0005)
    assert rotation == pytest.approx(0.3, abs=1e-6)
    assert jsm.pose_change(reference, reference) == (0.0, 0.0)


@pytest.mark.parametrize(
    "measurement,latched,expected",
    [
        ({"state": "jaw_mask_occluded", "reason": "too_few_unmasked_patch_pixels"}, False, True),
        (
            {
                "state": "tracking_lost",
                "reason": "optical_flow_failed",
                "jaw_mask_dropped_valid": 1,
                "jaw_mask_unmasked_flow_would_pass": True,
            },
            False,
            True,
        ),
        (
            {
                "state": "tracking_lost",
                "reason": "optical_flow_failed",
                "jaw_mask_dropped_valid": 0,
                "jaw_mask_unmasked_flow_would_pass": True,
            },
            False,
            False,
        ),
        (
            {
                "state": "tracking_lost",
                "reason": "optical_flow_failed",
                "jaw_mask_dropped_valid": 2,
                "jaw_mask_unmasked_flow_would_pass": False,
            },
            False,
            False,
        ),
        ({"state": "tracking_lost", "reason": "optical_flow_failed"}, False, False),
        ({"state": "tracking_lost", "reason": "explicit_initialization_required"}, True, True),
        ({"state": "tracking_lost", "reason": "explicit_initialization_required"}, False, False),
        ({"state": "tracking_lost", "reason": "appearance_changed_or_occluded"}, True, False),
        ({"state": "tracking_lost", "reason": "low_confidence"}, True, False),
        ({"state": "tracking_lost", "reason": "world_target_jump_or_wrong_surface"}, True, False),
        ({"state": "invalid_depth", "reason": "depth_measurement_rejected"}, True, False),
        ({"state": "tracking", "jaw_mask_depth_window_in_mask": True}, True, False),
        (None, True, False),
    ],
)
def test_only_the_three_registered_losses_are_explained(measurement, latched, expected):
    explained, explanation = jsm.closure_hold_explanation(measurement, latched)
    assert explained is expected
    assert (explanation is not None) is expected
