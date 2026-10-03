"""Protect the registered constants of the depth-aware appearance check and its fail-closed jaw input."""

from __future__ import annotations

import json

import numpy as np

from isaaclab_pruning.perception import depth_appearance as da


def test_registered_constants_are_the_held_out_protocols_and_json_native():
    constants = da.registered_depth_appearance()
    assert constants == {
        "arm": "strict",
        "min_verified_fraction": 0.75,
        "min_same_surface_px": 20,
        "near_tolerance_m": 0.010,
        "max_near_fraction": 0.05,
        "max_median_abs_m": 0.003,
        "max_pixel_disagreement_px": 3.0,
        "same_surface_m": 0.025,
        "median_window_half": 1,
        "pixel_centre": 0.5,
        "jaw_mask_margin_px": 2.0,
    }
    assert json.loads(json.dumps(constants)) == constants


def test_each_arm_registers_the_same_thresholds_under_its_own_name():
    strict = da.registered_depth_appearance()
    assert da.registered_depth_appearance("strict") == strict
    agreement = da.registered_depth_appearance("agreement")
    assert agreement["arm"] == "agreement" and {**agreement, "arm": "strict"} == strict
    for bad in ("lenient", None, ""):
        try:
            da.registered_depth_appearance(bad)
        except ValueError:
            continue
        raise AssertionError(f"arm {bad!r} was accepted")


def test_frame_jaw_boxes_fail_closed_on_an_unusable_pose_or_progress():
    pose = np.array([1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0])
    assert len(da.frame_jaw_boxes(pose, 0.0, 0.5, 0.004)) == 2
    assert da.frame_jaw_boxes(pose, 0.0, 1.5, 0.004) is None
    assert da.frame_jaw_boxes(pose, 0.0, None, 0.004) is None
    assert da.frame_jaw_boxes(np.full(7, np.nan), 0.0, 0.0, 0.004) is None
