from __future__ import annotations

import json

import numpy as np
import pytest

from isaaclab_pruning.demo.simulation import DemoConfig, cast_tof, make_scene, run_demo, run_episode


@pytest.fixture(scope="module")
def report():
    return run_demo()


def test_measured_approach_accepts_cut_only_after_geometry_checks(report):
    episode = report["episodes"][0]
    assert episode["metrics"]["outcome"] == "success"
    assert not episode["frames"][0]["checks"]["mouth_hit"]
    assert all(episode["metrics"]["final_checks"][key] for key in ("mouth_hit", "failure_clear", "perpendicular"))
    assert episode["metrics"]["final_target_distance_m"] < episode["metrics"]["initial_target_distance_m"]


def test_blackout_holds_tool_then_stops_and_clutter_rejects_cut(report):
    blackout, clutter = report["episodes"][1:]
    assert blackout["metrics"]["outcome"] == "sensor_lock_lost"
    tail = blackout["frames"][-4:]
    assert all(frame["valid_returns"] == 0 for frame in tail)
    np.testing.assert_allclose([frame["pose_w"] for frame in tail], [tail[0]["pose_w"]] * 4)
    assert clutter["metrics"]["outcome"] == "failure_zone_blocked"
    assert clutter["metrics"]["final_checks"]["other_wood_in_failure_zone"]


def test_rays_respond_to_actual_tool_motion():
    scene = make_scene("nominal")
    first = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    second = first.copy()
    second[2] = 0.005
    before, _, _ = cast_tof(first, scene)
    after, _, _ = cast_tof(second, scene)
    shared = np.isfinite(before) & np.isfinite(after)
    assert shared.any()
    assert np.median(before[shared] - after[shared]) > 0.001


def test_replay_is_strict_json_deterministic_and_honest_about_scope(report):
    json.dumps(report, allow_nan=False)
    assert report["episodes"][1] == run_episode("sensor_dropout")
    assert all(not e["metrics"]["arm_collision_evaluated"] for e in report["episodes"])
    assert "synthetic" in report["scope"]["fusion"]


def test_unknown_scenario_and_invalid_config_fail_early():
    with pytest.raises(ValueError, match="Unknown scenario"):
        make_scene("made_up")
    with pytest.raises(ValueError, match="Step counts"):
        DemoConfig(max_steps=0)
