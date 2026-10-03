"""The offline replay of recorded runs through the real controller: refusal rules, stop injection, exactness."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RECORDED_530 = ROOT / "artifacts/vision_robustness/planned-pose-gpu-r2-20260929/run_00_source_tree0_v530_planned_pose"


@pytest.fixture(scope="module")
def replay():
    spec = importlib.util.spec_from_file_location("replay_jaw_self_mask", ROOT / "tools/replay_jaw_self_mask.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_output_inside_the_repository_or_artifacts_or_existing_is_refused(replay, tmp_path):
    assert "inside" in replay.output_refusal(ROOT / "replay.json")
    assert "inside" in replay.output_refusal(ROOT / "artifacts/vision_robustness/replay.json")
    assert "inside" in replay.output_refusal(ROOT / "docs/evidence/new.json")
    existing = tmp_path / "exists.json"
    existing.write_text("{}")
    assert "overwrite" in replay.output_refusal(existing)
    assert replay.output_refusal(tmp_path / "new.json") is None
    with pytest.raises(SystemExit):
        replay.main(["--run-dir", str(tmp_path), "--output", str(ROOT / "replay.json")])
    assert not (ROOT / "replay.json").exists()


def _frame(cut_phase, external=None):
    return {"live_vision": {"external_stop_reason": external, "cut": {"phase": cut_phase}}}


def test_only_outside_stops_are_injected_never_the_runner_echo_of_the_cutters_own_stop(replay):
    report = {"initial_live_vision": {"external_stop_reason": None, "cut": {"phase": "approach"}}}
    frames = [
        _frame("approach"),
        _frame("stopped", "tof_minimum_clearance"),  # an outside stop first seen here
        _frame("stopped", "tof_minimum_clearance"),
    ]
    assert replay.genuine_external_stops(report, frames) == {1: "tof_minimum_clearance"}
    echo = [
        _frame("closing"),
        _frame("stopped"),
        _frame("stopped", "vision_invalid"),
        _frame("stopped", "vision_invalid"),
    ]
    assert replay.genuine_external_stops(report, echo) == {}
    rejected = {
        "initial_live_vision": {"external_stop_reason": "initial_target_not_visible", "cut": {"phase": "stopped"}}
    }
    assert replay.genuine_external_stops(rejected, [_frame("stopped", "initial_target_not_visible")]) == {
        -1: "initial_target_not_visible"
    }


def test_discovery_skips_running_batches_and_runs_without_frames(replay, tmp_path):
    for batch, run, frames in (
        ("planned-pose-gpu-r1-20260929", "run_00_a", True),
        ("jaw-shadow-eve-a-r1-20260930", "run_00_b", True),
        ("targets-source-20260923", "run_04_refused", False),
    ):
        path = tmp_path / batch / run
        (path / "frames").mkdir(parents=True)
        (path / "report.json").write_text("{}")
        (path / "frames.json").write_text('{"fps": 10, "frames": []}')
        if frames:
            (path / "frames/wrist_00000.png").write_bytes(b"")
    assert [p.name for p in replay.discover_runs(tmp_path)] == ["run_00_a"]


def test_measurement_criterion_is_exact_except_the_correlation_tolerance(replay):
    recorded = {"state": "tracking", "reason": None, "pixel_xy": [1.5, 2.5], "patch_correlation": 0.9}
    assert replay.same_measurement({**recorded, "patch_correlation": 0.9 + 9e-7}, recorded)
    assert not replay.same_measurement({**recorded, "patch_correlation": 0.9 + 2e-6}, recorded)
    assert not replay.same_measurement({**recorded, "pixel_xy": [1.5, 2.5000001]}, recorded)
    assert not replay.same_measurement({**recorded, "patch_correlation": None}, recorded)
    assert not replay.same_measurement({**recorded, "state": "invalid_depth"}, recorded)


@pytest.mark.skipif(not (RECORDED_530 / "frames.json").is_file(), reason="recorded planned-pose run not present")
def test_flags_off_reproduces_a_recorded_run_and_flags_on_hold_the_530_closure(replay):
    pytest.importorskip("cv2")
    off = replay.replay_run(RECORDED_530)
    assert off["reproduces_recording_through_recorded_stop"] and off["tracker_config_matches_recording"]
    assert off["comparison_after_recorded_stop"]["measurement_mismatch_frames"] == []
    assert off["comparison_through_recorded_stop"]["command_phase_mismatch_frames"] == []
    assert off["replay"]["stop_frame"] == off["recorded"]["stop_frame"] == 79
    # Every command equals the recorded one (to the 1 um replay tolerance), so frames are exact through the stop.
    assert off["first_command_divergence_frame"] is None
    assert all(row["exact_counterfactual"] for row in off["frames_detail"][:80])
    on = replay.replay_run(RECORDED_530, jaw_self_mask=True, closure_hold=True)
    rows = on["frames_detail"]
    assert on["replay"]["first_jaw_mask_occluded_frame"] == 76 and rows[76]["unmasked_patch_elements"] == 100
    assert [index for index in on["replay"]["held_frames"] if rows[index]["evaluable"]] == [76, 77, 78, 79]
    assert (on["replay"]["cut_without_hold_stop_frame"], on["replay"]["cut_without_hold_stop_reason"]) == (
        76,
        "vision_invalid",
    )
    assert [round(rows[index]["held_target_age_s"], 9) for index in (76, 77, 78, 79)] == [0.2, 0.3, 0.4, 0.5]
    assert all(rows[index]["waived_checks"] for index in (76, 77, 78, 79))
    # The recording stopped at 79, so frame 80 was rendered with another closure progress: never evidence.
    assert on["first_cut_decision_divergence_frame"] == 79 and not rows[80]["evaluable"]
    # The masked tracker's approach steps differ by micrometres, so its closure frames are approximate.
    assert 56 <= on["first_command_divergence_frame"] < 76 and not rows[76]["exact_counterfactual"]
    assert on["max_command_difference_mm_while_evaluable"] < 0.1
    json.dumps(on, allow_nan=False)


def test_rebuilt_tracker_config_must_match_every_recorded_option_and_default_the_rest(replay):
    from dataclasses import asdict

    report = {
        "blender_scene": {"target": {"id": "branch_7", "axis_w": [0.0, 0.0, 1.0], "radius_m": 0.004}},
        "initial_live_vision": {"closing_axis_tool": [1.0, 0.0, 0.0]},
        "initial_tool_pose_wxyz": [[1.0, 2.0, 3.0, 1.0, 0.0, 0.0, 0.0]],
        "photometric_normalization": "raw",
    }
    built = json.loads(
        json.dumps(
            asdict(
                replay.build_demo(
                    {**report, "tracker_config": {"motion_model": "translation"}}, False, False
                ).tracker.config
            )
        )
    )
    # A recording made before motion_model and the self-mask minimum existed: both now hold their defaults.
    older = {key: value for key, value in built.items() if key not in ("motion_model", "min_unmasked_patch_elements")}
    report["tracker_config"] = older
    assert replay.tracker_config_matches(replay.build_demo(report, False, False), report)
    report["tracker_config"] = {**older, "feature_quality_level": 0.02}
    assert not replay.tracker_config_matches(replay.build_demo(report, False, False), report)
    report["tracker_config"] = {**older, "an_option_this_code_lacks": 1}
    assert not replay.tracker_config_matches(replay.build_demo(report, False, False), report)
    report["tracker_config"] = {**older, "motion_model": "similarity"}
    assert replay.tracker_config_matches(replay.build_demo(report, False, False), report)


def test_flags_name_the_depth_arm_only_with_the_check_on_and_the_cli_refuses_an_arm_alone(replay, tmp_path):
    assert replay._flags(False, False, False, "strict") == {
        "jaw_self_mask": False,
        "closure_hold": False,
        "depth_appearance": False,
    }
    assert replay._flags(False, False, True, "agreement")["depth_appearance_arm"] == "agreement"
    with pytest.raises(SystemExit):
        replay.main(
            ["--run-dir", str(tmp_path), "--output", str(tmp_path / "out.json"), "--depth-appearance-arm", "agreement"]
        )
