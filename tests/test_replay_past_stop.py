"""Protect the replay past the stop: comparison rules, pose metrics, refusals and the fidelity bookkeeping.

Synthetic throughout; the input-hash test needs the recorded runs and skips without them.
"""

from __future__ import annotations

import copy
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "artifacts/vision_robustness"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("replay_past_stop", ROOT / "tools/replay_past_stop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _rot_z(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_x(deg):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _quat_x(deg):
    half = math.radians(deg) / 2
    return [math.cos(half), math.sin(half), 0.0, 0.0]


# ------------------------------------------------------------------------------------------------ compare
def test_numbers_within_tolerance_are_equal_and_the_largest_difference_is_reported(tool):
    recorded = {"state": "tracking", "reason": None, "pixel_xy": [1.5, 2.5], "confidence": 0.5, "depth": {"m": 0.2}}
    replayed = copy.deepcopy(recorded)
    replayed["confidence"] += 5e-7
    replayed["depth"]["m"] += 2e-7
    result = tool.compare(replayed, recorded)
    assert result["equal"]
    assert result["max_abs_difference"]["confidence"] == pytest.approx(5e-7, rel=1e-3)
    assert result["max_abs_difference"]["depth"] == pytest.approx(2e-7, rel=1e-3)

    replayed["confidence"] = 0.5 + 2e-6
    result = tool.compare(replayed, recorded)
    assert not result["equal"]
    assert [item["path"] for item in result["differences"]] == ["confidence"]


def test_exact_fields_tolerate_no_difference(tool):
    recorded = {"state": "tracking", "reason": None, "pixel_xy": [1.5, 2.5], "confidence": 0.5}
    moved = dict(recorded, pixel_xy=[1.5, 2.5 + 1e-9])
    result = tool.compare(moved, recorded)
    assert not result["equal"]
    assert result["differences"][0]["path"] == "pixel_xy[1]"
    assert not tool.compare(dict(recorded, reason="low_confidence"), recorded)["equal"]
    assert not tool.compare(dict(recorded, state="tracking_lost"), recorded)["equal"]
    # The same nudge is accepted on a field that is not exact.
    assert tool.compare(dict(recorded, confidence=0.5 + 1e-9), recorded)["equal"]


def test_types_missing_keys_lists_and_ignore(tool):
    assert not tool.compare({"a": True}, {"a": 1})["equal"]  # a bool never equals a number
    assert not tool.compare({"a": None}, {"a": 0.0})["equal"]
    assert tool.compare({"a": 1}, {"a": 1.0})["equal"]
    result = tool.compare({"a": 1, "b": 2}, {"a": 1, "c": 3})
    assert result["only_in_replayed"] == ["b"] and result["only_in_recorded"] == ["c"] and not result["equal"]
    assert not tool.compare({"a": [1, 2]}, {"a": [1, 2, 3]})["equal"]
    assert tool.compare({"a": 1, "arm": "strict"}, {"a": 1, "arm": "agreement"}, ignore=("arm",))["equal"]
    assert not tool.compare({"a": 1.0}, {"a": 1.0 + 1e-12}, tolerance=0.0)["equal"]


def test_listed_differences_are_capped_but_counted(tool):
    replayed = {f"k{i}": float(i) for i in range(100)}
    recorded = {f"k{i}": float(i) + 1.0 for i in range(100)}
    result = tool.compare(replayed, recorded)
    assert result["difference_count"] == 100
    assert len(result["differences"]) == tool.MAX_LISTED_DIFFERENCES


def test_event_core_keeps_only_the_shared_depth_test(tool):
    event = {
        "ncc": 0.1,
        "accept": True,
        "stats": {"agreement_fraction": 1.0},
        "strict_confidence": 0.03,
        "agreement_confidence": 0.375,
        "arm": "agreement",
        "decision": "continues",
        "appearance_gate_value": 1.0,
        "ended_by": None,
        "result_state": "tracking",
        "result_reason": None,
        "reached_confidence_gate": True,
        "frame": 67,
        "recorded": {},
        "replayed": {},
        "recorded_cut_phase": "stopped",
        "after_recorded_detach": False,
    }
    assert tool.event_core(event) == {
        "ncc": 0.1,
        "accept": True,
        "stats": {"agreement_fraction": 1.0},
        "strict_confidence": 0.03,
        "agreement_confidence": 0.375,
    }


def test_target_shift_and_merge_max(tool):
    assert tool.target_shift_mm(None, [0, 0, 0]) is None
    assert tool.target_shift_mm([0.0, 0.0, 0.0], [0.003, 0.004, 0.0]) == pytest.approx(5.0)
    assert tool.merge_max({"a": 1.0}, {"a": 0.5, "b": 2.0}) == {"a": 1.0, "b": 2.0}


# ------------------------------------------------------------------------------------------------ poses
def test_wrist_rotation_angle_is_exact_on_nearly_orthonormal_matrices(tool):
    a = np.eye(3) * (1 + 3e-7)  # the recorded matrices are orthonormal only to ~4e-7
    b = _rot_z(0.01) * (1 - 2e-7)
    assert tool.rotation_matrix_angle_deg(a, b) == pytest.approx(0.01, abs=1e-6)
    assert tool.rotation_matrix_angle_deg(b, b) == pytest.approx(0.0, abs=1e-9)
    naive = math.degrees(math.acos(min(1.0, (np.trace(a.T @ b) - 1) / 2)))
    assert abs(naive - 0.01) > 1e-3  # why the closest rotation and atan2 are used
    assert tool.orthonormality_error(a) == pytest.approx(6e-7, rel=1e-3)


def test_quaternion_angle_matches_the_committed_rotation_angle(tool):
    rng = np.random.default_rng(5)
    for _ in range(20):
        q0, q1 = rng.normal(size=4), rng.normal(size=4)
        assert tool.quaternion_angle_deg(q0, q1) == pytest.approx(tool.rotation_angle_deg(q0, q1), abs=1e-6)
    q = rng.normal(size=4)
    assert tool.quaternion_angle_deg(q, -q) == pytest.approx(0.0, abs=1e-9)  # the double cover
    assert tool.quaternion_angle_deg(_quat_x(0.0), _quat_x(0.03)) == pytest.approx(0.03, abs=1e-9)


def _frame(position_m, tool_deg, wrist_position_m, wrist_deg):
    return {
        "tool_pose_wxyz": [*position_m, *_quat_x(tool_deg)],
        "wrist_position_w_m": list(wrist_position_m),
        "wrist_rotation_w_ros": _rot_x(wrist_deg).tolist(),
    }


def test_pose_difference_and_largest(tool):
    a = _frame([0.3, 0.5, 0.8], 0.0, [0.2, 0.4, 0.7], 0.0)
    b = _frame([0.301, 0.5, 0.8], 0.03, [0.2, 0.402, 0.7], 0.05)
    c = _frame([0.3, 0.5, 0.8005], 0.01, [0.2, 0.4, 0.7], 0.0)
    difference = tool.pose_difference(a, b)
    assert difference["tool_position_mm"] == pytest.approx(1.0)
    assert difference["tool_rotation_deg"] == pytest.approx(0.03, abs=1e-6)
    assert difference["tool_rotation_deg_quaternion_atan2"] == pytest.approx(0.03, abs=1e-9)
    assert difference["wrist_position_mm"] == pytest.approx(2.0)
    assert difference["wrist_rotation_deg"] == pytest.approx(0.05, abs=1e-9)
    best = tool.largest({"a|b": difference, "a|c": tool.pose_difference(a, c)})
    assert best["tool_position_mm"] == {"value": pytest.approx(1.0), "pair": "a|b"}
    assert set(best) == set(tool.POSE_METRICS)


def test_window_max_reports_value_frame_and_pair(tool):
    def row(frame, value):
        metrics = {metric: {"value": value, "pair": f"p{frame}"} for metric in tool.POSE_METRICS}
        floor = dict.fromkeys(tool.POSE_METRICS, value / 10)
        return {"frame": frame, "largest_shadowed_vs_no_shadow": metrics, "no_shadow_r1_vs_r2_floor": floor}

    per_frame = [row(60, 0.3), row(67, 1.05), row(70, 1.03)]
    best = tool.window_max(per_frame, 60, 77, "largest_shadowed_vs_no_shadow")
    assert best["tool_position_mm"] == {"value": 1.05, "frame": 67, "pair": "p67"}
    floor = tool.window_max(per_frame, 68, 72, "no_shadow_r1_vs_r2_floor")
    assert floor["tool_position_mm"] == {"value": pytest.approx(0.103), "frame": 70}


# ------------------------------------------------------------------------------------------------ refusals
def test_output_inside_the_repository_or_artifacts_or_existing_is_refused(tool, tmp_path):
    assert tool.output_refusal(ROOT / "out.json")
    assert tool.output_refusal(ROOT / "artifacts/out.json")
    existing = tmp_path / "exists.json"
    existing.write_text("{}")
    assert tool.output_refusal(existing)
    assert tool.output_refusal(tmp_path / "missing_dir/out.json")
    assert tool.output_refusal(tmp_path / "new.json") is None


def test_the_study_is_fixed(tool):
    assert tool.ARMS == ("strict", "agreement")
    assert (tool.EXPECTED_STOP, tool.LAST_FRAME, tool.TABLE_FIRST) == (67, 72, 60)
    assert len(tool.SHADOWED) == 5 and len(tool.NO_SHADOW) == 2
    assert all("v14944" in spec["run"] for spec in [*tool.SHADOWED.values(), *tool.NO_SHADOW.values()])
    assert {spec["live_tracker"] for spec in tool.SHADOWED.values()} == {"base", "strict"}


# ------------------------------------------------------------------------------------------------ fidelity
def _measurement(state="tracking", reason=None, pixel=(10.0, 20.0), ncc=0.95, confidence=0.9):
    return {
        "state": state,
        "reason": reason,
        "pixel_xy": list(pixel),
        "patch_correlation": ncc,
        "confidence": confidence,
        "feature_count": 16,
        "target_position_world_m": None if state != "tracking" else [0.25, 0.63, 0.81],
    }


def _event(arm):
    core = {
        "ncc": 0.05,
        "accept": True,
        "reasons": [],
        "failed_conditions": [],
        "not_evaluated_conditions": [],
        "stats": {"agreement_fraction": 1.0, "median_abs_m": 5e-5},
        "jaw": {"touches": False, "reason": None},
        "feature_ratio": 0.375,
        "depth_valid_fraction": 1.0,
        "strict_confidence": 0.019,
        "agreement_confidence": 0.375,
    }
    decision = "continues" if arm == "agreement" else "stops_downstream"
    return {**core, "arm": arm, "decision": decision, "appearance_gate_value": 1.0 if arm == "agreement" else 0.35}


def _fake_replay(live_tracker="base"):
    """Frames 0-2, recorded stop at 2: base lost on appearance, strict lost on confidence, agreement continues."""
    stop = 2
    before = [_measurement(pixel=(10.0 + i, 20.0)) for i in range(stop)]
    base_stop = _measurement("tracking_lost", "appearance_changed_or_occluded", ncc=0.05, confidence=0.0)
    strict_stop = dict(
        _measurement("tracking_lost", "low_confidence", pixel=(12.0, 20.0), ncc=0.05, confidence=0.0),
        depth_appearance=_event("strict"),
    )
    agreement_stop = dict(
        _measurement(pixel=(12.0, 20.0), ncc=0.05, confidence=0.375), depth_appearance=_event("agreement")
    )
    recorded_stop = base_stop if live_tracker == "base" else strict_stop
    results = {
        "base": [*copy.deepcopy(before), base_stop],
        "strict": [*copy.deepcopy(before), strict_stop],
        "agreement": [*copy.deepcopy(before), agreement_stop],
    }
    raw_events = {arm: {stop: _event(arm)} for arm in ("strict", "agreement")}
    records = {arm: {stop: {"frame": stop, **_event(arm)}} for arm in ("strict", "agreement")}
    init = {"state": "initialized", "reason": "awaiting_rgb_depth_measurement", "feature_count": 16}
    preview = _measurement(pixel=(9.0, 20.0))
    replay = {
        "recorded_stop": stop,
        "recorded_init": copy.deepcopy(init),
        "recorded_preview": copy.deepcopy(preview),
        "init": {name: copy.deepcopy(init) for name in results},
        "preview": {name: copy.deepcopy(preview) for name in results},
        "results": results,
        "raw_events": raw_events,
        "records": records,
        "frame_info": [{"recorded_measurement": copy.deepcopy(m)} for m in [*before, recorded_stop]],
    }
    return replay


def _committed_run(tool, replay):
    stop = replay["recorded_stop"]
    return {
        "replayed_through_frame": stop,
        "base_reproduces_recording": True,
        "grey_failures": [stop],
        "divergence": {name: {"mismatch_frames": []} for name in tool.TRACKERS},
        "measurement_at_recorded_stop": {
            name: tool.rda._brief(replay["results"][name][stop]) for name in tool.TRACKERS
        },
        "events": {arm: [copy.deepcopy(replay["records"][arm][stop])] for arm in tool.ARMS},
    }


def test_fidelity_of_a_faithful_replay(tool):
    replay = _fake_replay("base")
    spec = {"run": "fake/run_00", "live_tracker": "base", "committed_replay": None}
    result = tool.fidelity(spec, replay, _committed_run(tool, replay), {})
    checks = result["checks"]
    assert checks.pop("stop_is_expected_frame") is False  # the fake stops at 2, the study at 67
    assert all(checks.values()), checks
    assert not result["passed"]
    assert result["per_tracker"]["base"]["frame_at_stop_vs_recorded"]["equal"]
    assert not result["per_tracker"]["agreement"]["frame_at_stop_vs_recorded"]["equal"]  # the expected divergence
    assert result["event_at_stop"]["agreement_vs_strict_replayed_core"]["equal"]


def test_fidelity_reports_a_drift_the_committed_criterion_misses(tool):
    replay = _fake_replay("base")
    replay["results"]["agreement"][1]["confidence"] += 1e-3  # outside 1e-6, invisible to state/reason/pixel/NCC
    spec = {"run": "fake/run_00", "live_tracker": "base", "committed_replay": None}
    result = tool.fidelity(spec, replay, _committed_run(tool, replay), {})
    frames = result["per_tracker"]["agreement"]["frames_before_stop"]
    assert [item["frame"] for item in frames["mismatching_frames"]] == [1]
    assert frames["committed_criterion_mismatches"] == []
    assert frames["max_abs_difference_per_field"]["confidence"] == pytest.approx(1e-3)
    assert not result["checks"]["all_trackers_reproduce_before_stop"]

    replay["results"]["base"][0]["pixel_xy"][0] += 0.25
    result = tool.fidelity(spec, replay, _committed_run(tool, replay), {})
    assert result["per_tracker"]["base"]["frames_before_stop"]["committed_criterion_mismatches"] == [
        {"frame": 0, "fields": ["pixel_xy"]}
    ]


def test_fidelity_against_a_live_strict_event_and_a_failed_cross_check(tool):
    replay = _fake_replay("strict")
    spec = {"run": "fake/run_00", "live_tracker": "strict", "committed_replay": None}
    result = tool.fidelity(spec, replay, _committed_run(tool, replay), {})
    assert result["checks"]["live_tracker_reproduces_frame_at_stop"]
    assert result["checks"]["agreement_event_equals_recorded_live_event"]
    assert result["checks"]["strict_event_equals_recorded_live_event"]

    live = replay["frame_info"][2]["recorded_measurement"]["depth_appearance"]
    live["stats"]["median_abs_m"] += 1e-4  # the live depth test saw something else
    result = tool.fidelity(spec, replay, {"error": "RuntimeError('boom')"}, {})
    assert not result["checks"]["agreement_event_equals_recorded_live_event"]
    assert not result["checks"]["equals_committed_replay_run"]
    assert result["committed_replay_run_cross_check"]["error"] == "RuntimeError('boom')"


def test_fidelity_against_committed_evidence(tool):
    replay = _fake_replay("base")
    spec = {"run": "batch/run_00", "live_tracker": "base", "committed_replay": "evidence.json"}
    entry = {
        "run": "batch/run_00",
        "events": {arm: [copy.deepcopy(replay["records"][arm][2])] for arm in tool.ARMS},
        "measurement_at_recorded_stop": {name: tool.rda._brief(replay["results"][name][2]) for name in tool.TRACKERS},
    }
    evidence = {"evidence.json": {"code_revision": "abc", "runs": [entry]}}
    result = tool.fidelity(spec, replay, _committed_run(tool, replay), evidence)
    assert result["checks"]["events_equal_committed_evidence"]
    assert result["checks"]["measurements_equal_committed_evidence"]
    entry["events"]["agreement"][0]["agreement_confidence"] = 0.5
    result = tool.fidelity(spec, replay, _committed_run(tool, replay), evidence)
    assert not result["checks"]["events_equal_committed_evidence"]


def test_committed_hashes_from_replay_evidence_and_loop_verdicts(tool):
    evidence = {
        "replay.json": {"runs": [{"run": "b/r", "source_sha256": {"report.json": "r1", "frames.json": "f1"}}]},
        tool.DEPTH_LOOP_VERDICTS: {
            "runs": [{"batch": "loop", "run_directory": "run_00", "report_sha256": "r2", "frames_sha256": "f2"}]
        },
    }
    assert tool.committed_hashes({"run": "b/r", "committed_replay": "replay.json"}, evidence) == {
        "source": "replay.json",
        "report.json": "r1",
        "frames.json": "f1",
    }
    assert tool.committed_hashes({"run": "loop/run_00", "committed_replay": None}, evidence)["frames.json"] == "f2"


# ------------------------------------------------------------------------------------------------ summaries
def test_grey_check_and_cut_gate_arithmetic(tool):
    assert tool.grey_check(None, 0.35) == "not_reached"
    assert tool.grey_check(0.35, 0.35) == "passed"
    assert tool.grey_check(0.349, 0.35) == "failed"
    config = {
        "mouth_position_tolerance_m": 0.008,
        "alignment_tolerance_deg": 15.0,
        "max_stable_speed_m_s": 0.025,
        "max_target_shift_m": 0.01,
        "stable_frames": 4,
        "closure_duration_s": 0.6,
    }
    rows = {
        67: {
            "state": "tracking",
            "mouth_distance_mm": 5.0,
            "perpendicularity_error_deg": 0.35,
            "mouth_step_mm": 3.84,
            "target_shift_from_previous_frame_mm": 0.27,
        },
        68: {
            "state": "tracking",
            "mouth_distance_mm": 8.5,
            "perpendicularity_error_deg": 0.34,
            "mouth_step_mm": 1.9,
            "target_shift_from_previous_frame_mm": None,
        },
    }
    out = tool.cut_gate_arithmetic(config, 10.0, rows, [67, 68])
    assert out["thresholds"]["mouth_position_tolerance_mm"] == pytest.approx(8.0)
    assert out["thresholds"]["max_stable_speed_mm_s"] == pytest.approx(25.0)
    assert out["frames"]["67"]["mouth_speed_mm_s"] == pytest.approx(38.4)
    assert not out["frames"]["67"]["below_stable_speed"] and out["frames"]["67"]["within_mouth_tolerance"]
    assert out["frames"]["68"]["below_stable_speed"] and not out["frames"]["68"]["within_mouth_tolerance"]
    assert not out["frames"]["68"]["within_target_shift"]


def test_pointer_check(tool):
    def frames(at68, later):
        return {"68": {"ncc": at68}, **{str(i): {"ncc": value} for i, value in zip(range(69, 73), later)}}

    out = tool.pointer_check(
        {"a": frames(0.8166, [0.9785, 0.998, 0.9967, 0.9978]), "b": frames(0.8796, [0.9892, 0.9988, 0.9987, 0.9996])}
    )
    assert out["ncc_68_range"] == [0.8166, 0.8796]
    assert out["lowest_ncc_69_72"] == {"value": 0.9785, "run": "a", "frame": 69}
    assert out["frames_69_72_below_0_98"] == [["a", 69, 0.9785]]
    assert out["lowest_ncc_70_72"] == pytest.approx(0.9967)


# ------------------------------------------------------------------------------------------------ recordings
@pytest.mark.skipif(not RUNS.is_dir(), reason="needs artifacts/vision_robustness (the recorded runs)")
def test_recorded_inputs_match_the_committed_evidence(tool):
    specs = [*tool.SHADOWED.values(), *tool.NO_SHADOW.values()]
    if not all((RUNS / spec["run"] / "frames.json").is_file() for spec in specs):
        pytest.skip("the evening 14944 recordings are not all present")
    evidence = {name: json.loads((ROOT / name).read_text()) for name in tool.EVIDENCE_FILES}
    for spec in specs:
        committed = tool.committed_hashes(spec, evidence)
        assert committed["report.json"] == tool.sha256(RUNS / spec["run"] / "report.json"), spec["run"]
        assert committed["frames.json"] == tool.sha256(RUNS / spec["run"] / "frames.json"), spec["run"]
