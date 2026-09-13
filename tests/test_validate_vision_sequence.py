"""Independent sequence grading uses telemetry, never the renderer's success label."""

from __future__ import annotations

import copy
import json

import pytest

from tools import validate_vision_sequence as validator


@pytest.fixture
def sequence():
    report = {
        "job_id": "synthetic_cpu_fixture_not_gpu_evidence",
        "task_outcome": "untrusted_renderer_label",
        "stage": "complete",
        "rendering_ok": True,
        "frame_count": 5,
        "checks": {"frames_complete": True},
        "initial_tool_pose_wxyz": [[0, 0, 0, 1, 0, 0, 0]],
        "initial_piece_pose_wxyz": [0.3, 0, 1, 1, 0, 0, 0],
        "vision_command_count": 1,
        "stopped_reason": None,
    }
    frames = []
    for index, (phase, x, z) in enumerate(
        [
            ("vision_approach", 0.2, 1),
            ("align", 0.3, 1),
            ("simulated_closure", 0.3, 1),
            ("retreat", 0.1, 0.94),
            ("complete", 0.001, 0.8),
        ]
    ):
        frames.append(
            {
                "index": index,
                "time_s": (index + 1) / 10,
                "phase": phase,
                "controller_source_frame_index": index - 1,
                "tool_position_m": [x, 0, 0],
                "selected_piece_pose_wxyz": [0.3, 0, z, 1, 0, 0, 0],
                "detachment_requested_after_capture": index == 2,
                "capture_physics_steps_advanced": 0,
                "capture_timeline_advanced_s": 0,
                "contact_force_n": 0.0,
                "visual_servo_decision": {"state": "tracking" if index == 0 else "hold"},
                "live_vision": {
                    "external_stop_reason": None,
                    "cut": {
                        "phase": "retreat" if index >= 2 else "approach",
                        "closure_progress": 1.0 if index >= 2 else 0.0,
                        "detach_event": index == 2,
                        "detached": index >= 2,
                        "stopped_reason": None,
                        "certificate": {"ready_to_close": True, "safe_to_approach": True, "reasons": []},
                    },
                },
            }
        )
    return report, frames


def test_success_is_measured_and_strict_json_without_mutating_inputs(sequence):
    report, frames = sequence
    original = copy.deepcopy(sequence)
    grade = validator.grade_sequence(report, {"frames": frames})
    assert grade["ok"] is True
    assert grade["failed_checks"] == []
    assert all(grade["checks"].values())
    assert grade["source_task_outcome"] == "untrusted_renderer_label"
    assert grade["metrics"]["maximum_post_detach_drop_m"] == pytest.approx(0.2)
    assert grade["metrics"]["post_detach_home_distance_reduction_m"] == pytest.approx(0.299)
    assert grade["metrics"]["final_home_error_m"] == pytest.approx(0.001)
    assert "capture rate" in grade["evidence_scope"]
    assert "not certify" in grade["evidence_scope"]
    assert json.loads(json.dumps(grade, allow_nan=False)) == grade
    assert sequence == original


def test_incomplete_approach_returns_failed_grade_without_exception(sequence):
    report, frames = sequence
    frames = frames[:2]
    report.update(frame_count=2, task_outcome="vision_guided_simulated_detachment_and_retreat")
    grade = validator.grade_sequence(report, frames)
    assert grade["ok"] is False
    assert grade["checks"]["capture_complete_and_nonblank"] is True
    assert grade["checks"]["one_matching_detach_event_and_request"] is False
    assert grade["metrics"]["maximum_post_detach_drop_m"] is None
    assert grade["metrics"]["post_detach_home_distance_reduction_m"] is None


@pytest.mark.parametrize("frames", [[], [{}], {"frames": []}])
def test_missing_evidence_fails_closed(frames):
    grade = validator.grade_sequence({}, frames)
    assert grade["ok"] is False
    json.dumps(grade, allow_nan=False)


@pytest.mark.parametrize("event,requested", [(False, True), (True, False), (False, False)])
def test_event_and_actual_detachment_request_both_required(sequence, event, requested):
    report, frames = sequence
    frames[2]["live_vision"]["cut"]["detach_event"] = event
    frames[2]["detachment_requested_after_capture"] = requested
    assert validator.grade_sequence(report, frames)["checks"]["one_matching_detach_event_and_request"] is False


@pytest.mark.parametrize("key", ["detach_event", "request"])
def test_repeated_events_or_requests_fail(sequence, key):
    report, frames = sequence
    if key == "detach_event":
        frames[3]["live_vision"]["cut"][key] = True
    else:
        frames[3]["detachment_requested_after_capture"] = True
    assert validator.grade_sequence(report, frames)["ok"] is False


def test_request_must_match_event_frame(sequence):
    report, frames = sequence
    frames[2]["detachment_requested_after_capture"] = False
    frames[3]["detachment_requested_after_capture"] = True
    assert validator.grade_sequence(report, frames)["checks"]["one_matching_detach_event_and_request"] is False


@pytest.mark.parametrize(
    "field,value", [("closure_progress", 0.99), ("closure_progress", True), ("detached", False), ("phase", "closing")]
)
def test_event_requires_completed_detached_closure(sequence, field, value):
    report, frames = sequence
    frames[2]["live_vision"]["cut"][field] = value
    assert validator.grade_sequence(report, frames)["checks"]["detach_has_completed_certified_closure"] is False


@pytest.mark.parametrize(
    "field,value", [("ready_to_close", False), ("safe_to_approach", False), ("reasons", ["vision_invalid"])]
)
def test_event_requires_ready_certificate(sequence, field, value):
    report, frames = sequence
    frames[2]["live_vision"]["cut"]["certificate"][field] = value
    assert validator.grade_sequence(report, frames)["checks"]["detach_has_completed_certified_closure"] is False


@pytest.mark.parametrize("index,value", [(0, True), (4, False)])
def test_detached_flag_must_latch_once_at_event(sequence, index, value):
    report, frames = sequence
    frames[index]["live_vision"]["cut"]["detached"] = value
    assert validator.grade_sequence(report, frames)["checks"]["detached_state_consistent_with_event"] is False


def test_pre_event_drop_cannot_be_counted_as_post_event_drop(sequence):
    report, frames = sequence
    frames[1]["selected_piece_pose_wxyz"][2] = 0.5
    frames[3]["selected_piece_pose_wxyz"][2] = 1
    frames[4]["selected_piece_pose_wxyz"][2] = 1
    grade = validator.grade_sequence(report, frames)
    assert grade["checks"]["piece_stationary_before_release"] is False
    assert grade["checks"]["piece_dropped_after_release"] is False
    assert grade["metrics"]["maximum_post_detach_drop_m"] == 0


@pytest.mark.parametrize("index", [0, 2])
def test_attached_piece_must_not_move_in_any_axis_including_event_capture(sequence, index):
    report, frames = sequence
    frames[index]["selected_piece_pose_wxyz"][0] += 0.003
    assert validator.grade_sequence(report, frames)["checks"]["piece_stationary_before_release"] is False


def test_detachment_in_final_capture_has_no_post_event_evidence(sequence):
    report, frames = sequence
    frames = frames[:3]
    report["frame_count"] = 3
    grade = validator.grade_sequence(report, frames)
    assert grade["checks"]["one_matching_detach_event_and_request"] is True
    assert grade["checks"]["piece_dropped_after_release"] is False
    assert grade["checks"]["post_detach_retreat_toward_home"] is False


def test_retreat_requires_home_return_not_twenty_mm_away_from_target(sequence):
    report, frames = sequence
    frames[-1]["tool_position_m"][0] = 0.27
    grade = validator.grade_sequence(report, frames)
    assert grade["checks"]["post_detach_retreat_toward_home"] is True
    assert grade["checks"]["returned_home_and_final_phase_complete"] is False
    assert grade["ok"] is False


def test_retreat_away_from_home_cannot_pass(sequence):
    report, frames = sequence
    frames[-1]["tool_position_m"][0] = 0.4
    assert validator.grade_sequence(report, frames)["checks"]["post_detach_retreat_toward_home"] is False


def test_home_return_requires_complete_phase_and_retreat_after_event(sequence):
    report, frames = sequence
    frames[3]["phase"] = "align"
    frames[-1]["phase"] = "retreat"
    grade = validator.grade_sequence(report, frames)
    assert grade["checks"]["post_detach_retreat_toward_home"] is True
    assert grade["checks"]["returned_home_and_final_phase_complete"] is False
    frames[-1]["phase"] = "complete"
    assert validator.grade_sequence(report, frames)["checks"]["post_detach_retreat_toward_home"] is False


@pytest.mark.parametrize("where", ["report", "sensor", "cut", "external", "phase", "final_report_cut"])
def test_any_recorded_stop_invalidates_otherwise_completed_sequence(sequence, where):
    report, frames = sequence
    if where == "report":
        report["stopped_reason"] = "tof_minimum_clearance"
    elif where == "sensor":
        frames[3]["sensor_stop_reason"] = "both_tof_missing_4_frames"
    elif where == "cut":
        frames[3]["live_vision"]["cut"]["stopped_reason"] = "hazard_contact"
    elif where == "external":
        frames[3]["live_vision"]["external_stop_reason"] = "tracking_error"
    elif where == "phase":
        frames[3]["phase"] = "stopped_failure"
    else:
        report["final_live_vision"] = {"cut": {"stopped_reason": "vision_invalid"}}
    grade = validator.grade_sequence(report, frames)
    assert grade["checks"]["no_recorded_stops"] is False
    assert grade["metrics"]["recorded_stops"]


@pytest.mark.parametrize("force", [5.01, -1, None, True])
def test_contact_limit_and_measurement_validity(sequence, force):
    report, frames = sequence
    frames[1]["contact_force_n"] = force
    assert validator.grade_sequence(report, frames)["checks"]["captured_contact_within_limit"] is False


def test_contact_limit_is_inclusive(sequence):
    report, frames = sequence
    frames[1]["contact_force_n"] = 5.0
    assert validator.grade_sequence(report, frames)["ok"] is True


@pytest.mark.parametrize(
    "field,value",
    [("controller_source_frame_index", 2), ("controller_source_frame_index", True), ("index", 5), ("time_s", 0.1)],
)
def test_frame_causality_and_order(sequence, field, value):
    report, frames = sequence
    frames[2][field] = value
    assert validator.grade_sequence(report, frames)["ok"] is False


@pytest.mark.parametrize(
    "field,value",
    [
        ("capture_physics_steps_advanced", 1),
        ("capture_timeline_advanced_s", 0.01),
        ("capture_physics_steps_advanced", None),
    ],
)
def test_rendering_must_not_advance_physics(sequence, field, value):
    report, frames = sequence
    frames[1][field] = value
    assert validator.grade_sequence(report, frames)["checks"]["capture_did_not_advance_physics"] is False


@pytest.mark.parametrize(
    "field,value", [("rendering_ok", False), ("rendering_ok", 1), ("frame_count", 6), ("stage", "record")]
)
def test_capture_completeness_cannot_be_overridden_by_success_label(sequence, field, value):
    report, frames = sequence
    report[field] = value
    report["task_outcome"] = "vision_guided_simulated_detachment_and_retreat"
    assert validator.grade_sequence(report, frames)["checks"]["capture_complete_and_nonblank"] is False


@pytest.mark.parametrize("count", [0, 2, None, True])
def test_applied_vision_command_count_must_match_actual_command_frames(sequence, count):
    report, frames = sequence
    report["vision_command_count"] = count
    assert validator.grade_sequence(report, frames)["checks"]["vision_commands_applied_and_counted"] is False


def test_requested_but_overridden_command_is_not_applied(sequence):
    report, frames = sequence
    frames[0]["visual_servo_decision"]["state"] = "hold"
    assert validator.grade_sequence(report, frames)["checks"]["vision_commands_applied_and_counted"] is False


@pytest.mark.parametrize("key", ["selected_piece_pose_wxyz", "tool_position_m"])
def test_missing_pose_cannot_be_replaced_with_report_summary(sequence, key):
    report, frames = sequence
    del frames[3][key]
    report["metrics"] = {"selected_piece_drop_observed": True, "retreat_return_error_m": 0}
    assert validator.grade_sequence(report, frames)["ok"] is False


@pytest.mark.parametrize("report,frames", [([], []), ({}, {}), ({}, [1]), ({}, None)])
def test_malformed_top_level_inputs_raise_clear_value_error(report, frames):
    with pytest.raises(ValueError):
        validator.grade_sequence(report, frames)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_json_is_rejected(sequence, bad):
    report, frames = sequence
    frames[0]["contact_force_n"] = bad
    with pytest.raises(ValueError):
        validator.grade_sequence(report, frames)


def _write_capture(path, sequence):
    path.mkdir()
    report, frames = sequence
    (path / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (path / "frames.json").write_text(json.dumps({"frames": frames}), encoding="utf-8")


def test_cli_writes_new_output_and_prints_successful_grade(tmp_path, sequence, capsys):
    source, output = tmp_path / "capture", tmp_path / "grade.json"
    _write_capture(source, sequence)
    assert validator.main(["--input-dir", str(source), "--output", str(output)]) == 0
    grade = json.loads(output.read_text())
    assert grade["ok"] is True
    assert json.loads(capsys.readouterr().out) == grade


def test_cli_incomplete_capture_returns_one_without_traceback(tmp_path, sequence, capsys):
    report, frames = sequence
    report["stage"] = "record"
    source = tmp_path / "capture"
    _write_capture(source, (report, frames[:2]))
    assert validator.main(["--input-dir", str(source)]) == 1
    captured = capsys.readouterr()
    assert json.loads(captured.out)["ok"] is False
    assert captured.err == ""


def test_cli_refuses_existing_output_and_preserves_it(tmp_path, sequence, capsys):
    source, output = tmp_path / "capture", tmp_path / "grade.json"
    _write_capture(source, sequence)
    output.write_text("existing user evidence\n", encoding="utf-8")
    with pytest.raises(SystemExit) as error:
        validator.main(["--input-dir", str(source), "--output", str(output)])
    assert error.value.code == 2
    assert "Refusing to overwrite" in capsys.readouterr().err
    assert output.read_text() == "existing user evidence\n"


def test_cli_nonfinite_json_is_malformed_not_incomplete(tmp_path, sequence, capsys):
    source = tmp_path / "capture"
    sequence[0]["malformed"] = float("nan")
    _write_capture(source, sequence)
    with pytest.raises(SystemExit) as error:
        validator.main(["--input-dir", str(source)])
    assert error.value.code == 2
    assert "Nonfinite JSON" in capsys.readouterr().err
