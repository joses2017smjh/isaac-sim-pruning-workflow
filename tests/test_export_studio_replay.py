"""Protect the published replay: recorded values only, no invented ones, budget held."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def exporter(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("export_studio_replay", tools / "export_studio_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _frame(index, **overrides):
    frame = {
        "index": index,
        "time_s": round(0.1 * (index + 1), 3),
        "phase": "vision_approach",
        "tool_position_m": [0.4, 0.3, 0.75],
        "command_pose_root_wxyz": [0.4, 0.3, 0.75, 1.0, 0.0, 0.0, 0.0],
        "contact_force_n": 0.0,
        "controller_source_frame_index": index - 1,
        "detachment_requested_after_capture": False,
        "visual_jaw_closure_progress": 0.0,
        "sensor_stop_reason": None,
        "tof_left_m": [[0.25] * 8 for _ in range(8)],
        "tof_right_m": [[0.25] * 8 for _ in range(8)],
        "tof_left_valid": [[True] * 8 for _ in range(8)],
        "tof_right_valid": [[True] * 8 for _ in range(8)],
        "live_vision": {
            "measurement": {
                "state": "tracking",
                "reason": None,
                "feature_count": 20,
                "confidence": 0.91,
                "depth_valid_fraction": 1.0,
                "pixel_xy": [210.0, 99.0],
            },
            "cut": {"phase": "approach", "stopped_reason": None},
        },
        "visual_servo_decision": {
            "state": "tracking",
            "reason": None,
            "delta_world_m": [0.001, -0.002, 0.0005],
            "remaining_distance_m": 0.25,
        },
    }
    frame.update(overrides)
    return frame


def test_invalid_tof_zones_are_null_not_a_number(exporter):
    values = [[0.25] * 8 for _ in range(8)]
    mask = [[True] * 8 for _ in range(8)]
    mask[0][0] = False
    values[1][1] = None
    grid = exporter._grid(values, mask)
    # An invalid zone must not reach the browser as a range it could colour.
    assert grid[0][0] is None
    assert grid[1][1] is None
    assert grid[2][2] == 250.0  # metres converted to millimetres


def test_non_finite_values_never_reach_the_page(exporter):
    assert exporter._round(float("nan")) is None
    assert exporter._round(float("inf")) is None
    assert exporter._round(None) is None
    assert exporter._round(1.23456, 2) == 1.23


def test_frame_record_keeps_proposed_and_applied_apart(exporter):
    record = exporter.frame_record(_frame(5))
    assert record["i"] == 5
    assert record["proposed_delta_mm"] == [1.0, -2.0, 0.5]
    assert record["applied_xyz"] == [0.4, 0.3, 0.75]
    assert record["source_frame"] == 4
    assert record["state"] == "tracking"
    assert record["decision_state"] == "tracking"


def test_stop_reason_is_surfaced_from_either_recorded_location(exporter):
    from_cut = _frame(9)
    from_cut["live_vision"]["cut"]["stopped_reason"] = "hazard_contact"
    assert exporter.frame_record(from_cut)["stopped_reason"] == "hazard_contact"

    from_sensor = _frame(9, sensor_stop_reason="sensor_lock_lost")
    from_sensor["live_vision"]["cut"]["stopped_reason"] = None
    assert exporter.frame_record(from_sensor)["stopped_reason"] == "sensor_lock_lost"


def _capture(tmp_path, frames=4, with_grade=False):
    directory = tmp_path / "capture"
    directory.mkdir()
    report = {
        "job_id": "fixture",
        "photometric_normalization": "raw",
        "blender_scene": {"daylight": {"preset": "source"}, "target": {"id": "fixture_SPUR_component_1"}},
    }
    (directory / "report.json").write_text(json.dumps(report), encoding="utf-8")
    (directory / "frames.json").write_text(
        json.dumps({"fps": 10, "frames": [_frame(index) for index in range(frames)]}), encoding="utf-8"
    )
    if with_grade:
        (directory / "sequence_grade.json").write_text(
            json.dumps({"ok": False, "checks": {"a": True, "b": False}, "failed_checks": ["b"]}), encoding="utf-8"
        )
    return directory


def test_export_records_provenance_and_the_stored_grade(tmp_path, exporter):
    capture = _capture(tmp_path, with_grade=True)
    summary = exporter.export_run(capture, tmp_path / "out", "run", "A label", "fail", encode=False)
    assert summary["grade"]["ok"] is False
    assert summary["grade"]["graded_here"] is False
    assert summary["grade"]["failed_checks"] == ["b"]

    payload = json.loads((tmp_path / "out/run/telemetry.json").read_text())
    assert payload["provenance"]["report_sha256"]
    assert payload["provenance"]["frames_sha256"]
    assert len(payload["frames"]) == payload["frame_count"] == 4
    assert len(payload["tof_mm"]["left"]) == 4
    # Every displayed frame must carry the index it came from.
    assert [frame["i"] for frame in payload["frames"]] == [0, 1, 2, 3]


def test_a_capture_without_a_stored_grade_is_graded_not_left_unknown(tmp_path, exporter):
    capture = _capture(tmp_path, with_grade=False)
    summary = exporter.export_run(capture, tmp_path / "out", "run", "A label", "fail", encode=False)
    assert summary["grade"] is not None
    # Marked so a reader knows the grade was produced here, not by the run.
    assert summary["grade"]["graded_here"] is True
    assert summary["grade"]["checks_total"] > 0


def test_payload_budget_is_enforced_rather_than_reported(tmp_path, exporter, capsys):
    capture = _capture(tmp_path)
    output = tmp_path / "out"
    code = exporter.main(
        ["--run", f"run={capture}=A label=fail", "--output", str(output), "--no-video", "--budget-bytes", "10"]
    )
    assert code == 1, "an oversized payload must fail, not merely warn"
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["within_budget"] is False

    ok = exporter.main(["--run", f"run2={capture}=A label=fail", "--output", str(output), "--no-video"])
    assert ok == 0
    capsys.readouterr()


def test_no_mesh_or_texture_is_ever_written(tmp_path, exporter):
    capture = _capture(tmp_path)
    output = tmp_path / "out"
    exporter.export_run(capture, output, "run", "A label", "fail", encode=False)
    written = {path.suffix.lower() for path in output.rglob("*") if path.is_file()}
    # Redistribution-restricted formats must never appear in a published bundle.
    assert written <= {".json", ".mp4"}
    assert not any(suffix in written for suffix in (".glb", ".gltf", ".stl", ".usd", ".usda", ".png", ".jpg"))
