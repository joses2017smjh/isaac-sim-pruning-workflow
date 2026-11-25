"""The long-episode delay scorer on a synthetic batch: what supports, refutes and leaves a prediction unscored."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
ROOT_XYZW = [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0]
FRAMES = 300


@pytest.fixture
def scorer(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("score_delay_episode", TOOLS / "score_delay_episode.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(index, delay):
    return {
        "index": index,
        "daylight": "source",
        "photometric_normalization": "raw",
        "target_tree_index": 1,
        "component_first_vertex": 14944,
        "strategy": {"name": "baseline", "mode": "straight", "standoff_m": 0.0, "max_step_m": 0.004},
        "ros2_loop": {
            "delay_steps": delay,
            "delay_ms": round(delay * 1000 / 60, 1),
            "operator_hold_after_cycles": None,
        },
    }


def _gated(index):
    return [0.30 + 0.0001 * index, 0.5, 0.80, 1.0, 0.0, 0.0, 0.0]


def _run(run_dir, row, *, cut, home, ok=True, frames=FRAMES, applied_offset=0.0):
    run_dir.mkdir(parents=True)
    records = []
    for index in range(frames):
        command = _gated(index)
        applied = [command[0] + applied_offset, command[1], command[2] - 0.7, *command[3:]]
        phase = "complete" if home is not None and index >= home else "vision_approach"
        records.append(
            {
                "index": index,
                "phase": phase,
                "command_pose_root_wxyz": applied,
                "live_vision": {"cut": {"detach_event": index == cut}},
                "ros2_loop": {
                    "gate_decision_next_cycle": {"passed": True},
                    "gated_command_next_cycle_wxyz": _gated(index + 1),
                },
            }
        )
    report = {
        "task_outcome": "vision_guided_simulated_detachment_and_retreat",
        "initial_control_state": {"root_pose_w_xyzw": [ROOT_XYZW]},
        "ros2_loop": {
            "enabled": True,
            "delay_steps": row["ros2_loop"]["delay_steps"],
            "in_process_sensor_guard": "disabled; the C++ gate checks ToF",
            "timed_out": None,
            "initial_cycle": {"gated_command_wxyz": _gated(0)},
        },
    }
    checks = {f"check_{i}": True for i in range(17)}
    if not ok:
        checks["check_16"] = False
    grade = {"checks": checks, "failed_checks": [] if ok else ["check_16"], "metrics": {"final_home_error_m": 0.001}}
    (run_dir / "experiment_result.json").write_text(json.dumps({"ok": ok, "configuration_matches": True}))
    (run_dir / "report.json").write_text(json.dumps(report))
    (run_dir / "frames.json").write_text(json.dumps({"frames": records}))
    (run_dir / "sequence_grade.json").write_text(json.dumps(grade))


def _batch(tmp_path, scorer, overrides=None):
    from run_vision_experiment import run_label

    overrides = overrides or {}
    batch = tmp_path / "delay-episode-test"
    rows = [_row(index, delay) for index, delay in enumerate([4, 6, 4, 6, 4, 6])]
    plan = {"protocol": scorer.PROTOCOL, "frames": FRAMES, "code_revision": "abc", "runs": rows}
    batch.mkdir(parents=True)
    (batch / "plan.json").write_text(json.dumps(plan))
    for row in rows:
        expected = scorer.EXPECTED[row["ros2_loop"]["delay_steps"]]
        kwargs = {"cut": expected["cut_frame"], "home": expected["home_frame"], **overrides.get(row["index"], {})}
        _run(batch / run_label(row["index"], row), row, **kwargs)
    return batch


def test_the_predicted_outcome_supports_every_prediction(scorer, tmp_path):
    result = scorer.score(_batch(tmp_path, scorer))
    assert {name: v["verdict"] for name, v in result["verdicts"].items()} == {
        "D0_ros2_drove_the_arm": "supported",
        "D1_all_pass_with_a_30_s_episode": "supported",
        "D2_home_frame": "supported",
        "D3_first_200_frames_unchanged": "supported",
    }


def test_a_failed_run_refutes_d1(scorer, tmp_path):
    result = scorer.score(_batch(tmp_path, scorer, {3: {"ok": False}}))
    assert result["verdicts"]["D1_all_pass_with_a_30_s_episode"]["verdict"] == "refuted"
    assert result["verdicts"]["D2_home_frame"]["verdict"] == "supported"


def test_home_outside_the_tolerance_or_never_reached_refutes_d2(scorer, tmp_path):
    late = scorer.score(_batch(tmp_path / "a", scorer, {0: {"home": 233 + 11}}))
    assert late["verdicts"]["D2_home_frame"]["verdict"] == "refuted"
    inside = scorer.score(_batch(tmp_path / "b", scorer, {0: {"home": 233 - 10}}))
    assert inside["verdicts"]["D2_home_frame"]["verdict"] == "supported"
    never = scorer.score(_batch(tmp_path / "c", scorer, {1: {"home": None}}))
    assert never["verdicts"]["D2_home_frame"]["verdict"] == "refuted"


def test_a_moved_cut_refutes_d3(scorer, tmp_path):
    result = scorer.score(_batch(tmp_path, scorer, {5: {"cut": 142 + 3}}))
    assert result["verdicts"]["D3_first_200_frames_unchanged"]["verdict"] == "refuted"


def test_an_unscored_run_keeps_predictions_from_being_supported_without_refuting(scorer, tmp_path):
    short = scorer.score(_batch(tmp_path / "a", scorer, {2: {"frames": 200, "home": None}}))
    assert short["runs"][2]["scored"] is False
    assert short["verdicts"]["D1_all_pass_with_a_30_s_episode"]["verdict"] == "not supported: unscored runs"
    assert short["verdicts"]["D2_home_frame"]["verdict"] == "not supported: unscored runs"
    ungated = scorer.score(_batch(tmp_path / "b", scorer, {4: {"applied_offset": 0.001}}))
    assert ungated["runs"][4]["scored"] is False
    assert ungated["verdicts"]["D0_ros2_drove_the_arm"]["verdict"] == "refuted"


def test_a_batch_registered_elsewhere_is_refused(scorer, tmp_path):
    batch = _batch(tmp_path, scorer)
    plan = json.loads((batch / "plan.json").read_text())
    (batch / "plan.json").write_text(json.dumps({**plan, "frames": 200}))
    with pytest.raises(ValueError):
        scorer.score(batch)
