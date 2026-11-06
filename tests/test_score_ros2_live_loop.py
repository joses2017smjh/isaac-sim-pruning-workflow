"""The live-loop scorer on a synthetic batch: it passes what it should and catches what it must."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
ROOT_XYZW = [0.0, 0.0, 0.7, 0.0, 0.0, 0.0, 1.0]
FPS = 10


@pytest.fixture
def scorer(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("score_ros2_live_loop", TOOLS / "score_ros2_live_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(index, delay, hold_after=None):
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
            "operator_hold_after_cycles": hold_after,
        },
    }


def _gated(index):
    return [0.30 + 0.001 * index, 0.5, 0.80, 1.0, 0.0, 0.0, 0.0]


def _run(run_dir, row, *, ok=True, frames=4, hold_at=None):
    """A run whose every applied command is the gated one; ``hold_at`` = first operator-held cycle index."""
    run_dir.mkdir(parents=True)
    initial = _gated(0)
    records, decisions = [], []
    for index in range(frames):
        held = hold_at is not None and index + 1 >= hold_at
        command = _gated(index)  # the gated answer for cycle `index`
        root_action = [command[0], command[1], command[2] - 0.7, *command[3:]]
        stamp = int(round((index + 1) * 1e9 / FPS))
        decision = {"stamp_ns": stamp, "passed": not held, "reason": "operator_hold" if held else "",
                    "operator_hold": held, "publish_wall_ns": 1_000 + 100 * (index + 1)}  # fmt: skip
        decisions.append(decision)
        records.append(
            {
                "index": index,
                "phase": "stopped_failure" if hold_at is not None and index >= hold_at else "vision_approach",
                "command_pose_root_wxyz": root_action,
                "ros2_loop": {
                    "gate_decision_next_cycle": decision,
                    "gated_command_next_cycle_wxyz": _gated(index + 1),
                    "applied_command_wall_ns": 1_050 + 100 * index,
                },
            }
        )
    decisions.insert(0, {"stamp_ns": 0, "passed": True, "reason": "", "operator_hold": False, "publish_wall_ns": 1_000})
    report = {
        "task_outcome": "vision_guided_simulated_detachment_and_retreat",
        "initial_control_state": {"root_pose_w_xyzw": [ROOT_XYZW]},
        "ros2_loop": {
            "enabled": True,
            "delay_steps": row["ros2_loop"]["delay_steps"],
            "in_process_sensor_guard": "disabled; the C++ gate checks ToF",
            "timed_out": None,
            "initial_cycle": {"gated_command_wxyz": initial},
        },
    }
    (run_dir / "experiment_result.json").write_text(json.dumps({"ok": ok}))
    (run_dir / "report.json").write_text(json.dumps(report))
    (run_dir / "frames.json").write_text(json.dumps({"frames": records}))
    (run_dir / "sequence_grade.json").write_text(json.dumps({"failed_checks": [] if ok else ["no_recorded_stops"]}))
    (run_dir / "ros2_loop_decisions.jsonl").write_text("\n".join(json.dumps(d) for d in decisions) + "\n")
    return decisions


def _panel_record(run_dir, decisions, press_wall_ns):
    events = [{"event": "start", "wall_ns": 0}]
    for decision in decisions:
        events.append(
            {"event": "decision", "stamp_ns": decision["stamp_ns"], "passed": decision["passed"],
             "pass_hold_changed": decision["operator_hold"] and decision["stamp_ns"] == min(
                 d["stamp_ns"] for d in decisions if d["operator_hold"]),
             "receive_wall_ns": decision["publish_wall_ns"] + 2, "applied_wall_ns": decision["publish_wall_ns"] + 3}
        )  # fmt: skip
        events.append({"event": "rendered", "wall_ns": decision["publish_wall_ns"] + 5,
                       "shown_decision_stamp_ns": decision["stamp_ns"]})  # fmt: skip
        events.append({"event": "frame", "stamp_ns": decision["stamp_ns"]})
    events.append({"event": "synthesized_click", "wall_ns": press_wall_ns - 1})
    events.append({"event": "hold_pressed", "wall_ns": press_wall_ns})
    (run_dir / "operator_panel_record.jsonl").write_text("\n".join(json.dumps(e) for e in events) + "\n")


def _batch(tmp_path, scorer):
    from run_vision_experiment import run_label

    batch = tmp_path / "batch"
    rows = [_row(0, 0), _row(1, 6), _row(2, 0, hold_after=2)]
    batch.mkdir()
    (batch / "plan.json").write_text(json.dumps({"protocol": "p", "code_revision": "c", "runs": rows}))
    _run(batch / run_label(0, rows[0]), rows[0])
    _run(batch / run_label(1, rows[1]), rows[1], ok=False)
    decisions = _run(batch / run_label(2, rows[2]), rows[2], ok=False, hold_at=2)
    # The click lands after cycle 2's decision (wall 1200), before cycle 3's is published (1300).
    _panel_record(batch / run_label(2, rows[2]), decisions, press_wall_ns=1_250)
    return batch, rows


def test_a_batch_that_meets_the_protocol_is_scored_as_such(tmp_path, scorer):
    batch, _ = _batch(tmp_path, scorer)
    result = scorer.score(batch)
    verdicts = result["verdicts"]
    assert verdicts["R1_ros2_drove_the_arm"]["supported"], [r["r1"] for r in result["runs"]]
    assert verdicts["R2_same_result_at_0ms"] == {"supported": True, "passes": 1, "runs": 1}
    assert verdicts["R3_delay"]["first_failing_delay_ms"] == 100.0
    assert verdicts["R4_hold_stops_the_arm"]["supported"], verdicts["R4_hold_stops_the_arm"]
    assert verdicts["R4_hold_stops_the_arm"]["control_cycles_to_hold"] <= 1
    assert verdicts["R5_panel_keeps_up"]["supported"], verdicts["R5_panel_keeps_up"]


def test_an_applied_command_that_is_not_the_gated_one_refutes_r1(tmp_path, scorer):
    from run_vision_experiment import run_label

    batch, rows = _batch(tmp_path, scorer)
    path = batch / run_label(0, rows[0]) / "frames.json"
    frames = json.loads(path.read_text())
    frames["frames"][2]["command_pose_root_wxyz"][0] += 0.001  # 1 mm off the gate's command
    path.write_text(json.dumps(frames))
    r1 = scorer.score(batch)["verdicts"]["R1_ros2_drove_the_arm"]
    assert not r1["supported"] and r1["runs_ok"] == 2


def test_a_passing_decision_after_the_press_refutes_r4(tmp_path, scorer):
    from run_vision_experiment import run_label

    batch, rows = _batch(tmp_path, scorer)
    run_dir = batch / run_label(2, rows[2])
    lines = [json.loads(line) for line in (run_dir / "ros2_loop_decisions.jsonl").read_text().splitlines()]
    late = copy.deepcopy(lines[-1])
    late.update(passed=True, operator_hold=False, publish_wall_ns=10_000, stamp_ns=late["stamp_ns"] + 100_000_000)
    lines.append(late)
    (run_dir / "ros2_loop_decisions.jsonl").write_text("\n".join(json.dumps(d) for d in lines) + "\n")
    assert not scorer.score(batch)["verdicts"]["R4_hold_stops_the_arm"]["supported"]


def test_a_decision_the_panel_never_showed_refutes_r5(tmp_path, scorer):
    from run_vision_experiment import run_label

    batch, rows = _batch(tmp_path, scorer)
    path = batch / run_label(2, rows[2]) / "operator_panel_record.jsonl"
    events = [json.loads(line) for line in path.read_text().splitlines()]
    dropped = next(e for e in events if e.get("event") == "decision" and e["stamp_ns"] > 0)
    events.remove(dropped)
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n")
    r5 = scorer.score(batch)["verdicts"]["R5_panel_keeps_up"]
    assert not r5["supported"] and dropped["stamp_ns"] in r5["decisions_missing_from_panel"]
