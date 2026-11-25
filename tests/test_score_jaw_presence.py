"""tools/score_jaw_presence.py on a two-capture example: one usable closure and one with the tool beneath the ground."""

from __future__ import annotations

import copy
import gzip
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TARGET = "tree0_SPUR_component_8175"
ABOVE, BENEATH = "artifacts/demo/above", "artifacts/demo/beneath"


@pytest.fixture(scope="module")
def tools():
    sys.path.insert(0, str(TOOLS))
    out = {}
    for name in ("gate_replay", "score_jaw_presence"):
        spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
        out[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(out[name])
    return out


def _records(replay, label):
    rows = []
    phases = [("approach", "approach", 0.0), ("align", "approach", 0.0), ("closing", "align", 0.0)]
    phases += [("closing", "closing", 0.25), ("closing", "closing", 0.5), ("closing", "closing", 0.75)]
    for index, (phase, prior, progress) in enumerate(phases):
        rows.append(
            {
                "capture": label,
                "index": index,
                "stamp_ns": 1_000_000_000 + index * 100_000_000,
                "tool_pose_wxyz": [0.002, 0.0, -0.07, 1.0, 0.0, 0.0, 0.0],
                "mouth_offset_tool_m": [0.0, 0.0, 0.07],
                "closing_axis_tool": [1.0, 0.0, 0.0],
                "contact_force_n": 0.0,
                "tof_left": [0.3] * 64,
                "tof_right": [0.3] * 64,
                "target": {
                    "id": TARGET,
                    "semantic": "branch",
                    "valid": True,
                    "source": "live",
                    "vision_time_s": 1.0 + index * 0.1,
                    "held_time_s": None,
                    "position": [0.0, 0.0, 0.0],
                    "axis": [0.0, 1.0, 0.0],
                    "radius_m": 0.005,
                },
                "prior_cut_phase": prior,
                "prior_detached": False,
                "proposal": {"cut_phase": phase, "cut_detached": False, "closure_progress": progress},
                "python": {"tof_missing_frames": 0, "tof_min_range_m": 0.3},
            }
        )
    for row, verdict in zip(rows, replay.replay(rows)):
        row["python"] = {key: verdict[key] for key in verdict if key != "jaw_presence_ok"}
    return rows


def _reading(replay, records, beneath):
    n = len(records)
    return {
        "capture": records[0]["capture"],
        "index": [r["index"] for r in records],
        "stamp_ns": [r["stamp_ns"] for r in records],
        "pose_digest": [replay.pose_digest(r["tool_pose_wxyz"]) for r in records],
        "beams": ["...TT...."] * n,
        "target_beams": ["000110000"] * n,
        "back_beams": ["........."] * n,
        "back_target_beams": ["000000000"] * n,
        "beneath": ["100" if beneath else "000"] * n,
        "first_ready": 0,
        "displaced_at_first_ready": [
            {"axis": "tool", "offset_mm": -20.0, "beams": ".........", "target": "000000000"},
            {"axis": "lateral", "offset_mm": 30.0, "beams": "W........", "target": "000000000"},
            {"axis": "tool", "offset_mm": 10.0, "beams": "....T....", "target": "000010000"},
        ],
    }


def _scenario(replay, records, reading, key):
    joined = replay.attach_beams(records, reading, key)
    expected = replay.replay(joined, jaw_min_beams=1, jaw_key="jaw_beams", node_rules=True)
    index, reason = replay.first_hold(expected, joined)
    return {
        "cycles": len(joined),
        "missing_outputs": 0,
        "mismatched_fields": {},
        "first_hold_index": index,
        "first_hold_reason": reason,
        "expected_first_hold_index": index,
        "expected_first_hold_reason": reason,
        "max_gated_jaw": max(v["gated_jaw"] for v in expected),
        "expected_max_gated_jaw": max(v["gated_jaw"] for v in expected),
        "recorded_max_jaw": 0.75,
        "blocked_cycles": sum(replay.blocked_count(r["jaw_beams"]) > 0 for r in joined),
        "forwarding_problems": {},
        "held_cycles_commanding_motion": 0,
        "gate_returncode": 0,
    }


def _write(tmp_path, tools, mutate=None):
    replay = tools["gate_replay"]
    captures = {label: _records(replay, label) for label in (ABOVE, BENEATH)}
    readings = {ABOVE: _reading(replay, captures[ABOVE], False), BENEATH: _reading(replay, captures[BENEATH], True)}
    cycles, digest = tmp_path / "cycles.jsonl.gz", hashlib.sha256()
    with gzip.open(cycles, "wt", encoding="utf-8") as stream:
        for records in captures.values():
            for record in records:
                line = json.dumps(record, sort_keys=True)
                digest.update(line.encode())
                stream.write(line + "\n")
    beams = tmp_path / "jaw_beams.jsonl.gz"
    with gzip.open(beams, "wt", encoding="utf-8") as stream:
        for reading in readings.values():
            stream.write(json.dumps(reading) + "\n")
    fault = {"ok": True, "first_hold_index": 60, "first_hold_reason": "vision_invalid"}
    parity = {"cycles_sha256": digest.hexdigest(), "parity": {"totals": {}}, "faults": {}}
    parity["parity"]["totals"] = {
        "cycles": 12,
        "matched_cycles": 12,
        "mismatched_cycles": 0,
        "cpp_holds": 0,
        "captures": 2,
    }
    parity["faults"] = {"scenarios": {"vision_lost": fault}, "stopping_faults": 1, "stopping_faults_held": 1}
    jaw = {
        "cycles_sha256": digest.hexdigest(),
        "jaw_beams_sha256": hashlib.sha256(beams.read_bytes()).hexdigest(),
        "jaw_presence": {
            "min_beams": 1,
            "captures": {
                ABOVE: {
                    "recorded": _scenario(replay, captures[ABOVE], readings[ABOVE], "beams"),
                    "all_clear": _scenario(replay, captures[ABOVE], readings[ABOVE], "........."),
                    "tool_short": _scenario(replay, captures[ABOVE], readings[ABOVE], "back_beams"),
                }
            },
            "input_fault_capture": ABOVE,
            "input_faults": {
                "jaw_reading_dropped": {
                    "ok": True,
                    "expected_reason": "input_missing:jaw_presence",
                    "first_hold_reason": "input_missing:jaw_presence",
                    "first_hold_index": 20,
                    "held_through_end": True,
                }
            },
        },
    }
    documents = {"baseline": copy.deepcopy(parity), "harness_off": copy.deepcopy(parity), "harness_jaw": jaw}
    if mutate:
        mutate(documents)
    paths = {}
    for name, document in documents.items():
        paths[name] = tmp_path / f"{name}.json"
        paths[name].write_text(json.dumps(document))
    return cycles, beams, paths


def _score(tmp_path, tools, mutate=None):
    cycles, beams, paths = _write(tmp_path, tools, mutate)
    out = tmp_path / "verdicts.json"
    argv = ["--cycles", str(cycles), "--beams", str(beams), "--baseline-cycles", str(cycles), "--output", str(out)]
    argv += ["--baseline", str(paths["baseline"]), "--harness-off", str(paths["harness_off"])]
    argv += ["--harness-jaw", str(paths["harness_jaw"])]
    assert tools["score_jaw_presence"].main(argv) == 0
    return json.loads(out.read_text())


def test_a_clean_replay_supports_every_prediction(tmp_path, tools):
    result = _score(tmp_path, tools)
    assert {name: item["verdict"] for name, item in result["verdicts"].items()} == {
        name: "supported" for name in ("J1", "J2", "J3", "J4", "J6")
    }
    assert result["captures"]["tool_above_the_ground"] == 1 and result["captures"]["tool_beneath_the_ground"] == 1
    j3 = result["verdicts"]["J3"]
    # The jaw is announced closing in cycle 2 and first moves in cycle 3; the hold lands on cycle 3.
    assert j3["first_hold_reasons"] == {"jaw_empty_during_closure": 1}
    assert j3["cycles_from_first_closing_to_hold"] == {"1": 1}
    audit = result["descriptive"]["ground_audit"]
    assert audit["by_target"] == {TARGET: 1} and audit["captures"][0]["capture"] == BENEATH
    j5 = result["descriptive"]["J5_displaced_at_first_ready"]["total"]
    assert (j5["poses"], j5["target_between_jaws"], j5["blocked_without_the_target"]) == (3, 1, 1)
    assert j5["target_without_a_blocked_beam"] == 0


def test_a_jaw_that_moved_refutes_the_rule(tmp_path, tools):
    def mutate(documents):
        documents["harness_jaw"]["jaw_presence"]["captures"][ABOVE]["all_clear"]["max_gated_jaw"] = 0.25

    result = _score(tmp_path, tools, mutate)
    assert result["verdicts"]["J3"]["verdict"] == "refuted"
    assert result["verdicts"]["J3"]["captures_whose_jaw_advanced"] == [ABOVE]
    assert result["verdicts"]["J2"]["verdict"] == "supported"


def test_a_changed_fault_or_a_node_mismatch_refutes_their_predictions(tmp_path, tools):
    def mutate(documents):
        documents["harness_off"]["faults"]["scenarios"]["vision_lost"]["first_hold_index"] = 61
        documents["harness_jaw"]["jaw_presence"]["captures"][ABOVE]["recorded"]["mismatched_fields"] = {"reason": 1}
        documents["harness_jaw"]["jaw_presence"]["input_faults"]["jaw_reading_dropped"]["ok"] = False

    result = _score(tmp_path, tools, mutate)
    assert result["verdicts"]["J1"]["verdict"] == "refuted" and result["verdicts"]["J1"]["faults"]["changed"] == [
        "vision_lost"
    ]
    assert result["verdicts"]["J2"]["verdict"] == "refuted" and result["verdicts"]["J6"]["verdict"] == "refuted"


def test_a_blocked_beam_with_the_tool_short_refutes_the_channel_prediction(tmp_path, tools):
    def mutate(documents):
        documents["harness_jaw"]["jaw_presence"]["captures"][ABOVE]["tool_short"]["max_gated_jaw"] = 0.75

    result = _score(tmp_path, tools, mutate)
    assert result["verdicts"]["J4"]["verdict"] == "refuted" and result["verdicts"]["J4"]["closures_accepted"] == [ABOVE]


def test_inputs_from_a_different_run_leave_the_jaw_predictions_untested(tmp_path, tools):
    def mutate(documents):
        documents["harness_jaw"]["jaw_beams_sha256"] = "0" * 64

    result = _score(tmp_path, tools, mutate)
    assert [result["verdicts"][name]["verdict"] for name in ("J2", "J3", "J4", "J6")] == ["untested"] * 4
    assert result["verdicts"]["J1"]["verdict"] == "supported"
