"""tools/jaw_reading_rules.py and tools/score_jaw_reading.py: one centred spur and nothing else in the gap."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
CLEAR = [-1.0, -1.0]
SPUR = [10.0, 10.0]  # in a 32 mm gap: 12 mm thick, centred
EDGE = [1.0, 22.0]  # 9 mm thick, hugging the first jaw face: centre 10.5 mm off the middle
THICK = [3.0, 4.0]  # 25 mm thick: a spur with something else beside it


@pytest.fixture(scope="module")
def tools():
    sys.path.insert(0, str(TOOLS))
    out = {}
    for name in ("gate_replay", "jaw_reading_rules", "score_jaw_reading"):
        spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
        out[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(out[name])
    return out


def _grid(*beams):
    """Nine beams: the given ones on the tip row (indices 6 to 8), the rest clear unless more are given."""
    grid = [CLEAR] * 9
    for index, beam in zip((6, 7, 8, 3, 4, 5), beams):
        grid[index] = beam
    return grid


def test_a_beam_is_clear_spur_like_or_foreign(tools):
    rules = tools["jaw_reading_rules"]
    assert rules.beam_kind(-1.0, -1.0, 32.0, 6.7) == rules.CLEAR
    assert rules.beam_kind(10.0, 10.0, 32.0, 6.7) == rules.SPUR
    assert rules.beam_kind(1.0, 22.0, 32.0, 6.7) == rules.FOREIGN  # off centre
    assert rules.beam_kind(3.0, 4.0, 32.0, 6.7) == rules.FOREIGN  # thicker than the claimed spur
    assert rules.beam_kind(10.0, -1.0, 32.0, 6.7) == rules.FOREIGN  # only one side sees it
    # The limits themselves: 6 mm off centre and 2 mm over the diameter are still spur-like.
    assert (
        rules.beam_kind(16.0, 4.0, 32.0, 6.7) == rules.SPUR and rules.beam_kind(16.1, 3.9, 32.0, 6.7) == rules.FOREIGN
    )
    assert rules.beam_kind(8.3, 8.3, 32.0, 6.7) == rules.SPUR and rules.beam_kind(8.2, 8.2, 32.0, 6.7) == rules.FOREIGN


def test_the_rule_wants_two_spur_beams_and_nothing_else(tools):
    rules = tools["jaw_reading_rules"]
    assert rules.read(_grid(SPUR, SPUR, SPUR), 32.0, 6.7) == {"accept": True, "spur_beams": 3, "foreign_beams": 0}
    assert rules.accepts(_grid(SPUR, SPUR), 32.0, 6.7)
    assert not rules.accepts(_grid(SPUR), 32.0, 6.7)  # one beam is what the refuted rule accepted
    assert not rules.accepts(_grid(SPUR, SPUR, SPUR, EDGE), 32.0, 6.7)  # a spur, and other wood beside it
    assert not rules.accepts(_grid(THICK, THICK, THICK), 32.0, 6.7)
    assert not rules.accepts([CLEAR] * 9, 32.0, 6.7)


def _records(replay, label, target):
    rows = []
    phases = [("approach", "approach", 0.0), ("align", "approach", 0.0), ("closing", "align", 0.0)]
    phases += [("closing", "closing", 0.25), ("closing", "closing", 0.5)]
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
                    "id": target,
                    "semantic": "branch",
                    "valid": True,
                    "source": "live",
                    "vision_time_s": 1.0 + index * 0.1,
                    "held_time_s": None,
                    "position": [0.0, 0.0, 0.0],
                    "axis": [0.0, 1.0, 0.0],
                    "radius_m": 0.0067,
                },
                "prior_cut_phase": prior,
                "prior_detached": False,
                "proposal": {"cut_phase": phase, "cut_detached": False, "closure_progress": progress},
                "python": {"tof_missing_frames": 0, "tof_min_range_m": 0.3},
            }
        )
    return rows


def _reading(replay, records, recorded, short, beneath=False):
    n = len(records)
    return {
        "capture": records[0]["capture"],
        "index": [r["index"] for r in records],
        "stamp_ns": [r["stamp_ns"] for r in records],
        "pose_digest": [replay.pose_digest(r["tool_pose_wxyz"]) for r in records],
        "gap_mm": [32.0] * n,
        "beams": ["".join("." if b == CLEAR else "W" for b in recorded)] * n,
        "ranges": [recorded] * n,
        "back_beams": ["".join("." if b == CLEAR else "W" for b in short)] * n,
        "back_ranges": [short] * n,
        "beneath": ["100" if beneath else "000"] * n,
        "displaced_at_first_ready": [
            {"axis": "tool", "offset_mm": -20.0, "beams": "......W..", "target": "000000000", "ranges": _grid(EDGE)},
            {
                "axis": "lateral",
                "offset_mm": 50.0,
                "beams": "......WWW",
                "target": "000000000",
                "ranges": _grid(SPUR, SPUR, SPUR),
            },
            {
                "axis": "tool",
                "offset_mm": 5.0,
                "beams": "......TTT",
                "target": "000000111",
                "ranges": _grid(SPUR, SPUR, SPUR),
            },
        ],
    }


def test_the_scorer_replays_each_rule_and_judges_the_test_side(tools, tmp_path):
    replay, scorer = tools["gate_replay"], tools["score_jaw_reading"]
    clean = "artifacts/vision_robustness/demo/run_clean"
    crowded = "artifacts/isaac_render/job_crowded"
    buried = "artifacts/vision_robustness/demo/run_buried"
    captures = {
        clean: _records(replay, clean, "tree1_SPUR_component_14944"),
        crowded: _records(replay, crowded, "tree0_SPUR_component_8235"),
        buried: _records(replay, buried, "tree1_SPUR_component_14944"),
    }
    readings = [
        # A clean spur in the gap; with the tool short, one off-centre beam of other wood.
        _reading(replay, captures[clean], _grid(SPUR, SPUR, SPUR), _grid(EDGE)),
        # The target with other wood beside it; with the tool short, the other wood alone.
        _reading(replay, captures[crowded], _grid(THICK, THICK, THICK, EDGE), _grid(EDGE)),
        _reading(replay, captures[buried], _grid(SPUR, SPUR, SPUR), _grid(), beneath=True),
    ]
    cycles, beams, out = (
        tmp_path / "cycles.jsonl.gz",
        tmp_path / "beams" / "jaw_beams.jsonl.gz",
        tmp_path / "verdicts.json",
    )
    beams.parent.mkdir()
    with gzip.open(cycles, "wt", encoding="utf-8") as stream:
        for records in captures.values():
            for record in records:
                stream.write(json.dumps(record) + "\n")
    with gzip.open(beams, "wt", encoding="utf-8") as stream:
        for reading in readings:
            stream.write(json.dumps(reading) + "\n")
    argv = ["--cycles", str(cycles), "--beams", str(beams), "--side", "test", "--output", str(out)]
    assert scorer.main(argv) == 0
    result = json.loads(out.read_text())
    assert result["captures"] == 2 and result["skipped"] == {"tool_beneath_the_ground": 1}
    one, rule = result["summary"]["one_beam"], result["summary"]["centred_spur"]
    # The one-beam rule keeps both recorded cuts and accepts both short jaws: the refuted behaviour.
    assert one["all"]["recorded_kept"] == 2 and one["all"]["short_refused"] == 0
    # The centred-spur rule keeps the clean cut, refuses the crowded one, and refuses both short jaws.
    assert rule["other_targets"]["recorded_kept"] == 1 and rule["target_8235"]["recorded_kept"] == 0
    assert rule["all"]["short_refused"] == 2 and rule["target_8235"]["recorded_refused"] == [crowded]
    verdicts = {name: item["verdict"] for name, item in result["verdicts"].items()}
    assert verdicts == {"K1": "supported", "K2": "supported", "K3": "supported", "K4": "supported"}
    k4 = result["verdicts"]["K4"]
    # Of two poses without the target per capture, the one-beam rule accepts both and the new rule the spur-like one.
    assert (k4["one_beam_accepted"], k4["accepted"], k4["poses_without_the_target"]) == (4, 2, 4)
    design = tmp_path / "design.json"
    assert (
        scorer.main(["--cycles", str(cycles), "--beams", str(beams), "--side", "design", "--output", str(design)]) == 0
    )
    assert json.loads(design.read_text())["verdicts"] is None
    with pytest.raises(SystemExit):
        scorer.main(argv)  # never overwrites
