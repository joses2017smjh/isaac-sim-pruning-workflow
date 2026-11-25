"""tools/gate_replay.py: the reference gate over rebuilt records, the jaw-presence input and the node's jaw rule."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

from isaaclab_pruning.sim.vision_demo_controller import demo_cut_config

TOOLS = Path(__file__).resolve().parents[1] / "tools"
TARGET = "tree0_SPUR_component_8175"


@pytest.fixture(scope="module")
def replay():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("gate_replay", TOOLS / "gate_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(index, *, phase="approach", prior="approach", progress=0.0, beams=None, mouth_x=0.002, valid=True):
    """One cycle with the target 2 mm from the mouth (ready to close) unless ``mouth_x`` says otherwise."""
    out = {
        "index": index,
        "stamp_ns": 1_000_000_000 + index * 100_000_000,
        "tool_pose_wxyz": [mouth_x, 0.0, -0.07, 1.0, 0.0, 0.0, 0.0],
        "mouth_offset_tool_m": [0.0, 0.0, 0.07],
        "closing_axis_tool": [1.0, 0.0, 0.0],
        "contact_force_n": 0.0,
        "tof_left": [0.3] * 64,
        "tof_right": [0.3] * 64,
        "target": {
            "id": TARGET,
            "semantic": "branch",
            "valid": valid,
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
    if beams is not None:
        out["jaw_beams"] = beams
    return out


def closure(beams, first_progress=0.0):
    """Two approach cycles, then four closing cycles; the jaw moves from the second closing cycle on."""
    rows = [record(0, beams=beams), record(1, beams=beams)]
    rows.append(record(2, phase="closing", progress=first_progress, beams=beams))
    for k, progress in enumerate((0.25, 0.5, 0.75), start=3):
        rows.append(record(k, phase="closing", prior="closing", progress=progress, beams=beams))
    return rows


def test_the_harness_config_is_the_demo_config(replay):
    assert replay.demo_config(TARGET) == demo_cut_config(TARGET)
    assert replay.stamp_seconds(1_500_000_000) == 1.5


def test_with_the_channel_off_a_reading_changes_nothing(replay):
    plain = replay.replay(closure(None), node_rules=True)
    ignored = replay.replay(closure("........."), node_rules=True)
    assert plain == ignored
    assert all(row["passed"] and row["jaw_presence_ok"] is None for row in plain)
    assert [row["gated_jaw"] for row in plain] == [0.0, 0.0, 0.0, 0.25, 0.5, 0.75]


def test_blocked_beams_let_the_recorded_closure_through(replay):
    on = replay.replay(closure("...TT...."), jaw_min_beams=1, jaw_key="jaw_beams", node_rules=True)
    off = replay.replay(closure("...TT...."), node_rules=True)
    assert [(row["passed"], row["reason"], row["gated_jaw"]) for row in on] == [
        (row["passed"], row["reason"], row["gated_jaw"]) for row in off
    ]
    assert all(row["jaw_presence_ok"] for row in on)


def test_an_empty_jaw_stops_the_closure_before_the_jaw_moves(replay):
    rows = closure(".........")
    verdicts = replay.replay(rows, jaw_min_beams=1, jaw_key="jaw_beams", node_rules=True)
    # The controller announces closing in cycle 2 without moving the jaw; cycle 3 is the first with closing as
    # the prior phase, and the reference stops there.
    assert replay.first_hold(verdicts, rows) == (3, "jaw_empty_during_closure")
    assert max(row["gated_jaw"] for row in verdicts) == 0.0
    assert verdicts[2]["passed"] and not verdicts[2]["closure_authorized"]


def test_a_jaw_that_moves_in_its_first_closing_cycle_is_stopped_by_the_node_rule(replay):
    rows = closure(".........", first_progress=0.1)
    verdicts = replay.replay(rows, jaw_min_beams=1, jaw_key="jaw_beams", node_rules=True)
    assert replay.first_hold(verdicts, rows) == (2, replay.CLOSURE_WITHOUT_READY_GATE)
    assert max(row["gated_jaw"] for row in verdicts) == 0.0
    # The node's latch is written into the reference, so later cycles report it too.
    assert all(row["latched_reason"] == replay.CLOSURE_WITHOUT_READY_GATE for row in verdicts[2:])
    # Without the node's rule the reference alone stops one cycle later, on its own reason.
    alone = replay.replay(rows, jaw_min_beams=1, jaw_key="jaw_beams")
    assert replay.first_hold(alone, rows) == (3, "jaw_empty_during_closure")


def test_a_missing_reading_holds_when_the_channel_is_on(replay):
    rows = closure(None)
    verdicts = replay.replay(rows, jaw_min_beams=1, jaw_key="jaw_beams")
    assert replay.first_hold(verdicts, rows) == (0, "jaw_presence_missing")


def test_helpers(replay):
    assert replay.blocked_count(None) is None
    assert replay.blocked_count(".T.WG....") == 3
    assert replay.jaw_advances(closure(None)) == [3, 4, 5]
    assert replay.first_hold(replay.replay(closure(None)), closure(None)) == (None, None)


def test_a_beam_reading_is_attached_only_to_its_own_cycles(replay):
    rows = closure(None)
    reading = {
        "index": [row["index"] for row in rows],
        "stamp_ns": [row["stamp_ns"] for row in rows],
        "pose_digest": [replay.pose_digest(row["tool_pose_wxyz"]) for row in rows],
        "beams": ["....T...."] * len(rows),
        "beneath": ["000"] * len(rows),
    }
    joined = replay.attach_beams(rows, reading)
    assert [row["jaw_beams"] for row in joined] == ["....T...."] * 6 and "jaw_beams" not in rows[0]
    assert {row["jaw_beams"] for row in replay.attach_beams(rows, reading, ".........")} == {"........."}
    assert replay.tool_above_the_ground(reading)
    assert not replay.tool_above_the_ground({**reading, "beneath": ["000", "100"] + ["000"] * 4})
    moved = {**reading, "pose_digest": [replay.pose_digest([9.0] * 7)] + reading["pose_digest"][1:]}
    with pytest.raises(ValueError, match="different cycle"):
        replay.attach_beams(rows, moved)
    with pytest.raises(ValueError, match="different lengths"):
        replay.attach_beams(rows[:3], reading)
