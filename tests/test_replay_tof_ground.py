"""tools/replay_tof_ground.py: grids with and without a ground sheet, and how a changed first hold is named."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture(scope="module")
def tool():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("replay_tof_ground", TOOLS / "replay_tof_ground.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quad(a, b, c, d):
    return np.array([a, b, c, d], float), np.array([[0, 1, 2], [0, 2, 3]])


def _scenes(tool, ground_z):
    """A wall 0.5 m ahead of the sensor on its axis, and a sheet across the axis at ``ground_z``."""
    wall = _quad([-0.04, -0.04, 0.5], [0.04, -0.04, 0.5], [0.04, 0.04, 0.5], [-0.04, 0.04, 0.5])
    far = _quad([9, 9, 9], [9.1, 9, 9], [9.1, 9.1, 9], [9, 9.1, 9])
    sheet = _quad([-3, -3, ground_z], [3, -3, ground_z], [3, 3, ground_z], [-3, 3, ground_z])
    index = tool.kmp.NumpyMeshIndex
    trees = tool.study.Scene([("tree0_BRANCH", *wall), ("TARGET", *far)], index_cls=index)
    full = tool.study.Scene(
        [("tree0_BRANCH", *wall), ("TARGET", *far), (tool.orchard_ground.GROUND, *sheet)], index_cls=index
    )
    return trees, full


def _frame(index=0):
    pose = {"position_w_m": [[0.0, 0.0, 0.0]], "quaternion_w_ros_xyzw": [[0.0, 0.0, 0.0, 1.0]]}
    return {"index": index, "tof_state": {"sensors": {"tof0": pose, "tof1": pose}}}


def test_the_ground_fills_zones_the_trees_left_empty(tool):
    trees, full = _scenes(tool, ground_z=1.0)
    cast, _ = tool.cast_grid(_frame(), "tof0", trees)
    both, labels = tool.cast_grid(_frame(), "tof0", full)
    # The small wall is hit by the four centre zones only; every other zone sees the sheet behind it.
    assert int(np.isfinite(cast).sum()) == 4 and np.isfinite(both).all()
    assert labels.count(tool.orchard_ground.GROUND) == 60 and labels.count("tree0_BRANCH") == 4
    assert np.nanmin(cast) == pytest.approx(np.nanmin(both))
    assert tool.as_record_grid(cast).count(None) == 60
    check = tool.compare_grids(cast, cast)
    assert check == {"zones": 64, "validity_same": 64, "both_valid": 4, "within_1mm": 4}


def test_a_return_beyond_the_range_is_no_return(tool):
    trees, full = _scenes(tool, ground_z=5.0)
    both, labels = tool.cast_grid(_frame(), "tof0", full)
    assert int(np.isfinite(both).sum()) == 4 and tool.orchard_ground.GROUND not in labels


def test_changed_holds_are_named(tool):
    def row(before, after):
        return {
            "ground_changes_the_first_hold": before != after,
            "first_hold_trees_only_cast": before,
            "first_hold_with_ground": after,
        }

    assert tool.kind_of_change(row([60, "hazard_contact"], [60, "hazard_contact"])) == "unchanged"
    assert (
        tool.kind_of_change(row([60, "hazard_contact"], [3, "tof_minimum_clearance"]))
        == "ground_within_clearance_stops_earlier"
    )
    assert (
        tool.kind_of_change(row([None, None], [12, "tof_minimum_clearance"])) == "ground_within_clearance_stops_earlier"
    )
    assert (
        tool.kind_of_change(row([40, "both_tof_missing_4_frames"], [70, "vision_invalid"]))
        == "blind_stop_removed_or_delayed"
    )
    assert tool.kind_of_change(row([40, "both_tof_missing_4_frames"], [None, None])) == "blind_stop_removed_or_delayed"
    assert tool.kind_of_change(row([40, "vision_invalid"], [None, None])) == "hold_removed"
    assert tool.kind_of_change(row([40, "vision_invalid"], [41, "hazard_contact"])) == "other"
