"""Protect the closure-hold sweep P0 gate's register and strategy checks."""

from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def p0():
    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("check_hold_sweep_p0", ROOT / "tools/check_hold_sweep_p0.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(p0):
    search = json.loads((p0.EVIDENCE / p0.SEARCH).read_text())
    pools = {
        t: {e["component_first_vertex"]: e for e in json.loads((p0.EVIDENCE / n).read_text())["targets"]}
        for t, n in p0.POOLS.items()
    }
    return search, pools, json.loads((p0.EVIDENCE / p0.REGISTER).read_text())


def test_the_committed_register_passes_and_departures_are_caught(p0):
    search, pools, register = _inputs(p0)
    problems, detail = p0.check_register(search, pools, register)
    assert problems == [] and detail["registered"] == 28
    changed = copy.deepcopy(register)
    changed["targets"][0]["planned_final_tool_quat_wxyz"] = [1.0, 0.0, 0.0, 0.0]
    assert [p["kind"] for p in p0.check_register(search, pools, changed)[0]] == ["planned_orientation"]
    dropped = copy.deepcopy(register)
    dropped["targets"].pop()
    assert "registered_set" in [p["kind"] for p in p0.check_register(search, pools, dropped)[0]]


def test_the_strategy_must_be_the_jaw_hold_arm(p0, monkeypatch):
    assert p0.check_strategy() == []
    monkeypatch.setitem(p0.launcher.STRATEGIES, p0.STRATEGY, {**p0.launcher.STRATEGIES[p0.STRATEGY], "standoff_m": 0.1})
    assert [p["kind"] for p in p0.check_strategy()] == ["strategy"]
