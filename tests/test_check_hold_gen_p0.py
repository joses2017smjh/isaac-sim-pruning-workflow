"""Protect the closure-hold generalization P0 gate's register and strategy checks."""

from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path

import pytest

pytest.importorskip("cv2")

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def p0():
    import sys

    sys.path.insert(0, str(ROOT / "tools"))
    spec = importlib.util.spec_from_file_location("check_hold_gen_p0", ROOT / "tools/check_hold_gen_p0.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _variant(a, b, c):
    return {"conditions": {"a_clear_path": a, "b_approach_keeps_patch": b, "c_closure_loses_patch": c}}


def test_a_jaw_free_plan_is_a_clear_variant_that_keeps_the_patch_through_closure(p0):
    assert p0.jaw_free({"variants": [_variant(True, True, False)]})
    assert not p0.jaw_free({"variants": [_variant(True, True, True), _variant(False, True, False)]})
    assert not p0.jaw_free({"variants": [_variant(True, False, False)]})
    assert not p0.jaw_free({})


def _evidence(p0):
    search = json.loads((p0.EVIDENCE / p0.SEARCH).read_text())
    pools = {
        t: {e["component_first_vertex"]: e for e in json.loads((p0.EVIDENCE / n).read_text())["targets"]}
        for t, n in p0.POOLS.items()
    }
    return search, pools


def test_the_committed_registers_pass_and_a_changed_orientation_or_set_is_caught(p0):
    search, pools = _evidence(p0)
    assert p0.check_registers(search, pools)[0] == []
    other = copy.deepcopy(search)
    for candidate in other["candidates"]:
        if candidate["target"] == "tree0_3721":
            candidate["quat_wxyz"] = [1.0, 0.0, 0.0, 0.0]
    assert [p["kind"] for p in p0.check_registers(other, pools)[0]] == ["planned_orientation"]
    other = copy.deepcopy(search)
    other["units"]["tree0_3721"]["variants"].append(_variant(True, True, False))  # 3721 gains a jaw-free plan
    kinds = [p["kind"] for p in p0.check_registers(other, pools)[0]]
    assert "selected_set" in kinds


def test_the_strategies_may_differ_from_the_jaw_in_view_arm_only_in_the_standoff(p0, monkeypatch):
    assert p0.check_strategies() == []
    changed = dict(p0.launcher.STRATEGIES["planned_pose_jaw_hold_gen"], max_step_m=0.006)
    monkeypatch.setitem(p0.launcher.STRATEGIES, "planned_pose_jaw_hold_gen", changed)
    assert [p["kind"] for p in p0.check_strategies()] == ["strategy"]


def test_an_existing_output_is_refused(p0, tmp_path):
    output = tmp_path / "p0.json"
    output.write_text("keep\n")
    with pytest.raises(SystemExit):
        p0.main(["--output", str(output)])
    assert output.read_text() == "keep\n"
