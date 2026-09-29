"""Protect the round's scoring: a refutation clause decides the verdict, whatever the other clauses say."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def scorer(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("score_perception_round", tools / "score_perception_round.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(vertex, outcome, **extra):
    return {"component_first_vertex": vertex, "outcome": outcome, **extra}


def _tree0(scorer, overrides=None):
    classes = {
        530: scorer.HAZARD,
        590: scorer.TOF,
        7524: scorer.HAZARD,
        8353: scorer.HAZARD,
        10001: scorer.LAYOUT,
        12142: scorer.VISION,
        18669: scorer.TOF,
        22988: scorer.HAZARD,
    }
    classes.update(overrides or {})
    return [_row(v, o, stop_frame=31 if v == 12142 else 60) for v, o in classes.items()]


def _tree1(scorer, overrides=None):
    classes = {14884: "failed_other", 14944: "pass", 15004: "pass", 19145: scorer.LAYOUT, 19264: scorer.NOT_VISIBLE}
    classes.update(overrides or {})
    return [_row(v, o) for v, o in classes.items()]


REFERENCE = {
    530: "stopped_hazard_contact",
    590: "stopped_vision_invalid",
    7524: "stopped_hazard_contact",
    8353: "stopped_vision_invalid",
    10001: "rejected_layout_startup_contact",
    12142: "stopped_vision_invalid",
    18669: "stopped_tof_minimum_clearance",
    22988: "stopped_hazard_contact",
    14884: "failed_other",
    14944: "pass",
    15004: "pass",
    19145: "rejected_layout_startup_contact",
    19264: "stopped_initial_target_not_visible",
}


def test_a_failed_expectation_without_its_refutation_is_only_partly_supported(scorer):
    clauses = [scorer.clause("a", 1, True), scorer.clause("b", 2, False)]
    assert scorer.verdict(clauses) == "partly supported"
    clauses.append(scorer.clause("c", 3, False, refutes=True))
    assert scorer.verdict(clauses) == "refuted"
    assert scorer.verdict([scorer.clause("a", 1, True)]) == "supported"


def test_frame_zero_is_a_frame_not_a_missing_value(scorer):
    assert scorer.within(0, 0, 10)
    assert not scorer.within(None, 0, 10)
    assert not scorer.within(11, 0, 10)


def test_p4_is_refuted_when_8353_still_stops_on_vision(scorer):
    tree0 = _tree0(scorer, {8353: scorer.VISION})
    result = scorer.score_p4(tree0, _tree1(scorer), REFERENCE)
    assert result["verdict"] == "refuted"
    assert scorer.score_p4(_tree0(scorer), _tree1(scorer), REFERENCE)["verdict"] == "supported"


def test_p4_records_a_contact_target_that_turned_into_a_vision_stop(scorer):
    tree0 = _tree0(scorer, {22988: scorer.VISION})
    result = scorer.score_p4(tree0, _tree1(scorer), REFERENCE)
    kept = next(c for c in result["clauses"] if c["clause"].startswith("contact, ToF"))
    assert not kept["holds"] and 22988 in kept["observed"]
    assert result["verdict"] == "partly supported"


def test_p6_is_refuted_by_any_class_change_under_either_light(scorer):
    source = _tree1(scorer)
    steady = {"morning": _tree1(scorer), "evening": _tree1(scorer)}
    assert scorer.score_p6(steady, REFERENCE, source)["verdict"] == "supported"
    changed = {"morning": _tree1(scorer), "evening": _tree1(scorer, {15004: scorer.VISION})}
    result = scorer.score_p6(changed, REFERENCE, source)
    assert result["verdict"] == "refuted"
    assert result["clauses"][0]["observed"]["evening"][15004]["source_repeat_r1_outcome"] == "pass"


def test_p8_uses_both_thresholds_and_the_presentability_calls(scorer):
    calls = [{"component_first_vertex": v, "prediction": "predicted_layout_overlap"} for v in range(10)]
    calls.append({"component_first_vertex": 99, "prediction": "predicted_clear"})
    rows = [_row(v, scorer.LAYOUT) for v in range(4)] + [_row(v, scorer.HAZARD) for v in range(4, 10)]
    rows.append(_row(99, "pass"))
    # 4 of 11 refused: above 30%, below 40%, so neither held nor refuted.
    assert scorer.score_p8(rows, calls)["verdict"] == "partly supported"
    rows[5] = _row(5, "pass")
    assert scorer.score_p8(rows, calls)["verdict"] == "refuted"


def test_a_perception_stop_before_the_predicted_event_leaves_the_geometry_untested(scorer):
    register = {
        "predicted_passes_at_most": 0,
        "targets": [
            {"component_first_vertex": 1, "class": "tof_minimum_clearance", "recorded_frame_window": [36, 40]},
            {"component_first_vertex": 2, "class": "hazard_contact", "recorded_frame_window": [20, 24]},
            {"component_first_vertex": 3, "class": "geometry_clear"},
        ],
    }
    rows = [
        _row(1, scorer.VISION, decision_frame=0),
        _row(2, scorer.LAYOUT),
        _row(3, scorer.VISION, decision_frame=5),
    ]
    table = {t["target"]: t["status"] for t in scorer.score_swept(rows, register)["targets"]}
    assert table == {
        1: "untested: perception stopped before the predicted event",
        2: "disagrees",
        3: "agrees",
    }
