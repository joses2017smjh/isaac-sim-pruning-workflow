"""Protect the C1-C6 scoring of the depth-aware closed-loop test: registered windows, bounds and untested cases."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scorer(monkeypatch):
    tools = ROOT / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("score_depth_loop", tools / "score_depth_loop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(accept=True, confidence=0.2):
    return {"accept": accept, "strict_confidence": confidence if accept else None, "failed_conditions": []}


def _record(batch, target, outcome="pass", stop=None, events=None):
    events = events or {}
    kind = (
        "control"
        if "-ctl-" in batch
        else "source"
        if "-src-" in batch
        else "evening"
        if "-eve-" in batch
        else "morning"
    )
    return {
        "batch": batch,
        "kind": kind,
        "run_directory": f"run_{target}",
        "target": target,
        "daylight": kind,
        "outcome": outcome,
        "checks": "17/17" if outcome == "pass" else "9/17",
        "stop": stop,
        "grey_failures": sorted(events),
        "events": events,
    }


def _stop(frame, reason):
    return {"frame": frame, "stopped_reason": "vision_invalid", "tracker_reason": reason}


def _expected():
    """The registered expectation: 15004 continues and passes, evening 14944 stops at 67 on low confidence."""
    records = []
    for r in (1, 2):
        records.append(
            _record(
                f"depth-loop-eve-r{r}-20261001",
                14944,
                "stopped_vision_invalid",
                _stop(67, "low_confidence"),
                {67: _event(confidence=0.03)},
            )
        )
        records.append(_record(f"depth-loop-eve-r{r}-20261001", 15004, events={67: _event(confidence=0.25)}))
        records.append(_record(f"depth-loop-mor-r{r}-20261001", 15004, events={75: _event(confidence=0.2)}))
    records += [_record("depth-loop-src-20261001", 14944), _record("depth-loop-src-20261001", 15004)]
    records.append(
        _record(
            "depth-loop-ctl-19444-20261001",
            19444,
            "stopped_vision_invalid",
            _stop(68, "appearance_changed_or_occluded"),
            {68: _event(accept=False)},
        )
    )
    records.append(
        _record(
            "depth-loop-ctl-12142-20261001",
            12142,
            "stopped_vision_invalid",
            _stop(29, "appearance_changed_or_occluded"),
            {29: _event(accept=False)},
        )
    )
    return records


def _verdicts(scorer, records):
    comparisons = {scorer.name(r): [] for r in records}
    return {p["id"]: p["verdict"] for p in scorer.score(records, comparisons)}


def test_the_registered_expectation_supports_every_prediction(scorer):
    assert _verdicts(scorer, _expected()) == {f"C{i}": "supported" for i in range(1, 7)}


def test_c1_is_refuted_by_a_rejected_shadow_event_and_untested_without_one(scorer):
    records = _expected()
    records[2]["events"] = {74: _event(accept=False), 75: _event()}
    records[2]["grey_failures"] = [74, 75]
    assert scorer.score_c1(records)["verdict"] == "refuted"
    records = _expected()
    records[1]["events"], records[1]["grey_failures"] = {}, []
    assert scorer.score_c1(records)["verdict"] == "partly supported"
    records = _expected()
    records[1]["events"], records[1]["grey_failures"] = {80: _event(accept=False)}, [80]  # outside the window
    assert scorer.score_c1(records)["verdict"] == "partly supported"


def test_c2_bounds(scorer):
    records = _expected()
    records[1].update(outcome="stopped_vision_invalid", stop=_stop(67, "low_confidence"))
    assert scorer.score_c2(records)["verdict"] == "supported"  # 3 of 4 pass, the stop is not on appearance
    records[4].update(outcome="stopped_vision_invalid", stop=_stop(67, "low_confidence"))
    assert scorer.score_c2(records)["verdict"] == "refuted"  # two stops in the window
    records = _expected()
    records[2].update(outcome="stopped_vision_invalid", stop=_stop(75, "appearance_changed_or_occluded"))
    assert scorer.score_c2(records)["verdict"] == "partly supported"  # 3 pass but one appearance stop
    records = _expected()
    for index in (1, 2, 4):
        records[index].update(outcome="stopped_vision_invalid", stop=_stop(150, "hazard_contact"))
    assert scorer.score_c2(records)["verdict"] == "refuted"  # fewer than 2 pass


def test_c3_is_refuted_when_14944_continues_or_stops_with_the_test_rejecting(scorer):
    records = _expected()
    records[0].update(outcome="pass", stop=None)
    assert scorer.score_c3(records)["verdict"] == "refuted"
    records = _expected()
    records[0]["events"] = {67: _event(accept=False)}
    records[0]["stop"] = _stop(67, "appearance_changed_or_occluded")
    assert scorer.score_c3(records)["verdict"] == "refuted"
    records = _expected()
    records[0]["events"], records[0]["grey_failures"] = {}, []
    records[0]["stop"] = _stop(90, "hazard_contact")
    assert scorer.score_c3(records)["verdict"] == "partly supported"


def test_c4_refuted_by_a_passing_control_or_any_accepted_grey_failure(scorer):
    records = _expected()
    records[-1].update(outcome="pass", stop=None)
    assert scorer.score_c4(records)["verdict"] == "refuted"
    records = _expected()
    records[-2]["events"][20] = _event()
    records[-2]["grey_failures"] = [20, 68]
    assert scorer.score_c4(records)["verdict"] == "refuted"
    records = _expected()
    records[-1]["events"], records[-1]["grey_failures"] = {}, []
    assert scorer.score_c4(records)["verdict"] == "partly supported"


def test_c5_and_c6(scorer):
    records = _expected()
    records[6].update(outcome="stopped_vision_invalid")
    assert scorer.score_c5(records)["verdict"] == "refuted"
    records = _expected()
    comparisons = {scorer.name(r): [] for r in records}
    comparisons[scorer.name(records[3])] = [{"kind": "event", "frame": "67", "fields": ["accept"]}]
    assert scorer.score_c6(records, comparisons)["verdict"] == "refuted"
    assert scorer.score_c6(records[:-1], {scorer.name(r): [] for r in records[:-1]})["verdict"] == "partly supported"
