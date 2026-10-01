"""Protect the H1-H4 scoring of the depth-aware appearance replay: registered bounds, refutations and kept stops."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def scorer(monkeypatch):
    tools = ROOT / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("score_depth_heldout", tools / "score_depth_heldout.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _event(frame, accept=True, confidence=0.2, continues=None, reason=None, agreement=True):
    continues = (confidence >= 0.15) if continues is None else continues
    return {
        "frame": frame,
        "accept": accept,
        "reasons": [] if accept else ["occluder_nearer_than_predicted"],
        "failed_conditions": [] if accept else ["3_near_fraction"],
        "strict_confidence": confidence if accept else None,
        "strict_decision": {
            "continues": continues,
            "state": "tracking" if continues else "tracking_lost",
            "reason": None if continues else (reason or "low_confidence"),
        },
        "agreement_decision": {"continues": agreement},
        "jaw": {"touches": False},
    }


def _run(name, stop=None, reason=None, depth_reason=None, events=(), strict_div=(), agreement_div=(), state=None):
    return {
        "run": name,
        "recorded": {
            "stop_frame": stop,
            "stop_reason": None if stop is None else "vision_invalid",
            "stop_measurement": None if stop is None else {"reason": reason, "depth_reason": depth_reason},
        },
        "events_joined": list(events),
        "grey_failures": [e["frame"] for e in events],
        "base_reproduces_recording": True,
        "divergence": {
            "strict": {"mismatch_frames": list(strict_div)},
            "agreement": {"mismatch_frames": list(agreement_div)},
        },
        "measurement_at_recorded_stop": {"strict": {"state": state or "tracking_lost"}},
    }


LOSS = "appearance_changed_or_occluded"


def _heldout(a_events=None):
    """6 arm-A appearance stops (evening 14944 x2 low confidence, evening and morning 15004 continue), 6 clean B."""
    a_events = a_events or {}
    runs = []
    for r in (1, 2):
        for name, frame, confidence in (
            (f"jaw-shadow-eve-a-r{r}-20260930/run_00_evening_tree1_v14944_baseline", 67, 0.04),
            (f"jaw-shadow-eve-a-r{r}-20260930/run_01_evening_tree1_v15004_baseline", 67, 0.25),
            (f"jaw-shadow-mor-a-r{r}-20260930/run_00_morning_tree1_v15004_baseline", 75, 0.2),
        ):
            event = a_events.get(name, _event(frame, confidence=confidence))
            runs.append(_run(name, stop=frame, reason=LOSS, events=[event], strict_div=[frame], agreement_div=[frame]))
        for name in (
            f"jaw-shadow-eve-b-r{r}-20260930/run_00_evening_tree1_v14944_jaw_no_shadow",
            f"jaw-shadow-eve-b-r{r}-20260930/run_01_evening_tree1_v15004_jaw_no_shadow",
            f"jaw-shadow-mor-b-r{r}-20260930/run_00_morning_tree1_v15004_jaw_no_shadow",
        ):
            runs.append(_run(name))
    return runs


def _verdicts(scorer, heldout):
    return {
        p["id"]: p["verdict"] for p in (scorer.score_h1(heldout), scorer.score_h2(heldout), scorer.score_h3(heldout))
    }


def test_the_expected_held_out_pattern_supports_h1_h2_h3(scorer):
    assert _verdicts(scorer, _heldout()) == {"H1": "supported", "H2": "supported", "H3": "supported"}


def test_h1_is_refuted_by_one_rejected_appearance_stop(scorer):
    name = "jaw-shadow-mor-a-r2-20260930/run_00_morning_tree1_v15004_baseline"
    heldout = _heldout({name: _event(75, accept=False)})
    assert scorer.score_h1(heldout)["verdict"] == "refuted"


def test_h2_is_refuted_by_a_decision_against_its_own_confidence_or_an_agreement_stop(scorer):
    name = "jaw-shadow-eve-a-r1-20260930/run_01_evening_tree1_v15004_baseline"
    assert scorer.score_h2(_heldout({name: _event(67, confidence=0.1, continues=True)}))["verdict"] == "refuted"
    assert scorer.score_h2(_heldout({name: _event(67, confidence=0.2, agreement=False)}))["verdict"] == "refuted"
    other_gate = _event(67, confidence=0.2, continues=False, reason="depth_measurement_rejected")
    assert scorer.score_h2(_heldout({name: other_gate}))["verdict"] == "partly supported"


def test_h3_is_refuted_by_any_arm_b_divergence(scorer):
    heldout = _heldout()
    heldout[-1]["divergence"]["agreement"]["mismatch_frames"] = [120]
    assert scorer.score_h3(heldout)["verdict"] == "refuted"


def _regression(scorer, monkeypatch, extra=()):
    jaw = "planned-pose-gpu-r1-20260929/run_01_source_tree1_v19444_planned_pose"
    monkeypatch.setattr(scorer, "JAW_STOPS", (jaw,))
    monkeypatch.setattr(scorer, "REGRESSION_RUNS", 3 + len(extra))
    low_sun = "tree1-listed-evening-20260926/run_02_evening_tree1_v15004_baseline"
    runs = [
        _run(low_sun, stop=67, reason=LOSS, strict_div=[67], state="tracking"),
        _run(jaw, stop=68, reason=LOSS),
        _run(
            "strategy-fine-step-20260923/run_06_source_tree0_v12142_fine_step", stop=59, depth_reason="mixed_surfaces"
        ),
        *extra,
    ]
    return runs, [run["run"] for run in runs]


def test_h4_allows_only_the_low_sun_divergences_and_requires_every_real_stop_kept(scorer, monkeypatch):
    runs, inventory = _regression(scorer, monkeypatch)
    assert scorer.score_h4(runs, inventory)["verdict"] == "supported"
    other = _run("targets-source-20260923/run_00_source_tree0_v530", stop=40, reason=LOSS, strict_div=[40])
    runs, inventory = _regression(scorer, monkeypatch, extra=[other])
    assert scorer.score_h4(runs, inventory)["verdict"] == "refuted"
    runs, inventory = _regression(scorer, monkeypatch)
    runs[1]["measurement_at_recorded_stop"]["strict"]["state"] = "tracking"
    assert scorer.score_h4(runs, inventory)["verdict"] == "refuted"
    runs, inventory = _regression(scorer, monkeypatch)
    assert scorer.score_h4(runs, inventory[:-1] + ["someone-else/run_00"])["verdict"] == "partly supported"
