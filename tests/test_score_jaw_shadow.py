"""Protect the jaw-shadow scoring: only appearance losses count, and J5's thresholds are as registered."""

from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path

import pytest

pytest.importorskip("cv2")

_BATCH = itertools.count()


@pytest.fixture
def scorer(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("score_jaw_shadow", tools / "score_jaw_shadow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _record(arm, light, target, outcome, reason=None, corr=0.99, change=0.0, readback=(True, True)):
    frames = (57, 58, 59, 67) if light == "evening" else (73, 74, 75)
    stop = None if reason is None else {"decision_frame": 67, "phase": "vision_approach", "tracker_reason": reason}
    return {
        "batch": f"{arm}-{light}-r{next(_BATCH)}",
        "target": target,
        "daylight": light,
        "arm": arm,
        "outcome": outcome,
        "stop": stop,
        "patch_correlation": {f: corr for f in frames},
        "patch_darkening": {"change": change},
        "do_not_cast_shadows_readback": list(readback) if arm == "B" else None,
    }


def _records(scorer, b_outcomes=("pass",) * 4, b_reason=None, a_corr=0.1):
    records = []
    for target, outcome in zip((14944, 15004, 14944, 15004), b_outcomes, strict=True):
        records.append(_record("B", "evening", target, outcome, reason=None if outcome == "pass" else b_reason))
    records += [_record("B", "morning", 15004, "pass") for _ in range(2)]
    records += [
        _record("A", "evening", t, "stopped_vision_invalid", "appearance_changed_or_occluded", a_corr, -0.5)
        for t in (14944, 15004, 14944, 15004)
    ]
    records += [_record("A", "morning", 15004, "stopped_vision_invalid", "appearance_changed_or_occluded", 0.3)] * 2
    return records


def test_all_supported_on_the_expected_pattern(scorer):
    assert {p["id"]: p["verdict"] for p in scorer.score(_records(scorer))} == {
        f"J{i}": "supported" for i in range(1, 6)
    }


def test_an_arm_b_appearance_stop_refutes_j2_and_j5_counts_passes(scorer):
    outcomes = ("stopped_vision_invalid", "stopped_vision_invalid", "stopped_vision_invalid", "pass")
    verdicts = {
        p["id"]: p["verdict"] for p in scorer.score(_records(scorer, outcomes, "appearance_changed_or_occluded"))
    }
    assert verdicts["J2"] == "refuted" and verdicts["J5"] == "refuted"
    # A non-appearance stop is not J2's refutation, but it still costs J5 its pass count.
    verdicts = {p["id"]: p["verdict"] for p in scorer.score(_records(scorer, outcomes, "depth_measurement_rejected"))}
    assert verdicts["J2"] != "refuted" and verdicts["J5"] == "refuted"


def test_a_missing_readback_or_a_dark_arm_b_patch_refutes_j1(scorer):
    records = _records(scorer)
    records[0]["do_not_cast_shadows_readback"] = [True, None]
    assert scorer.score(records)[0]["verdict"] == "refuted"
    records = _records(scorer)
    records[1]["patch_darkening"]["change"] = -0.35
    assert scorer.score(records)[0]["verdict"] == "refuted"
