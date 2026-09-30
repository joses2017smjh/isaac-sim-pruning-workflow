"""Protect the planned-approach scoring: GPU labels, and each prediction's own refutation rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def scorer(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("score_planned_approach", tools / "score_planned_approach.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _runs(scorer, planned="stopped_vision_invalid", aligned=False, control="pass", contact=0.0):
    runs = {}
    for batch in scorer.BATCHES:
        for vertex in scorer.PLANNED:
            runs[(batch, vertex)] = {
                "outcome": planned,
                "reached_align": aligned,
                "max_contact_n": contact,
                "stop": None,
                "gpu_model": "A40",
            }
        for vertex in scorer.CONTROLS:
            runs[(batch, vertex)] = {"outcome": control}
    return runs


def test_gpu_model_comes_from_the_node_name(scorer):
    assert scorer.gpu_model("cn-r-5") == scorer.gpu_model("cn-s-2") == "A40"
    assert scorer.gpu_model("cn-gpu7") == "RTX 8000"
    assert scorer.gpu_model("dgxh-2") == "unknown"
    assert scorer.gpu_model(None) is None


def test_a_perception_stop_before_align_does_not_refute_q1_but_a_contact_stop_does(scorer):
    assert scorer.score_q1(_runs(scorer))["verdict"] == "partly supported"
    assert scorer.score_q1(_runs(scorer, planned="stopped_hazard_contact"))["verdict"] == "refuted"
    assert scorer.score_q1(_runs(scorer, planned="stopped_gate_lost_during_closure", aligned=True))["verdict"] == (
        "supported"
    )
    over = _runs(scorer, planned="stopped_gate_lost_during_closure", aligned=True, contact=6.0)
    assert scorer.score_q1(over)["verdict"] == "partly supported"


def test_q2_and_q3_follow_their_refutation_rules(scorer):
    assert scorer.score_q2(_runs(scorer))["verdict"] == "supported"
    assert scorer.score_q2(_runs(scorer, control="failed_other"))["verdict"] == "refuted"
    assert scorer.score_q3(_runs(scorer))["verdict"] == "refuted"
    assert scorer.score_q3(_runs(scorer, planned="pass", aligned=True))["verdict"] == "supported"
