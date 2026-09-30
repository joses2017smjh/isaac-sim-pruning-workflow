"""Protect the rendered-lighting scoring: per-tree counting and each prediction's own refutation rule."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


@pytest.fixture
def scorer(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("score_lighting_training", tools / "score_lighting_training.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TREES = ("lpy_envy_00000", "lpy_envy_00003", "lpy_envy_00008", "lpy_envy_00012")


def _frames(scorer, b_values, c_value=1.0):
    rows = []
    for tree, b in zip(TREES, b_values):
        for view in range(6):
            rows.append(
                {
                    "model": scorer.MODEL["B"],
                    "tree_id": tree,
                    "family": "envy",
                    "condition": "evening_x2.6",
                    "mask_mae_m": b + (0.01 if view % 2 else -0.01),
                }
            )
            rows.append(
                {
                    "model": scorer.MODEL["C"],
                    "tree_id": tree,
                    "family": "envy",
                    "condition": "evening_x2.6",
                    "mask_mae_m": c_value,
                }
            )
    return {"frames": rows}


def test_per_tree_values_are_frame_means(scorer):
    rows = _frames(scorer, (0.5, 0.5, 0.5, 0.5))["frames"]
    assert scorer.per_tree(rows, scorer.MODEL["B"], family="envy", condition="evening_x2.6") == pytest.approx(
        {tree: 0.5 for tree in TREES}
    )


def test_r1_counts_trees_against_both_thresholds(scorer):
    assert scorer.score_r1(_frames(scorer, (0.5, 0.5, 0.5, 0.9)))["verdict"] == "supported"
    assert scorer.score_r1(_frames(scorer, (0.7, 0.7, 0.5, 0.5)))["verdict"] == "partly supported"
    assert scorer.score_r1(_frames(scorer, (0.9, 0.9, 0.9, 0.1)))["verdict"] == "refuted"


def test_r5_is_refuted_only_when_most_trees_rise_by_five_centimetres(scorer):
    def matrix(rise):
        rows = []
        for i in range(8):
            tree = f"t{i}"
            rows.append({"model": scorer.MODEL["F"], "tree_id": tree, "lighting": "source", "mask_mae_m": 0.1})
            for arm in ("A", "B"):
                rows.append(
                    {"model": scorer.MODEL[arm], "tree_id": tree, "lighting": "source", "mask_mae_m": 0.1 + rise[i]}
                )
        comparison = [
            {"model": scorer.MODEL[a], "lighting": light, "envy_mask_mae_m": 0.1}
            for a in ("A", "J")
            for light in ("morning", "noon")
        ]
        return {"lighting_effect": rows, "family_comparison": comparison}

    assert scorer.score_r5(matrix([0.0] * 8))["verdict"] == "supported"
    assert scorer.score_r5(matrix([0.06] * 4 + [0.0] * 4))["verdict"] == "partly supported"
    assert scorer.score_r5(matrix([0.06] * 5 + [0.0] * 3))["verdict"] == "refuted"


def test_open_details_are_recorded(scorer):
    assert any("R6 Stage A uses the source light" in item for item in scorer.INTERPRETATIONS)
