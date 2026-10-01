"""Protect the render-gap scoring: the published ratios are reproduced and G0 re-applies the registered gate."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PUBLISHED = ROOT / "docs/evidence/finetune_rendered_family_matrix_2026-09-30.json"


@pytest.fixture
def scorer(monkeypatch):
    tools = ROOT / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("score_render_gap", tools / "score_render_gap.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def published():
    return json.loads(PUBLISHED.read_text())


def _trees(published):
    return sorted({row["tree_id"] for row in published["lighting_effect"]})


def _families(published):
    return {row["tree_id"]: row["family"] for row in published["lighting_effect"]}


def _span(values):
    return round(min(values), 3), round(max(values), 3)


def test_the_published_matrix_reproduces_its_published_ratios(scorer, published):
    trees = _trees(published)
    g1 = scorer.score_g1(published, trees)
    a_over_c, b_over_c = (c["observed"]["ratios"] for c in g1["clauses"][:2])
    assert _span(a_over_c.values()) == (0.573, 0.687)
    assert _span(b_over_c.values()) == (0.491, 0.602)
    assert g1["verdict"] == "refuted"  # the published matrix sits in G1's refutation region
    g2 = scorer.score_g2(published, trees, _families(published))
    assert _span(g2["clauses"][0]["observed"]["B_over_C"].values()) == (0.046, 0.064)
    assert _span(g2["clauses"][1]["observed"]["A_over_J"].values()) == (0.051, 0.115)
    assert g2["verdict"] == "supported"
    g3 = scorer.score_g3(published, published, trees)
    assert set(g3["clauses"][0]["observed"]["ratio"].values()) == {1.0}
    assert g3["verdict"] == "refuted"


def _rows(model, lighting, values):
    return [
        {
            "model": model,
            "family": "envy" if i < 4 else "ufo",
            "tree_id": f"t{i}",
            "lighting": lighting,
            "mask_mae_m": v,
        }
        for i, v in enumerate(values)
    ]


def test_g1_counts_use_the_registered_bounds(scorer):
    trees = [f"t{i}" for i in range(8)]
    c = _rows("da2_control", "source", [1.0] * 8)

    def verdict(a_values, b_values):
        rows = c + _rows("da2_rendered_jitter", "source", a_values) + _rows("da2_rendered_control", "source", b_values)
        return scorer.score_g1({"lighting_effect": rows}, trees)["verdict"]

    assert verdict([0.85] * 6 + [0.5] * 2, [0.9] * 8) == "supported"
    assert verdict([0.84] * 8, [0.9] * 8) == "partly supported"
    assert verdict([0.70] * 6 + [0.9] * 2, [0.9] * 8) == "refuted"
    assert verdict([0.9] * 8, [0.71] * 8) == "partly supported"
    # A tree left out of Matched counts toward neither side.
    assert scorer.score_g1(
        {
            "lighting_effect": c
            + _rows("da2_rendered_jitter", "source", [0.9] * 8)
            + _rows("da2_rendered_control", "source", [0.9] * 8)
        },
        trees[:5],
    )["verdict"] == ("partly supported")


def _pose_identity(trees, failing_view=None, recorded=None):
    def view(tree, index):
        iou = 0.99 if (tree, index) == failing_view else 1.0
        return {"gate": {"mask_iou": iou, "tree_depth_within_tolerance": 1.0, "tolerance_m": 0.001}}

    return {
        "trees": [{"tree_id": t, "ok": True, "views": [view(t, i) for i in range(6)]} for t in trees],
        "matched_trees": recorded if recorded is not None else list(trees),
        "excluded_trees": [],
        "not_checked": [],
    }


def test_g0_re_applies_the_gate_to_every_view(scorer):
    trees = [f"t{i}" for i in range(8)]
    g0, passing, _ = scorer.score_g0(_pose_identity(trees), {})
    assert g0["verdict"] == "supported" and passing == trees
    g0, passing, table = scorer.score_g0(_pose_identity(trees, failing_view=("t3", 4)), {})
    assert g0["verdict"] == "refuted" and "t3" not in passing
    assert table["t3"]["views_passing"] == 5
    g0, _, _ = scorer.score_g0(_pose_identity(trees, recorded=trees[:7]), {})
    assert g0["verdict"] == "partly supported"
    loose = _pose_identity(trees)
    loose["trees"][0]["views"][0]["gate"]["tolerance_m"] = 0.002
    assert scorer.score_g0(loose, {})[0]["verdict"] == "refuted"
