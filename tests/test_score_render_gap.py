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


TREES = [f"t{i}" for i in range(8)]
FAMILIES = {tree: ("envy" if i < 4 else "ufo") for i, tree in enumerate(TREES)}


def _view(index, iou=1.0, within=1.0, tolerance=0.001, values=True):
    gate = {"tolerance_m": tolerance, "ok": True}
    if values:
        gate.update(mask_iou=iou, tree_depth_within_tolerance=within)
    else:
        gate = {"ok": False, "reason": "no grid pass"}
    return {"view_id": f"v{index}", "gate": gate}


def _pose_identity(trees=TREES, failing=None, not_checked=(), recorded_ok=None, matched=None):
    """A consistent gate record: ``failing`` maps a tree to its failing view (index, kwargs for _view)."""
    failing = failing or {}
    records = []
    for tree in trees:
        if tree in not_checked:
            continue
        views = [_view(i) for i in range(6)]
        if tree in failing:
            index, kwargs = failing[tree]
            views[index] = _view(index, **kwargs)
        ok = tree not in failing if recorded_ok is None else recorded_ok.get(tree, tree not in failing)
        records.append(
            {
                "tree_id": tree,
                "ok": ok,
                "matrix_grid_pass": True,
                "views_checked": 6,
                "missing_views": [],
                "extra_views": [],
                "views": views,
            }
        )
    passing = [t for t in trees if t not in failing and t not in not_checked]
    return {
        "trees": records,
        "matched_trees": passing if matched is None else matched,
        "excluded_trees": sorted(failing),
        "not_checked": [{"tree_id": t, "reason": "render stalled"} for t in not_checked],
    }


def test_g0_re_applies_the_gate_to_every_view_and_keeps_each_reason(scorer):
    g0, passing, _ = scorer.score_g0(_pose_identity(), FAMILIES, TREES)
    assert g0["verdict"] == "supported" and passing == TREES
    g0, passing, table = scorer.score_g0(_pose_identity(failing={"t3": (4, {"iou": 0.99})}), FAMILIES, TREES)
    assert g0["verdict"] == "refuted" and "t3" not in passing and table["t3"]["views_passing"] == 5
    assert "mask IoU 0.99" in table["t3"]["failing_views"]["v4"]
    g0, _, table = scorer.score_g0(_pose_identity(failing={"t2": (0, {"values": False})}), FAMILIES, TREES)
    assert g0["verdict"] == "refuted" and table["t2"]["failing_views"]["v0"] == "no grid pass"
    g0, passing, table = scorer.score_g0(_pose_identity(failing={"t1": (2, {"tolerance": 0.002})}), FAMILIES, TREES)
    assert g0["verdict"] == "refuted" and "t1" not in passing
    g0, passing, _ = scorer.score_g0(_pose_identity(not_checked=("t7",)), FAMILIES, TREES)
    assert g0["verdict"] == "refuted" and "t7" not in passing
    assert g0["clauses"][0]["observed"]["failing"]["t7"] == "render stalled"


def test_a_gate_record_that_disagrees_with_the_re_applied_gate_is_refused(scorer):
    disagreeing = _pose_identity(failing={"t3": (4, {"iou": 0.99})}, recorded_ok={"t3": True})
    with pytest.raises(scorer.InputError, match="t3"):
        scorer.score_g0(disagreeing, FAMILIES, TREES)
    with pytest.raises(scorer.InputError, match="matched_trees"):
        scorer.score_g0(_pose_identity(matched=TREES[:7]), FAMILIES, TREES)
    record = _pose_identity()
    record["trees"] = record["trees"][1:]  # a registered tree absent and not listed as not checked
    with pytest.raises(scorer.InputError):
        scorer.score_g0(record, FAMILIES, TREES)


def _matched(values):
    """lighting_effect rows: {(model, lighting): [8 per-tree values]}."""
    rows = []
    for (model, lighting), per_tree in values.items():
        rows += [
            {"model": model, "family": FAMILIES[t], "tree_id": t, "lighting": lighting, "mask_mae_m": v}
            for t, v in zip(TREES, per_tree, strict=True)
        ]
    return {"lighting_effect": rows}


def _g2(b_over_c, a_over_j):
    return _matched(
        {
            ("da2_rendered_control", "evening"): [b_over_c] * 4 + [1.0] * 4,
            ("da2_control", "evening"): [1.0] * 8,
            ("da2_rendered_jitter", "evening"): [1.0] * 4 + [a_over_j] * 4,
            ("da2_jitter", "evening"): [1.0] * 8,
        }
    )


def test_g2_boundaries_and_its_ufo_clause_never_refutes(scorer):
    assert scorer.score_g2(_g2(0.3, 0.5), TREES, FAMILIES)["verdict"] == "supported"
    envy_refuted = scorer.score_g2(_g2(0.6, 0.5), TREES, FAMILIES)
    assert envy_refuted["verdict"] == "refuted" and envy_refuted["clauses"][0]["refutes"]
    ufo_bad = scorer.score_g2(_g2(0.3, 0.9), TREES, FAMILIES)
    assert ufo_bad["verdict"] == "partly supported" and not ufo_bad["clauses"][1]["refutes"]


def _g3(ratio, published_value=1.0):
    matched = _matched({("da2_frozen", "source"): [ratio * published_value] * 8})
    published = _matched({("da2_frozen", "source"): [published_value] * 8})
    return matched, published


def test_g3_boundaries(scorer):
    assert scorer.score_g3(*_g3(0.85), TREES)["verdict"] == "supported"
    assert scorer.score_g3(*_g3(1.0), TREES)["verdict"] == "refuted"
    assert scorer.score_g3(*_g3(0.9), TREES)["verdict"] == "partly supported"


def test_an_incomplete_cell_unscores_its_tree_and_too_few_scored_trees_are_not_decidable(scorer):
    from collections import Counter

    values = _matched(
        {
            ("da2_rendered_jitter", "source"): [0.9] * 8,
            ("da2_rendered_control", "source"): [0.9] * 8,
            ("da2_control", "source"): [1.0] * 8,
        }
    )
    cells = Counter(
        {(m, t, "source"): 6 for m in ("da2_rendered_jitter", "da2_rendered_control", "da2_control") for t in TREES}
    )
    cells[("da2_control", "t0", "source")] = 5
    g1 = scorer.score_g1(values, TREES, cells=cells, universe=TREES)
    observed = g1["clauses"][0]["observed"]
    assert "t0" in observed["unscored"] and "5 views" in observed["unscored"]["t0"]
    assert observed["scored"] == 7 and g1["verdict"] == "supported"
    five = scorer.score_g1(values, TREES[:5], universe=TREES)
    assert five["verdict"] == "partly supported"
    assert five["clauses"][0]["observed"]["decidable"] is False and five["clauses"][0]["observed"]["of"] == 8


def test_reading_gives_only_the_registered_cases(scorer):
    assert scorer.reading("supported", "supported")["text"].startswith("The source gain was renderer robustness")
    assert scorer.reading("refuted", "supported")["text"].startswith("Both gains are real")
    for g1 in ("supported", "partly supported", "refuted"):
        assert scorer.reading(g1, "refuted")["text"].startswith("The lighting claim does not survive")
    partly = scorer.reading("partly supported", "supported")
    assert partly["text"] is None and "names no reading" in partly["note"]


def test_interpretations_record_the_open_rules(scorer):
    text = " ".join(scorer.INTERPRETATIONS)
    for phrase in ("not_checked", "cannot refute", "absolute", "4 dp", "Unrounded"):
        assert phrase in text


def _data():
    """A minimal consistent input set for the integrity checks (two trees per arm)."""
    trees = ["t0", "t1"]
    plan = {
        "protocol": "docs/EVAL_PROTOCOL_RENDER_GAP_2026-09-30.md",
        "trees": [{"tree_id": t, "family": f} for t, f in zip(TREES, ["envy"] * 4 + ["ufo"] * 4, strict=True)],
        "models": {label: {"checkpoint_sha256": label * 8} for label in "FJCAB"},
        "pose_identity": {"tolerance_m": 0.001, "min_fraction": 0.995, "min_iou": 0.995},
        "matrix_evaluation_plan_sha256": "p" * 64,
    }
    published_frames = [
        {
            "tree_id": t,
            "lighting": "source",
            "view_id": "v0",
            "depth": f"pub/{t}.npy",
            "mask": f"pub/{t}.png",
            "K": [1],
            "target_pixel_xy": [1, 2],
            "rgb": f"pub/{t}.png",
        }
        for t in trees
    ]
    matched_frames = [{**f, "rgb": f"new/{f['tree_id']}.png"} for f in published_frames]
    native_frames = [
        {
            "tree_id": t,
            "lighting": "source",
            "view_id": "v0",
            "depth": f"x/render-gap-20260930/render/{t}/d.npy",
            "mask": f"x/render-gap-20260930/render/{t}/m.png",
            "rgb": f"x/render-gap-20260930/render/{t}/r.png",
        }
        for t in trees
    ]
    return {
        "plan": plan,
        "published_plan_sha256": "p" * 64,
        "pose_identity": {
            "gate": {"tolerance_m": 0.001, "min_fraction_within_tolerance": 0.995, "min_mask_iou": 0.995}
        },
        "published_plan_frames": published_frames,
        "arm_plan_frames": {"matched": matched_frames, "native": native_frames},
        "matched": {"frames": [{"tree_id": t} for t in trees]},
        "native": {"frames": [{"tree_id": t} for t in trees]},
        "render_status": {"trees": [{"tree_id": t, "matched": True} for t in trees]},
        "arm_status": {"matched": {"render_manifests": [f"m/manifests/{t}/render_manifest.json" for t in trees]}},
    }


def test_each_integrity_check_refuses_its_own_defect(scorer, tmp_path):
    (tmp_path / "docs/evidence").mkdir(parents=True)
    for path in scorer.PUBLISHED_PINS:
        (tmp_path / path).write_text("{}")
    with pytest.raises(scorer.InputError, match="I1"):
        scorer.check_pins(tmp_path)
    data = _data()
    scorer.check_plan(data)
    scorer.check_gate(data)
    scorer.check_tree_sets(data, {"t0", "t1"}, {"t0", "t1"})
    scorer.check_ground_truth(data)
    with pytest.raises(scorer.InputError, match="I2"):
        scorer.check_plan({**data, "plan": {**data["plan"], "protocol": "other.md"}})
    with pytest.raises(scorer.InputError, match="I3"):
        scorer.check_gate({**data, "pose_identity": {"gate": {"tolerance_m": 0.002}}})
    with pytest.raises(scorer.InputError, match="I6"):
        scorer.check_tree_sets(data, {"t0", "t1"}, {"t0"})
    moved = _data()
    moved["arm_plan_frames"]["matched"][0]["depth"] = "elsewhere.npy"
    with pytest.raises(scorer.InputError, match="I7"):
        scorer.check_ground_truth(moved)
    same_rgb = _data()
    same_rgb["arm_plan_frames"]["matched"][0]["rgb"] = same_rgb["published_plan_frames"][0]["rgb"]
    with pytest.raises(scorer.InputError, match="I7"):
        scorer.check_ground_truth(same_rgb)


def test_aggregate_links_are_checked_against_the_evaluation_files(scorer, tmp_path):
    batch = "artifacts/generalization/render-gap-20260930"
    data = {"plan": {"models": {label: {"checkpoint_sha256": label * 8} for label in "FJCAB"}}}
    queries = {"matrix": [{"file": "01.sql"}], "anchoring": [{"file": "01_cells.sql"}]}
    data["published"] = {"queries": queries["matrix"]}
    data["published_anchoring"] = {"queries": queries["anchoring"]}
    for arm in ("matched", "native"):
        for kind, name in (("matrix", arm), ("anchoring", f"{arm}_anchoring")):
            inputs = []
            for label, model in scorer.MODEL.items():
                path = tmp_path / batch / f"eval/{arm}/{label}/evaluation.json"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(f"{arm} {label}")
                inputs.append(
                    {
                        "model": model,
                        "path": f"{batch}/eval/{arm}/{label}/evaluation.json",
                        "sha256": scorer.sha256(path),
                        "checkpoint_sha256": label * 8,
                        "frames": 192,
                    }
                )
            data[name] = {"inputs": inputs, "queries": queries[kind]}
    scorer.check_aggregates(tmp_path, data)
    (tmp_path / batch / "eval/native/B/evaluation.json").write_text("changed")
    with pytest.raises(scorer.InputError, match="I5"):
        scorer.check_aggregates(tmp_path, data)
