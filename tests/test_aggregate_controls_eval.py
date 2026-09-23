"""Controls aggregation: paired against the registered baseline, sweep by distance, gates per cell."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

duckdb = pytest.importorskip("duckdb")
PIL = pytest.importorskip("PIL")
TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def aggregator(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("aggregate_controls_eval", TOOLS / "aggregate_controls_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(condition, tree="lpy_envy_00000", mae=0.15, **over):
    family = tree.split("_")[1]
    row = {
        "model": "da2",
        "family": family,
        "tree_id": tree,
        "lighting": condition.split("/")[0],
        "condition": condition,
        "condition_group": "lighting",
        "camera_model": "training_rig",
        "pose_set": "far",
        "view_id": "rig0_l",
        "target_visible": True,
        "mask_pixels": 500,
        "mask_mae_m": mae,
        "mask_rmse_m": mae,
        "mask_abs_relative": 0.07,
        "mask_p95_abs_m": 0.2,
        "mask_signed_median_m": -mae,
        "mask_signed_mean_m": -mae,
        "mask_mae_debiased_m": 0.05,
        "full_mae_m": 0.3,
        "full_rmse_m": 0.4,
        "full_signed_median_m": -0.2,
        "target_valid": True,
        "target_mae_m": 0.3,
        "target_masked_valid": True,
        "target_masked_mae_m": 0.1,
        "target_masked_p95_abs_m": 0.12,
        "target_masked_abs_relative": 0.05,
        "target_masked_coverage": 1.0,
        "target_masked_tree_pixels": 3,
        "target_masked_predicted_m": 2.1,
        "target_masked_reference_m": 2.0,
        "target_masked_signed_m": 0.1,
        "inference_seconds": 0.05,
    }
    row.update(over)
    return row


def test_paired_effect_uses_the_registered_baseline_within_each_tree(aggregator):
    rows = []
    for tree, source, evening in (("lpy_envy_00000", 0.10, 0.60), ("lpy_ufo_00001", 0.20, 0.80)):
        rows += [_row("source", tree, source), _row("evening", tree, evening), _row("evening/gamma", tree, evening / 2)]
    result = aggregator.aggregate(rows)
    effect = {(r["family"], r["condition"]): r for r in result["paired_effect"]}
    assert effect[("envy", "evening")]["baseline"] == "source"
    assert effect[("envy", "evening")]["mean_ratio_to_baseline"] == pytest.approx(6.0)
    assert effect[("ufo", "evening")]["mean_ratio_to_baseline"] == pytest.approx(4.0)
    # The photometric variant is paired with the plain evening cell, not with source.
    assert effect[("envy", "evening/gamma")]["baseline"] == "evening"
    assert effect[("envy", "evening/gamma")]["mean_ratio_to_baseline"] == pytest.approx(0.5)
    assert effect[("envy", "evening/gamma")]["trees_improved"] == 1
    assert "source" not in {r["condition"] for r in result["paired_effect"]}
    cells = {(r["family"], r["condition"]): r for r in result["cells"]}
    assert cells[("envy", "evening")]["n_trees"] == 1 and cells[("ufo", "source")]["mask_mae_mean_m"] == 0.2


def test_sweep_reports_predicted_and_reference_side_by_side(aggregator):
    rows = [
        _row(
            "source/sweep/training_rig",
            pose_set="sweep",
            nominal_distance_m=d,
            view_id=f"sweep{i}",
            target_masked_predicted_m=p,
            target_masked_reference_m=d if p is not None else None,
            target_masked_signed_m=p - d if p is not None else None,
            target_masked_valid=p is not None,
        )
        for i, (d, p) in enumerate(((0.16, 0.85), (0.39, 0.9), (2.3, 2.2), (0.9, None)))
    ]
    sweep = {r["nominal_distance_m"]: r for r in aggregator.aggregate(rows)["sweep"]}
    assert sweep[0.16]["target_ratio_mean"] == pytest.approx(0.85 / 0.16, abs=0.001)
    assert sweep[2.3]["target_signed_mean_m"] == pytest.approx(-0.1)
    assert sweep[0.9]["n_frames"] == 1 and sweep[0.9]["target_frames"] == 0
    assert sweep[0.9]["target_predicted_mean_m"] is None


def test_gates_are_per_condition_and_never_pass_on_nothing(aggregator):
    rows = [_row("source"), _row("evening", target_masked_valid=False, target_masked_p95_abs_m=None)]
    gates = {g["condition"]: g for g in aggregator.aggregate(rows)["gates"]}
    assert gates["evening"]["frames_with_target"] == 0 and not gates["evening"]["all_gates_pass"]
    assert gates["source"]["frames_with_target"] == 1 and not gates["source"]["all_gates_pass"]  # p95 0.12 > 0.02
    good = [_row("source", target_masked_p95_abs_m=0.01)]
    assert aggregator.aggregate(good)["gates"][0]["all_gates_pass"]


def test_every_controls_query_is_applied_and_hashed(aggregator):
    result = aggregator.aggregate([_row("source")])
    names = [q["file"] for q in result["queries"]]
    assert names == sorted(p.name for p in aggregator.SQL_DIR.glob("*.sql")) and len(names) == 5
    assert all(len(q["sha256"]) == 64 for q in result["queries"])
