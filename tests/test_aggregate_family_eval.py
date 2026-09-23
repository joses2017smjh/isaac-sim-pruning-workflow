"""Protect the depth aggregation: trees not frames count, nulls stay null, gates never pass on nothing."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")
PIL = pytest.importorskip("PIL")


@pytest.fixture
def aggregator(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("aggregate_family_eval", tools / "aggregate_family_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _row(model="da2", family="envy", tree="lpy_envy_00000", light="source", view="rig0_l", **over):
    row = {
        "model": model,
        "family": family,
        "tree_id": tree,
        "lighting": light,
        "view_id": view,
        "sequence": None,
        "target_visible": True,
        "mask_pixels": 500,
        "mask_mae_m": 0.15,
        "mask_rmse_m": 0.16,
        "mask_abs_relative": 0.07,
        "mask_p95_abs_m": 0.2,
        "mask_signed_median_m": -0.14,
        "mask_signed_mean_m": -0.14,
        "mask_mae_debiased_m": 0.05,
        "full_mae_m": 0.3,
        "full_rmse_m": 0.4,
        "full_signed_median_m": -0.2,
        "target_valid": True,
        "target_mae_m": 0.3,
        "target_p95_abs_m": 0.3,
        "target_abs_relative": 0.15,
        "target_coverage": 1.0,
        "target_masked_valid": True,
        "target_masked_mae_m": 0.1,
        "target_masked_p95_abs_m": 0.12,
        "target_masked_abs_relative": 0.05,
        "target_masked_coverage": 1.0,
        "target_masked_tree_pixels": 3,
        "inference_seconds": 0.05,
    }
    row.update(over)
    return row


def test_family_is_derived_from_tree_id_when_not_recorded(aggregator):
    assert aggregator._family({"tree_id": "lpy_ufo_00001"}) == "ufo"
    assert aggregator._family({"tree_id": "lpy_envy_00003"}) == "envy"
    assert aggregator._family({"tree_id": "original_orchard_tree0"}) == "original_orchard"
    assert aggregator._family({"family": "ufo", "tree_id": "lpy_envy_00000"}) == "ufo"
    assert aggregator._family({}) is None


def test_trees_not_frames_are_the_population_count(aggregator):
    rows = [_row(tree=f"lpy_envy_{n:05d}", view=v) for n in (0, 3) for v in ("rig0_l", "rig0_r", "rig1_l")]
    cells = aggregator.aggregate(rows)["cells"]
    assert len(cells) == 1
    assert cells[0]["n_trees"] == 2
    assert cells[0]["n_frames"] == 6


def test_lighting_effect_is_paired_within_a_tree(aggregator):
    rows = [_row(light="source", mask_mae_m=0.1), _row(light="evening", mask_mae_m=0.6, view="rig0_r")]
    effect = {r["lighting"]: r for r in aggregator.aggregate(rows)["lighting_effect"]}
    assert effect["source"]["ratio_to_source"] == pytest.approx(1.0)
    assert effect["evening"]["ratio_to_source"] == pytest.approx(6.0)


def test_family_comparison_uses_per_tree_means_so_views_cannot_dominate(aggregator):
    rows = [_row(family="envy", tree="lpy_envy_00000", mask_mae_m=0.1, view=f"v{i}") for i in range(6)]
    rows += [_row(family="ufo", tree="lpy_ufo_00000", mask_mae_m=0.3, view="v0")]
    comparison = aggregator.aggregate(rows)["family_comparison"][0]
    assert (comparison["envy_trees"], comparison["ufo_trees"]) == (1, 1)
    assert comparison["ufo_minus_envy_m"] == pytest.approx(0.2)


def test_maskless_frames_keep_full_frame_numbers_and_null_mask_columns(aggregator):
    # Isaac Stage A frames have no tree mask; nothing may be imputed for them.
    rows = [
        _row(
            family="original_orchard",
            tree="original_orchard_tree0",
            mask_pixels=None,
            mask_mae_m=None,
            mask_rmse_m=None,
            mask_abs_relative=None,
            mask_p95_abs_m=None,
            mask_signed_median_m=None,
            mask_signed_mean_m=None,
            mask_mae_debiased_m=None,
            target_masked_valid=False,
            target_masked_mae_m=None,
            target_masked_p95_abs_m=None,
            target_masked_abs_relative=None,
            target_masked_coverage=None,
            target_masked_tree_pixels=None,
        )
    ]
    result = aggregator.aggregate(rows)
    cell = result["cells"][0]
    assert cell["mask_mae_mean_m"] is None
    assert cell["full_mae_mean_m"] == pytest.approx(0.3)
    assert result["lighting_effect"] == []
    assert result["family_comparison"] == []


def test_gates_never_pass_over_zero_frames_and_pass_only_when_every_check_holds(aggregator):
    none_valid = [_row(target_masked_valid=False, target_valid=False)]
    for gate in aggregator.aggregate(none_valid)["gates"]:
        assert gate["frames_with_target"] == 0
        assert gate["all_gates_pass"] is False

    good = [
        _row(
            target_masked_p95_abs_m=0.01,
            target_masked_abs_relative=0.05,
            target_masked_coverage=1.0,
            target_p95_abs_m=0.01,
            target_abs_relative=0.05,
            inference_seconds=0.05,
        )
    ]
    gates = {g["target_kind"]: g for g in aggregator.aggregate(good)["gates"]}
    assert gates["masked"]["all_gates_pass"] is True
    assert gates["unmasked"]["all_gates_pass"] is True

    slow = [dict(good[0], inference_seconds=0.25)]
    assert all(not g["all_gates_pass"] for g in aggregator.aggregate(slow)["gates"])


def test_aggregating_nothing_is_refused(aggregator):
    with pytest.raises(ValueError, match="refusing to report over nothing"):
        aggregator.aggregate([])


def test_every_family_query_is_applied_and_hashed(aggregator):
    result = aggregator.aggregate([_row()])
    names = [entry["file"] for entry in result["queries"]]
    assert names == sorted(names)
    assert names[0] == "01_frames.sql"
    assert all(len(entry["sha256"]) == 64 for entry in result["queries"])


def test_rows_recompute_bias_and_masked_target_from_saved_predictions(tmp_path, aggregator):
    from PIL import Image

    gt = np.full((8, 8), 2.0, dtype=np.float32)
    pred = np.full((8, 8), 1.8, dtype=np.float32)  # 0.2 m too near everywhere on the tree
    pred[:, 6:] = 5.0  # background column predicted far off; must not reach the mask stats
    mask = np.zeros((8, 8), dtype=bool)
    mask[:, 2:5] = True
    np.save(tmp_path / "gt.npy", gt)
    np.save(tmp_path / "pred.npy", pred)
    Image.fromarray((mask * 255).astype(np.uint8)).save(tmp_path / "mask.png")
    evaluation = {
        "model": {"checkpoint_sha256": "x"},
        "rows": [
            {
                "frame": {
                    "tree_id": "lpy_ufo_00001",
                    "lighting": "source",
                    "view_id": "rig0_l",
                    "depth": str(tmp_path / "gt.npy"),
                    "mask": str(tmp_path / "mask.png"),
                    "target_pixel_xy": [3, 4],
                    "target_visible": True,
                },
                "inference_seconds": 0.1,
                "all_valid_gt": {"mae_m": 0.9, "rmse_m": 1.0},
                "mask_metrics": {"gt_pixels": 24, "mae_m": 0.2, "rmse_m": 0.2, "abs_relative": 0.1, "p95_abs_m": 0.2},
                "target": {"valid": True, "mae_m": 0.2, "p95_abs_m": 0.2, "abs_relative": 0.1, "valid_pixel_rate": 1},
                "prediction": str(tmp_path / "pred.npy"),
            }
        ],
    }
    (row,) = aggregator.build_rows("da2", evaluation)
    assert row["family"] == "ufo"
    assert row["mask_signed_median_m"] == pytest.approx(-0.2)
    assert row["mask_mae_debiased_m"] == pytest.approx(0.0, abs=1e-6)
    # The masked target was absent from the evaluation and is recomputed on tree pixels only.
    assert row["target_masked_valid"] is True
    assert row["target_masked_mae_m"] == pytest.approx(0.2)
    assert row["target_masked_tree_pixels"] == 9

    (quiet,) = aggregator.build_rows("da2", evaluation, recompute=False)
    assert quiet["mask_signed_median_m"] is None
    assert quiet["target_masked_valid"] is False


def test_evaluation_input_records_checkpoint_provenance(tmp_path, aggregator, capsys):
    evaluation = {"model": {"checkpoint": "/c.pth", "checkpoint_sha256": "abc"}, "job_id": "7", "rows": []}
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(evaluation), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to report over nothing"):
        # No frames at all must not become an evidence file.
        aggregator.main(["--evaluation", f"da2={path}", "--output", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()
    capsys.readouterr()
