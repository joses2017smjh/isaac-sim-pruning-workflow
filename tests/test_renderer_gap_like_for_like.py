"""Like-for-like renderer scoring: identical pixel sets, the tool excluded, nothing imputed."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def gap(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("renderer_gap_like_for_like", TOOLS / "renderer_gap_like_for_like.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_tool_pixels_and_their_edge_leave_the_shared_tree_set(gap):
    cycles = np.full((20, 20), 1e10)
    cycles[5:15, 5:15] = 0.30  # tree
    isaac = cycles.copy()
    isaac[12:18, 12:18] = 0.11  # tool in front of the tree and the sky
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 5:15] = True
    sets = gap.pixel_sets(isaac, cycles, mask)
    assert sets["occ"][12:18, 12:18].all() and sets["occ"].sum() == 36
    assert not sets["shared"][11:15, 11:15].any()  # tool plus its one-pixel edge
    assert sets["shared"][5:10, 5:10].all()
    # A 3 mm disagreement still counts as the same surface; 8 mm does not.
    isaac2 = cycles.copy()
    isaac2[5, 5] += 0.003
    isaac2[6, 6] += 0.008
    sets2 = gap.pixel_sets(isaac2, cycles, mask)
    assert sets2["agree"][5, 5] and not sets2["agree"][6, 6]


def test_scores_use_ground_truth_only_as_a_ceiling_and_empty_sets_stay_empty(gap):
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.2, 0.4, size=(10, 10))
    pred = 2.0 * gt + 0.1
    region = np.ones_like(gt, dtype=bool)
    row = gap.score(pred, gt, region)
    assert row["affine_ceiling_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["mae_m"] > 0.3 and row["correlation"] == pytest.approx(1.0)
    assert row["constant_floor_m"] == pytest.approx(np.mean(np.abs(gt - gt.mean())))
    empty = gap.score(pred, gt, np.zeros_like(region))
    assert empty["pixels"] == 0 and empty["affine_ceiling_m"] is None and empty["mae_m"] is None


def test_every_renderer_gap_query_is_applied(gap):
    rows = []
    for renderer, pixel_set, ceiling in (
        ("isaac", "isaac_on_shared_tree", 0.24),
        ("cycles", "cycles_on_shared_tree", 0.18),
    ):
        rows.append(
            {
                "renderer": renderer,
                "bark": "palm" if renderer == "cycles" else "not_applicable",
                "lighting": "source",
                "isaac_run": "run_00_source_raw",
                "isaac_frame_index": 0,
                "pixel_set": pixel_set,
                "pixels": 100,
                "mae_m": 0.4,
                "signed_median_m": 0.4,
                "median_prediction_m": 0.7,
                "median_reference_m": 0.3,
                "correlation": 0.8,
                "affine_ceiling_m": ceiling,
                "constant_floor_m": 0.29,
            }
        )
    result = gap.aggregate(rows)
    assert len(result["queries"]) == len(list(gap.SQL_DIR.glob("*.sql"))) == 2
    (paired,) = result["paired"]
    assert paired["gap_mean_m"] == pytest.approx(0.06) and paired["frames_isaac_worse"] == 1
    with pytest.raises(ValueError):
        gap.aggregate([])
