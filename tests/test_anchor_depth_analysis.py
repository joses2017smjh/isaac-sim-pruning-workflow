"""Protect the anchoring analysis: one range fixes one parameter, the ceiling is labelled, N never shrinks."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

duckdb = pytest.importorskip("duckdb")
PIL = pytest.importorskip("PIL")


@pytest.fixture
def anchoring(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("anchor_depth_analysis", tools / "anchor_depth_analysis.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _scene():
    """A depth ramp on a tree mask, so scale and shift are distinguishable."""
    gt = np.tile(np.linspace(1.0, 3.0, 16), (12, 1)).astype(float)
    mask = np.zeros_like(gt, dtype=bool)
    mask[3:9, :] = True
    return gt, mask


def test_a_pure_offset_is_removed_by_shift_and_not_by_scale(anchoring):
    gt, mask = _scene()
    pred = gt - 0.2
    row = anchoring.score_frame(pred, gt, mask, pixel=[8, 6], target_visible=True)
    assert row["anchored"] is True
    assert row["raw_mae_m"] == pytest.approx(0.2)
    assert row["shift_mae_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["scale_mae_m"] > 0.01
    # The ceiling is at least as good as the best single anchor.
    assert row["affine_ceiling_mae_m"] <= row["shift_mae_m"] + 1e-9


def test_a_pure_scale_error_is_removed_by_scale_and_not_by_shift(anchoring):
    gt, mask = _scene()
    pred = gt * 2.5
    row = anchoring.score_frame(pred, gt, mask, pixel=[8, 6], target_visible=True)
    assert row["scale_mae_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["shift_mae_m"] > 0.1


def test_affine_ceiling_recovers_scale_and_shift_but_only_as_a_ceiling(anchoring):
    gt, mask = _scene()
    pred = 0.5 * gt + 0.7
    row = anchoring.score_frame(pred, gt, mask, pixel=[8, 6], target_visible=True)
    assert row["affine_ceiling_mae_m"] == pytest.approx(0.0, abs=1e-9)
    assert row["affine_scale"] == pytest.approx(2.0)
    assert row["affine_shift_m"] == pytest.approx(-1.4)
    # A single anchor cannot fix both parameters; both single variants stay wrong.
    assert row["shift_mae_m"] > 0.05 and row["scale_mae_m"] > 0.05


def test_flat_prediction_has_no_correlation_and_no_anchor_rescues_it(anchoring):
    gt, mask = _scene()
    pred = np.full_like(gt, 0.85)
    row = anchoring.score_frame(pred, gt, mask, pixel=[8, 6], target_visible=True)
    assert row["pred_gt_correlation"] is None  # zero variance in the prediction
    assert row["shift_mae_m"] > 0.3 and row["scale_mae_m"] > 0.3


def test_invisible_or_missing_target_leaves_the_frame_unanchored_but_counted(anchoring):
    gt, mask = _scene()
    pred = gt - 0.2
    hidden = anchoring.score_frame(pred, gt, mask, pixel=[8, 6], target_visible=False)
    absent = anchoring.score_frame(pred, gt, mask, pixel=None, target_visible=True)
    outside = anchoring.score_frame(pred, gt, mask, pixel=[99, 99], target_visible=True)
    for row in (hidden, absent, outside):
        assert row["anchored"] is False
        assert row["shift_mae_m"] is None and row["scale_mae_m"] is None
        assert row["raw_mae_m"] == pytest.approx(0.2)


def test_anchor_window_is_mask_gated_so_background_cannot_set_the_range(anchoring):
    gt, mask = _scene()
    gt[~mask] = 9.0  # far background around the tree
    pred = gt.copy()
    pred[mask] -= 0.2
    # Target at the mask edge: window mixes tree and background pixels.
    row = anchoring.score_frame(pred, gt, mask, pixel=[8, 3], target_visible=True)
    assert row["anchored"] is True
    assert row["anchor_range_m"] < 9.0
    assert row["shift_mae_m"] == pytest.approx(0.0, abs=1e-9)


def test_rows_and_cells_keep_unanchored_frames_in_n_frames(tmp_path, anchoring):
    from PIL import Image

    gt, mask = _scene()
    np.save(tmp_path / "gt.npy", gt.astype(np.float32))
    np.save(tmp_path / "pred.npy", (gt - 0.2).astype(np.float32))
    Image.fromarray((mask * 255).astype(np.uint8)).save(tmp_path / "mask.png")

    def frame(visible):
        return {
            "tree_id": "lpy_ufo_00001",
            "lighting": "source",
            "view_id": "v",
            "depth": str(tmp_path / "gt.npy"),
            "mask": str(tmp_path / "mask.png"),
            "target_pixel_xy": [8, 6] if visible else None,
            "target_visible": visible,
        }

    evaluation = {
        "rows": [
            {"frame": frame(True), "prediction": str(tmp_path / "pred.npy")},
            {"frame": frame(False), "prediction": str(tmp_path / "pred.npy")},
        ]
    }
    rows = anchoring.build_rows("da2", evaluation)
    assert [r["anchored"] for r in rows] == [True, False]
    assert rows[0]["anchor_source"].startswith("gt_at_target_pixel")
    cells = anchoring.aggregate(rows)["cells"]
    assert cells[0]["n_frames"] == 2 and cells[0]["n_anchored"] == 1
    assert cells[0]["shift_mae_m"] == pytest.approx(0.0, abs=1e-6)
    assert cells[0]["raw_mae_m"] == pytest.approx(0.2, abs=1e-6)


def test_isaac_style_frames_without_mask_use_the_controller_anchor_label(tmp_path, anchoring):
    gt, _ = _scene()
    np.save(tmp_path / "gt.npy", gt.astype(np.float32))
    np.save(tmp_path / "pred.npy", (gt * 2.0).astype(np.float32))
    evaluation = {
        "rows": [
            {
                "frame": {
                    "tree_id": "original_orchard_tree0",
                    "lighting": "source",
                    "sequence": "run_00",
                    "depth": str(tmp_path / "gt.npy"),
                    "target_pixel_xy": [8, 6],
                    "time_s": 0.1,
                },
                "prediction": str(tmp_path / "pred.npy"),
            }
        ]
    }
    (row,) = anchoring.build_rows("da2_isaac", evaluation)
    assert row["family"] == "original_orchard"
    assert row["anchor_source"].startswith("rtx_optical_z")
    assert row["scale_mae_m"] == pytest.approx(0.0, abs=1e-9)


def test_aggregating_nothing_is_refused(anchoring):
    with pytest.raises(ValueError, match="refusing to report over nothing"):
        anchoring.aggregate([])


def test_every_anchoring_query_is_applied_and_hashed(anchoring, tmp_path):
    gt, mask = _scene()
    row = {
        "model": "m",
        "family": "envy",
        "tree_id": "t",
        "lighting": "source",
        "view_id": "v",
        "sequence": None,
        "anchor_source": "x",
        **anchoring.score_frame(gt - 0.1, gt, mask, [8, 6], True),
    }
    result = anchoring.aggregate([row])
    names = [q["file"] for q in result["queries"]]
    assert names == ["01_cells.sql", "02_per_tree.sql"]
    assert json.dumps(result["cells"], default=str)


def test_conditions_under_one_light_are_separate_cells(anchoring, tmp_path):
    import json

    import numpy as np

    rows = []
    for condition, offset in (("source", 0.1), ("source/close/isaac_wrist", 0.5)):
        for view in ("a", "b"):
            gt = np.full((6, 6), 2.0)
            pred = gt + offset
            mask = np.ones_like(gt, dtype=bool)
            frame = {
                "tree_id": "t",
                "family": "ufo",
                "lighting": "source",
                "condition": condition,
                "view_id": view,
                "target_visible": True,
                "target_pixel_xy": [3, 3],
            }
            for key, array in (("depth", gt), ("prediction", pred)):
                path = tmp_path / f"{condition.replace('/', '_')}_{view}_{key}.npy"
                np.save(path, array)
                frame[key] = str(path)
            mask_path = tmp_path / f"{condition.replace('/', '_')}_{view}_mask.png"
            from PIL import Image

            Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)
            frame["mask"] = str(mask_path)
            rows.append({"frame": frame, "prediction": frame.pop("prediction")})
    evaluation_path = tmp_path / "evaluation.json"
    evaluation_path.write_text(json.dumps({"ok": True, "rows": rows}))
    built = anchoring.build_rows("da2", json.loads(evaluation_path.read_text()))
    cells = {c["condition"]: c for c in anchoring.aggregate(built)["cells"]}
    assert set(cells) == {"source", "source/close/isaac_wrist"}
    assert cells["source"]["raw_mae_m"] == pytest.approx(0.1) and cells["source/close/isaac_wrist"][
        "raw_mae_m"
    ] == pytest.approx(0.5)
    assert all(c["lighting"] == "source" for c in cells.values())


def test_zone_fit_recovers_scale_and_shift_from_zone_medians_but_not_from_too_few_zones(anchoring):
    import numpy as np

    rng = np.random.default_rng(3)
    gt = rng.uniform(0.5, 2.5, size=(64, 96))
    pred = (gt - 0.3) / 1.2  # true depth = 1.2 * pred + 0.3
    K = [[80.0, 0, 47.5], [0, 80.0, 31.5], [0, 0, 1]]
    valid = anchoring.valid_pixels(pred, gt, np.ones_like(gt, dtype=bool))
    n, err, scale, shift, fitted = anchoring.zone_fit(pred, gt, valid, K)
    assert n >= 20 and err < 1e-6 and scale == pytest.approx(1.2) and shift == pytest.approx(0.3)
    row = anchoring.score_frame(pred, gt, np.ones_like(gt, dtype=bool), [48, 32], True, K=K)
    assert row["zone_mae_m"] < 1e-6 and row["zone_target_abs_m"] < 1e-6 and row["raw_target_abs_m"] > 0.1
    # Zones cover only a 65 degree diagonal; a narrow mask leaves fewer than three zones.
    narrow = np.zeros_like(gt, dtype=bool)
    narrow[31:33, 47:49] = True
    n, err, *_ = anchoring.zone_fit(pred, gt, anchoring.valid_pixels(pred, gt, narrow), K)
    assert n < 3 and err is None
    assert anchoring.score_frame(pred, gt, None, None, None)["n_zones"] == 0
