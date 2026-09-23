import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location(
    "depth_generalization", Path(__file__).parents[1] / "tools/depth_generalization.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_invalid_predictions_reduce_coverage_without_hiding_error():
    got = m.depth_metrics(np.array([[1.0, np.nan], [3.0, -1.0]]), np.ones((2, 2)))
    assert got["valid_pixel_rate"] == 0.5
    assert got["rmse_m"] == pytest.approx(2**0.5)
    assert got["outlier_gt_20mm_fraction"] == 0.5


def test_missing_target_and_slow_inference_cannot_pass():
    assert not m.sanity_gates([])["ok"]
    row = {"inference_seconds": 0.2, "target": m.target_metrics(np.ones((3, 3)), np.ones((3, 3)), [1, 1])}
    assert not m.sanity_gates([row])["checks"]["p95_inference_le_100ms"]
    row["inference_seconds"] = 0.01
    assert m.sanity_gates([row])["ok"]
    row["target"] = m.target_metrics(np.ones((3, 3)) * 1.2, np.ones((3, 3)), [1, 1])
    assert not m.sanity_gates([row])["ok"]


def test_mask_and_border_target_are_explicit():
    gt = np.ones((3, 3))
    pred = gt.copy()
    pred[0, 0] = 2
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    assert m.depth_metrics(pred, gt, mask)["mae_m"] == 1
    assert m.target_metrics(pred, gt, [0, 0])["gt_pixels"] == 4
    assert not m.target_metrics(pred, gt, [-1, 0])["valid"]
    with pytest.raises(ValueError):
        m.depth_metrics(pred, gt[:2])
