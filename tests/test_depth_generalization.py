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


def test_masked_target_scores_only_tree_pixels():
    # A one-pixel-wide spur at column 1 in a 3x3 window; background is far away.
    gt = np.full((3, 3), 5.0)
    gt[:, 1] = 2.0
    pred = np.full((3, 3), 5.0)
    pred[:, 1] = 2.1
    mask = np.zeros((3, 3), dtype=bool)
    mask[:, 1] = True
    unmasked = m.target_metrics(pred, gt, [1, 1])
    masked = m.target_metrics(pred, gt, [1, 1], mask=mask)
    # Unmasked, six of nine pixels are background and the reference is a blend.
    assert unmasked["gt_pixels"] == 9 and unmasked["mask_gated"] is False
    # Masked, only the three spur pixels count and the error is the spur's.
    assert masked["gt_pixels"] == 3 and masked["mask_gated"] is True
    assert masked["window_tree_pixels"] == 3
    assert masked["mae_m"] == pytest.approx(0.1)
    assert masked["reference_m"] == pytest.approx(2.0)


def test_masked_target_with_no_tree_pixels_is_invalid_not_background():
    gt, pred = np.full((3, 3), 5.0), np.full((3, 3), 5.0)
    empty = np.zeros((3, 3), dtype=bool)
    got = m.target_metrics(pred, gt, [1, 1], mask=empty)
    assert got["valid"] is False
    assert got["reason"] == "no_tree_pixels_in_target_window"
    assert got["predicted_m"] is None


def test_masked_target_rejects_a_mismatched_mask():
    gt = np.ones((3, 3))
    with pytest.raises(ValueError, match="Mask shape mismatch"):
        m.target_metrics(gt, gt, [1, 1], mask=np.ones((2, 2), dtype=bool))


def test_an_evaluation_document_with_an_embedded_plan_is_accepted(tmp_path, monkeypatch):
    import json

    class Args:
        manifest = tmp_path / "evaluation.json"
        output = tmp_path / "out"
        companion = tmp_path
        checkpoint = tmp_path / "ckpt"
        checkpoint_sha256 = "0" * 64
        device = "cpu"
        save_predictions = False

    Args.manifest.write_text(json.dumps({"ok": True, "plan": {"frames": [], "depth_convention": "z"}, "rows": []}))
    Args.checkpoint.write_bytes(b"")
    with pytest.raises(ValueError, match="Checkpoint hash"):
        m.evaluate(Args)
    written = json.loads((Args.output / "evaluation.json").read_text())
    assert written["plan"]["depth_convention"] == "z" and "frames" in written["plan"]
