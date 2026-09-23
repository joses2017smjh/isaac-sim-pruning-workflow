"""The disparity-space ceiling recovers a known affine, refuses degenerate frames, and stays a ceiling."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def rel(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("relative_depth_ceiling", TOOLS / "relative_depth_ceiling.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_fit_recovers_an_exact_affine_in_disparity_space(rel):
    rng = np.random.default_rng(0)
    gt = rng.uniform(0.2, 3.0, size=(40, 60)).astype(np.float32)
    disparity = (4.0 / gt - 0.5).astype(np.float32)  # 1/gt = 0.25 * d + 0.125
    valid = rel.valid_pixels(disparity, gt)
    fit = rel.disparity_affine_ceiling(disparity, gt, valid)
    assert fit["scale"] == pytest.approx(0.25, abs=1e-5) and fit["shift"] == pytest.approx(0.125, abs=1e-5)
    assert fit["mae_m"] < 1e-4 and fit["correlation"] == pytest.approx(1.0)
    assert fit["pixels"] == 2400


def test_uninformative_disparity_has_no_ceiling(rel):
    gt = np.full((20, 20), 2.0, dtype=np.float32)
    gt[5:, :] = 1.0
    flat = np.ones_like(gt)
    fit = rel.disparity_affine_ceiling(flat, gt, rel.valid_pixels(flat, gt))
    assert fit["mae_m"] is None and fit["scale"] is None
    few = rel.disparity_affine_ceiling(flat, gt, rel.valid_pixels(flat, gt) & (np.arange(400).reshape(20, 20) < 10))
    assert few["pixels"] == 10 and few["mae_m"] is None


def test_mask_and_target_window_gate_the_fit(rel):
    rng = np.random.default_rng(1)
    gt = rng.uniform(0.5, 2.5, size=(30, 30)).astype(np.float32)
    disparity = (1.0 / gt).astype(np.float32)
    disparity[:, :15] = 5.0  # garbage outside the tree mask
    mask = np.zeros_like(gt, dtype=bool)
    mask[:, 15:] = True
    row = rel.score_frame(disparity, gt, mask, [20, 10])
    assert row["frame_fit"]["mae_m"] < 1e-4 and row["frame_fit"]["pixels"] == 450
    assert row["target"]["pixels"] == 9 and row["target"]["mae_m"] < 1e-4
    assert rel.score_frame(disparity, gt, mask, [2, 2])["target"] is None  # outside the mask
    assert rel.score_frame(disparity, gt, mask, [-5, 2])["target"] is None
    assert rel.score_frame(disparity, gt, mask, None)["target"] is None
