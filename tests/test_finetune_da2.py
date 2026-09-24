"""The fine-tune wrapper: jitter stays in range and is off in the control arm; manifests filter honestly."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def ft(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("finetune_da2", TOOLS / "finetune_da2.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_jitter_stays_in_unit_range_and_records_its_draw(ft):
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 1, size=(20, 30, 3)).astype(np.float32)
    applied = 0
    for _ in range(50):
        out, drawn = ft.photometric_jitter(image, rng)
        assert out.dtype == np.float32 and out.shape == image.shape
        assert out.min() >= 0.0 and out.max() <= 1.0
        if drawn["applied"]:
            applied += 1
            assert 0.35 <= drawn["brightness"] <= 1.3 and 0.6 <= drawn["gamma"] <= 1.6
            assert all(0.8 <= g <= 1.2 for g in drawn["channel_gains"])
        else:
            np.testing.assert_array_equal(out, image)
    assert 25 <= applied <= 50
    never = dict(ft.JITTER, probability=0.0)
    out, drawn = ft.photometric_jitter(image, rng, never)
    assert not drawn["applied"] and np.array_equal(out, image)


def test_manifest_filter_keeps_only_complete_rows_and_reports_drops(tmp_path, ft):
    rows = []
    for i in range(3):
        files = {k: tmp_path / f"{i}_{k}" for k in ("rgb_path", "depth_path", "mask_path")}
        for k, p in files.items():
            if not (i == 1 and k == "depth_path"):
                p.write_bytes(b"x")
        rows.append({"tree": f"t{i}", **{k: str(p) for k, p in files.items()}})
    source = tmp_path / "src.csv"
    with source.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = tmp_path / "out.csv"
    report = ft.filter_manifest(source, out)
    assert report["kept"] == 2 and report["dropped"] == 1 and len(report["source_sha256"]) == 64
    kept = list(csv.DictReader(out.open()))
    assert [r["tree"] for r in kept] == ["t0", "t2"]
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            {"frames": [{"rgb": "a", "depth": "b", "mask": "c", "condition": "evening"}, {"rgb": "d", "depth": "e"}]}
        )
    )
    ft.plan_frames_to_manifest(plan, tmp_path / "extra.csv")
    extra = list(csv.DictReader((tmp_path / "extra.csv").open()))
    assert len(extra) == 1 and extra[0]["condition"] == "evening"


def test_silog_and_eval_depth_are_the_companion_formulas(ft):
    torch = pytest.importorskip("torch")
    pred = torch.tensor([1.0, 2.0, 4.0])
    target = torch.tensor([1.0, 2.0, 4.0])
    loss = ft.SiLogLoss(torch)(pred, target, torch.ones(3, dtype=torch.bool))
    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    metrics = ft.eval_depth(torch, pred * 1.1, target)
    assert metrics["abs_rel"] == pytest.approx(0.1, abs=1e-6) and metrics["d1"] == 1.0
