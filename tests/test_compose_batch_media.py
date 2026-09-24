"""Media composition never invents a capture and never overwrites without --force."""

from __future__ import annotations

import importlib.util
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1] / "tools"


def test_runs_without_capture_are_skipped_and_existing_media_is_kept(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("compose_batch_media", TOOLS / "compose_batch_media.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    batch = tmp_path / "batch"
    stub = batch / "run_00_source_tree0_v1_baseline"
    stub.mkdir(parents=True)
    recorded = batch / "run_01_source_tree0_v2_fine_step"
    (recorded / "media").mkdir(parents=True)
    (recorded / "frames.json").write_text("{}")
    (recorded / "experiment_result.json").write_text("{}")
    (recorded / "media" / "run_01_tree0_v2_fine_step.gif").write_bytes(b"gif")
    fresh = batch / "run_02_source_tree0_v3_fine_step"
    fresh.mkdir()
    (fresh / "frames.json").write_text("{}")
    (fresh / "experiment_result.json").write_text("{}")
    results = {r["run"]: r for r in (m.compose(run, "python", dry_run=True) for run in m.runs_of(batch))}
    assert results[stub.name]["status"] == "skipped_no_capture"
    assert results[recorded.name]["status"] == "kept"
    assert results[fresh.name]["status"] == "would_compose"
    assert results[fresh.name]["command"][-1] == "run_02_tree0_v3_fine_step"
    assert m.compose(recorded, "python", force=True, dry_run=True)["status"] == "would_compose"
