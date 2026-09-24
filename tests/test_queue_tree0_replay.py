"""Protect the tree0 replay launcher: fixed runs, frame count from the plan, honest preflight."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("queue_tree0_replay", TOOLS / "queue_tree0_replay.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_registers_three_runs_two_barks_and_every_selected_frame(launcher):
    plan = launcher.replay_plan(hash_assets=False)
    assert [r["name"] for r in plan["runs"]] == list(launcher.RUNS) and len(plan["runs"]) == 3
    assert plan["frames_per_run"] == 26 and plan["registered_frames"] == 26 * 3 * 2
    assert plan["max_frame"] == 77 and plan["frame_stride"] == 3
    assert plan["maximum_gpu_minutes"] == 50 and plan["max_concurrent_gpus"] == 1
    assert plan["models"]["da2"]["checkpoint_sha256"].startswith("5ecc5182")
    assert len(plan["relative_head"]["checkpoint_sha256"]) == 64 and "extra_frames" not in plan["relative_head"]
    with pytest.raises(ValueError):
        launcher.replay_plan(runs=("run_00_source_raw", "run_00_source_raw"), hash_assets=False)


def test_dirty_tree_and_bad_batch_id_are_refused(tmp_path, monkeypatch, launcher):
    monkeypatch.setattr(launcher, "_git", lambda root, *args: b"tools/x.py\n")
    with pytest.raises(ValueError):
        launcher.queue_batch(tmp_path, "replay-test", share_used_bytes=1)
    with pytest.raises(ValueError):
        launcher.queue_batch(tmp_path, "../x", share_used_bytes=1)
    assert not (tmp_path / "artifacts").exists()
