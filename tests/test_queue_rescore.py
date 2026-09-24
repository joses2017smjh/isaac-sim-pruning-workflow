"""The rescoring launcher pins every checkpoint and plan and refuses an empty pairing."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("queue_rescore", TOOLS / "queue_rescore.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_every_pair_is_registered_and_hashed_when_present(tmp_path, launcher):
    ckpt, plan = tmp_path / "best.pth", tmp_path / "plan.json"
    ckpt.write_bytes(b"w")
    plan.write_text("{}")
    result = launcher.rescore_plan({"jitter": ckpt}, {"stage_a": plan})
    assert len(result["checkpoints"]["jitter"]["sha256"]) == 64 and len(result["plans"]["stage_a"]["sha256"]) == 64
    assert result["maximum_gpu_minutes"] == 20 and result["max_concurrent_gpus"] == 1
    with pytest.raises(FileNotFoundError):
        launcher.rescore_plan({"missing": tmp_path / "none.pth"}, {"stage_a": plan})
    with pytest.raises(ValueError):
        launcher.rescore_plan({}, {"stage_a": plan})


def test_dirty_tree_is_refused(tmp_path, monkeypatch, launcher):
    monkeypatch.setattr(launcher, "_git", lambda root, *args: b"x\n")
    with pytest.raises(ValueError):
        launcher.queue_batch(tmp_path, "rescore-test", {"a": "b"}, {"c": "d"}, share_used_bytes=1)
