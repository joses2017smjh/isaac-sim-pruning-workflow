"""Protect the fine-tune launcher: two arms only, pinned warm start, honest budget."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("queue_finetune", TOOLS / "queue_finetune.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_arms_differ_only_in_the_jitter(launcher):
    jitter = launcher.finetune_plan("jitter", hash_assets=False)
    control = launcher.finetune_plan("control", hash_assets=False)
    assert jitter["jitter"]["probability"] == 0.8 and control["jitter"] is None
    for key in ("epochs", "warm_start", "max_runtime_seconds", "maximum_gpu_minutes", "evaluation_plans"):
        assert jitter[key] == control[key]
    assert jitter["warm_start"]["checkpoint_sha256"].startswith("5ecc5182")
    assert jitter["maximum_gpu_minutes"] == 8 * 60 + 60 and jitter["max_concurrent_gpus"] == 1
    assert jitter["max_runtime_seconds"] < 8 * 3600
    assert set(jitter["evaluation_plans"]) == {"family_matrix", "controls", "stage_a"}
    with pytest.raises(ValueError):
        launcher.finetune_plan("both", hash_assets=False)


def test_dirty_tree_and_bad_batch_id_are_refused(tmp_path, monkeypatch, launcher):
    monkeypatch.setattr(launcher, "_git", lambda root, *args: b"tools/x.py\n")
    with pytest.raises(ValueError):
        launcher.queue_batch(tmp_path, "ft-test", "jitter", share_used_bytes=1)
    with pytest.raises(ValueError):
        launcher.queue_batch(tmp_path, "../x", "jitter", share_used_bytes=1)
    assert not (tmp_path / "artifacts").exists()


def test_rendered_arms_train_on_registered_render_batches_with_matched_steps(launcher, tmp_path):
    batch = tmp_path / "training-lighting-full"
    batch.mkdir()
    (batch / "plan.json").write_text("{}")
    rendered = launcher.finetune_plan("rendered_jitter", hash_assets=False, rendered_batches=[batch])
    control = launcher.finetune_plan("rendered_control", hash_assets=False, rendered_batches=[batch])
    assert rendered["trainer_arm"] == "jitter" and control["trainer_arm"] == "control"
    assert rendered["jitter"]["probability"] == 0.8 and control["jitter"] is None
    assert rendered["epochs"] == control["epochs"] == 3
    assert (
        rendered["rendered_batches"] == [str(batch)]
        and "built inside the training job" in rendered["train_manifest_rule"]
    )
    assert rendered["protocol"].endswith("LIGHTING_TRAINING_2026-09-27.md")
    # The published arms are unchanged: six epochs, no render batches.
    published = launcher.finetune_plan("jitter", hash_assets=False)
    assert published["epochs"] == 6 and published["rendered_batches"] is None and published["trainer_arm"] == "jitter"
    with pytest.raises(ValueError):
        launcher.finetune_plan("rendered_jitter", hash_assets=False)
    with pytest.raises(ValueError):
        launcher.finetune_plan("jitter", hash_assets=False, rendered_batches=[batch])
    with pytest.raises(FileNotFoundError):
        launcher.finetune_plan("rendered_control", rendered_batches=[tmp_path / "missing"])
