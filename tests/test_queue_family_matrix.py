"""Protect the matrix launcher: fixed register, both families, sized array, refused preflight."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def launcher(monkeypatch):
    tools = Path(__file__).resolve().parents[1] / "tools"
    monkeypatch.syspath_prepend(str(tools))
    spec = importlib.util.spec_from_file_location("queue_family_matrix", tools / "queue_family_matrix.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_register_is_fixed_and_covers_both_families(launcher):
    plan = launcher.matrix_plan(hash_assets=False)
    trees = plan["trees"]
    assert len(trees) == 8
    assert [t["tree_id"] for t in trees if t["status"] == "rendered_pilot"] == ["lpy_envy_00000", "lpy_ufo_00000"]
    assert sum(1 for t in trees if t["status"] == "to_render") == 6
    assert {t["family"] for t in trees} == {"envy", "ufo"}
    assert sum(1 for t in trees if t["family"] == "envy") == sum(1 for t in trees if t["family"] == "ufo") == 4
    assert plan["max_concurrent_gpus"] == 1
    # Six reserved renders plus the evaluation, in minutes.
    assert plan["maximum_gpu_minutes"] == 6 * launcher.RENDER_MINUTES_RESERVED + launcher.EVAL_MINUTES_RESERVED
    assert plan["gates"]["origin"].endswith("unchanged")
    assert plan["lights"] == ["source", "morning", "noon", "evening"]


def test_repeated_or_single_family_registers_are_refused(launcher):
    with pytest.raises(ValueError, match="unique"):
        launcher.matrix_plan(trees=("lpy_envy_00000", "lpy_ufo_00001"), hash_assets=False)
    with pytest.raises(ValueError, match="both families"):
        launcher.matrix_plan(trees=("lpy_envy_00003",), pilot_manifests={"lpy_envy_00000": "x"}, hash_assets=False)
    with pytest.raises(ValueError, match="Unknown tree family"):
        launcher.family_of("lpy_pear_00000")


def test_array_index_resolves_the_same_tree_the_job_script_will_render(launcher):
    plan = launcher.matrix_plan(hash_assets=False)
    # This mirrors the python snippet inside hpc/slurm/family_matrix_frozen.sbatch.
    to_render = [t for t in plan["trees"] if t["status"] == "to_render"]
    assert [t["tree_id"] for t in to_render] == list(launcher.MATRIX_TREES)
    assert to_render[5]["tree_id"] == "lpy_ufo_00003"
    with pytest.raises(IndexError):
        to_render[6]


def test_storage_preflight_refuses_a_full_share_and_an_unfinished_measurement(tmp_path, launcher):
    batch = tmp_path / "batch"
    batch.mkdir()
    report = launcher.storage_preflight(batch, 1_000_000_000_000, None)
    assert report["ok"] is True
    assert json.loads((batch / "storage_preflight.json").read_text())["ok"] is True

    full = tmp_path / "full"
    full.mkdir()
    with pytest.raises(RuntimeError, match="did not pass"):
        launcher.storage_preflight(full, 1_699_000_000_000, None)
    assert json.loads((full / "storage_preflight.json").read_text())["ok"] is False

    empty = tmp_path / "du.txt"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="still running"):
        launcher.storage_preflight(tmp_path / "e", None, empty)

    measured = tmp_path / "du_done.txt"
    measured.write_text("123456789\t/nfs/hpc/share/user\n", encoding="utf-8")
    (tmp_path / "m").mkdir()
    assert launcher.storage_preflight(tmp_path / "m", None, measured)["share_used_bytes"] == 123456789


def test_dirty_tree_is_refused_before_anything_is_frozen(tmp_path, monkeypatch, launcher):
    monkeypatch.setattr(launcher, "_git", lambda root, *args: b"tools/x.py\n" if args[0] == "diff" else b"abc\n")
    with pytest.raises(ValueError, match="Commit tracked changes"):
        launcher.queue_batch(tmp_path, "family-test", share_used_bytes=1)
    assert not (tmp_path / "artifacts").exists()


def test_batch_id_is_validated(tmp_path, launcher):
    with pytest.raises(ValueError, match="batch-id"):
        launcher.queue_batch(tmp_path, "../escape", share_used_bytes=1)
