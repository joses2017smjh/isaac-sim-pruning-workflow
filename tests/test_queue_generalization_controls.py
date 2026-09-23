"""Protect the controls launcher: fixed register, all conditions, sized array, honest preflight."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def launcher(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "queue_generalization_controls", TOOLS / "queue_generalization_controls.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_register_is_the_matrix_register_and_every_condition_is_listed(launcher):
    plan = launcher.controls_plan(hash_assets=False)
    trees = [t["tree_id"] for t in plan["trees"]]
    assert len(trees) == 8 and len(set(trees)) == 8
    assert {t["family"] for t in plan["trees"]} == {"envy", "ufo"}
    assert all(t["status"] == "to_render" for t in plan["trees"])
    assert len(plan["conditions"]) == 10
    assert plan["frames_per_tree"] == 62 and plan["photometric_frames_per_tree"] == 24
    assert plan["registered_frames"] == 8 * (62 + 24)
    assert plan["maximum_gpu_minutes"] == 8 * 15 + 60
    assert plan["max_concurrent_gpus"] == 1
    assert plan["luma_factor"] == 2.6 and plan["isaac_pitch_deg"] == 39.7
    assert plan["photometric_parameters"]["clahe"]["clip_limit"] == 2.0
    assert plan["gates"]["target_p95_absolute_m"] == 0.02
    assert plan["protocol"].endswith("GENERALIZATION_CONTROLS_2026-09-23.md")
    assert len(plan["relative_head"]["checkpoint_sha256"]) == 64
    assert {s["frames"] for s in plan["relative_head"]["extra_frames"].values()} == {600, 192}


def test_repeated_or_single_family_registers_are_refused(launcher):
    with pytest.raises(ValueError):
        launcher.controls_plan(trees=("lpy_envy_00000", "lpy_envy_00000", "lpy_ufo_00000"), hash_assets=False)
    with pytest.raises(ValueError):
        launcher.controls_plan(trees=("lpy_envy_00000", "lpy_envy_00003"), hash_assets=False)


def test_array_index_resolves_the_tree_and_the_planner_expects_the_same_folder(launcher, monkeypatch):
    plan = launcher.controls_plan(hash_assets=False)
    to_render = [t for t in plan["trees"] if t["status"] == "to_render"]
    assert f"0-{len(to_render) - 1}%1" == "0-7%1"
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location(
        "prepare_controls_evaluation", TOOLS / "prepare_controls_evaluation.py"
    )
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    for index, tree in enumerate(to_render):
        assert prepare.manifest_path(Path("/b"), plan, tree["tree_id"]) == Path(
            f"/b/controls_{index}_{tree['tree_id']}/render_manifest.json"
        )


def test_storage_preflight_uses_the_larger_controls_estimate(tmp_path, launcher):
    from queue_family_matrix import storage_preflight

    assert launcher.ESTIMATED_OUTPUT_BYTES > 1_000 * 1024 * 1024
    report = storage_preflight(
        tmp_path, 1_000_000_000_000, None, estimated_output_bytes=launcher.ESTIMATED_OUTPUT_BYTES
    )
    assert report["ok"] and report["estimated_output_bytes"] == launcher.ESTIMATED_OUTPUT_BYTES
    written = json.loads((tmp_path / "storage_preflight.json").read_text())
    assert written["estimated_output_bytes"] == launcher.ESTIMATED_OUTPUT_BYTES
    with pytest.raises(RuntimeError):
        storage_preflight(tmp_path, 1_599_500_000_000, None, estimated_output_bytes=launcher.ESTIMATED_OUTPUT_BYTES)


def test_dirty_tree_is_refused_before_anything_is_frozen(tmp_path, monkeypatch, launcher):
    monkeypatch.setattr(launcher, "_git", lambda root, *args: b"tools/x.py\n")
    with pytest.raises(ValueError):
        launcher.queue_batch(tmp_path, "controls-test", share_used_bytes=1)
    assert not (tmp_path / "artifacts").exists()


def test_batch_id_is_validated(tmp_path, launcher):
    with pytest.raises(ValueError):
        launcher.queue_batch(tmp_path, "../escape", share_used_bytes=1)
