"""External texture and robot payload drift must invalidate paired trials."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pytest
import yaml


@pytest.fixture
def assets():
    path = Path(__file__).resolve().parents[1] / "tools/vision_experiment_assets.py"
    spec = importlib.util.spec_from_file_location("vision_experiment_assets", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def asset_tree(tmp_path, monkeypatch, assets):
    root = tmp_path / "repo"
    scene = root / "artifacts/blender_scene/orchard_two_trees_v1"
    scene.mkdir(parents=True)
    entries = []
    for name in ["tree0.usdc", "tree1.usdc", "environment.usdc", *[f"textures/texture{i}.png" for i in range(6)]]:
        path = scene / name
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(name.encode())
        entries.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    (scene / "manifest.json").write_text(json.dumps({"artifacts": entries}))
    robot_entries = []
    for name in ("artifacts/robot/root.usda", "artifacts/robot/payload.usda"):
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        robot_entries.append({"path": name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    evidence = root / "docs/evidence/import.json"
    evidence.parent.mkdir(parents=True)
    evidence.write_text(
        json.dumps(
            {
                "ok": True,
                "status": "complete",
                "imported": True,
                "asset_id": "robot",
                "output": {"root_layer": robot_entries[0]["path"], "files": robot_entries},
            }
        )
    )
    config = root / "source/isaaclab_pruning/isaaclab_pruning/config/robot/ur5e_pruner.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(
        yaml.safe_dump(
            {
                "usd": {
                    "relative_path": robot_entries[0]["path"],
                    "import_evidence": "docs/evidence/import.json",
                    "asset_id": "robot",
                }
            }
        )
    )
    env_script = tmp_path / "_env.sh"
    env_script.write_text("original cluster environment")
    monkeypatch.setattr(assets, "_environment_script", lambda: env_script)
    return root, scene, env_script


def test_inventory_covers_textures_robot_payloads_and_cluster_launcher(asset_tree, assets):
    root, scene, env_script = asset_tree
    inventory = assets.inventory_assets(root)
    assert len(inventory) == 15
    assert str(scene / "manifest.json") in inventory
    assert all(str(scene / f"textures/texture{i}.png") in inventory for i in range(6))
    assert str(root / "artifacts/robot/payload.usda") in inventory
    assert str(env_script) in inventory
    assets.verify_assets(inventory)


@pytest.mark.parametrize("relative", ["textures/texture0.png", "../robot/payload.usda"])
def test_changed_assets_are_rejected_at_submission(asset_tree, assets, relative):
    root, scene, _ = asset_tree
    path = scene / relative if relative.startswith("textures") else root / "artifacts/robot/payload.usda"
    path.write_text("changed after export/import")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        assets.inventory_assets(root)


@pytest.mark.parametrize("kind", ["texture", "payload", "manifest", "environment"])
def test_external_drift_after_inventory_is_rejected(asset_tree, assets, kind):
    root, scene, env_script = asset_tree
    inventory = assets.inventory_assets(root)
    paths = {
        "texture": scene / "textures/texture0.png",
        "payload": root / "artifacts/robot/payload.usda",
        "manifest": scene / "manifest.json",
        "environment": env_script,
    }
    paths[kind].write_text("changed while pending")
    with pytest.raises(ValueError, match="Pinned asset changed"):
        assets.verify_assets(inventory)


def test_retargeted_symlink_is_checked_at_the_path_rendering_uses(tmp_path, assets):
    original, replacement = tmp_path / "original", tmp_path / "replacement"
    original.write_text("original")
    replacement.write_text("changed")
    link = tmp_path / "asset"
    link.symlink_to(original)
    inventory = {str(link): hashlib.sha256(original.read_bytes()).hexdigest()}
    assets.verify_assets(inventory)
    link.unlink()
    link.symlink_to(replacement)
    with pytest.raises(ValueError, match="Pinned asset changed"):
        assets.verify_assets(inventory)
