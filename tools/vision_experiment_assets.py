"""Pin external render assets without copying or modifying their files."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path

import yaml


def _digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _environment_script() -> Path:
    return Path("/nfs/hpc/share") / os.environ["USER"] / "Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh"


def _recorded_path(base: Path, value: str) -> Path:
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"Asset path must not contain parent traversal: {value}")
    return path if path.is_absolute() else base / path


def _pin(path: Path, inventory: dict[str, str], expected: str | None = None) -> None:
    # Preserve symlink components: verification must detect a retargeted asset
    # symlink, not keep checking the previous target that rendering no longer uses.
    path = path.absolute()
    actual = _digest(path)
    if expected is not None and (not re.fullmatch(r"[0-9a-f]{64}", expected) or actual != expected):
        raise ValueError(f"Asset SHA-256 mismatch: {path}")
    previous = inventory.setdefault(str(path), actual)
    if previous != actual:
        raise ValueError(f"Asset changed during inventory: {path}")


def inventory_assets(root: Path) -> dict[str, str]:
    """Validate exporter/importer inventories and pin every listed local file.

    Includes the scene manifest, exported USD and textures, robot configuration
    and importer evidence, all robot payload files, and the external cluster
    launcher. Missing or already modified assets abort before submission.
    """
    root = root.absolute()
    inventory: dict[str, str] = {}
    scene_dir = root / "artifacts/blender_scene/orchard_two_trees_v1"
    manifest_path = scene_dir / "manifest.json"
    _pin(manifest_path, inventory)
    manifest = json.loads(manifest_path.read_text())
    artifacts = manifest["artifacts"]
    names = {item["path"] for item in artifacts}
    if not {"tree0.usdc", "tree1.usdc", "environment.usdc"} <= names:
        raise ValueError("Two-tree manifest must inventory both trees and the environment")
    for item in artifacts:
        if Path(item["path"]).is_absolute():
            raise ValueError("Scene artifact paths must be relative to the export directory")
        _pin(_recorded_path(scene_dir, item["path"]), inventory, item["sha256"])

    config_path = root / "source/isaaclab_pruning/isaaclab_pruning/config/robot/ur5e_pruner.yaml"
    _pin(config_path, inventory)
    config = yaml.safe_load(config_path.read_text())["usd"]
    evidence_path = _recorded_path(root, config["import_evidence"])
    _pin(evidence_path, inventory)
    evidence = json.loads(evidence_path.read_text())
    if (
        evidence.get("ok") is not True
        or evidence.get("status") != "complete"
        or evidence.get("imported") is not True
        or evidence.get("asset_id") != config["asset_id"]
    ):
        raise ValueError("Robot evidence must describe the configured successful import")
    output = evidence["output"]
    configured_usd = _recorded_path(root, config["relative_path"])
    if _recorded_path(root, output["root_layer"]).resolve() != configured_usd.resolve():
        raise ValueError("Robot evidence root layer differs from configured USD")
    files = output["files"]
    if not files:
        raise ValueError("Robot evidence has no output inventory")
    for item in files:
        _pin(_recorded_path(root, item["path"]), inventory, item["sha256"])
    if not any(_recorded_path(root, item["path"]).resolve() == configured_usd.resolve() for item in files):
        raise ValueError("Robot root layer is absent from the importer inventory")
    _pin(configured_usd, inventory)
    _pin(_environment_script(), inventory)
    verify_assets(inventory)
    return dict(sorted(inventory.items()))


def verify_assets(inventory: dict[str, str]) -> None:
    """Reject missing or changed assets, including changed symlink destinations."""
    if not inventory:
        raise ValueError("External asset inventory is empty")
    for name, expected in inventory.items():
        path = Path(name)
        if not path.is_absolute():
            raise ValueError(f"Pinned asset path must be absolute: {name}")
        if not re.fullmatch(r"[0-9a-f]{64}", expected) or _digest(path) != expected:
            raise ValueError(f"Pinned asset changed: {name}")
