from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_environment_lock_rejects_version_mismatch():
    check = _tool("check_isaac_render_stack").verify_versions
    lock = {"python_minor": "3.12", "packages": {"isaacsim": "6.0.0.1", "isaaclab": "3.0.0b2"}}
    check(lock, lock["packages"], "3.12")
    with pytest.raises(RuntimeError, match="Python"):
        check(lock, lock["packages"], "3.11")
    with pytest.raises(RuntimeError, match="isaacsim"):
        check(lock, {"isaacsim": "5.1.0", "isaaclab": "3.0.0b2"}, "3.12")


def _bundle(root):
    (root / "frames").mkdir()
    (root / "preflight.json").write_text(json.dumps({"ok": True}))
    (root / "report.json").write_text(json.dumps({"ok": True, "rendering_ok": True, "frame_count": 30}))
    (root / "frames.json").write_text(json.dumps({"frames": [{"index": i} for i in range(30)]}))
    pixels = np.indices((64, 64)).sum(axis=0).astype(np.uint8)
    for index in (0, 15, 29):
        for kind in ("overview", "wrist"):
            Image.fromarray(pixels).convert("RGB").save(root / "frames" / f"{kind}_{index:05d}.png")
        np.save(root / "frames" / f"depth_{index:05d}.npy", np.ones((64, 64)))


def test_render_gate_requires_images_not_only_exit_or_report(tmp_path):
    validate = _tool("validate_isaac_render").validate_bundle
    _bundle(tmp_path)
    assert validate(tmp_path)["frame_count"] == 30
    (tmp_path / "frames" / "overview_00029.png").unlink()
    with pytest.raises(FileNotFoundError):
        validate(tmp_path)


def test_render_gate_rejects_flat_render_and_invalid_depth(tmp_path):
    validate = _tool("validate_isaac_render").validate_bundle
    _bundle(tmp_path)
    path = tmp_path / "frames" / "wrist_00000.png"
    original = path.read_bytes()
    Image.new("RGB", (64, 64)).save(path)
    with pytest.raises(ValueError, match="flat"):
        validate(tmp_path)
    path.write_bytes(original)
    np.save(tmp_path / "frames" / "depth_00000.npy", np.full((64, 64), np.nan))
    with pytest.raises(ValueError, match="depth"):
        validate(tmp_path)


def test_render_gate_does_not_promote_failed_report(tmp_path):
    validate = _tool("validate_isaac_render").validate_bundle
    _bundle(tmp_path)
    (tmp_path / "report.json").write_text(json.dumps({"ok": False, "rendering_ok": True, "frame_count": 30}))
    with pytest.raises(ValueError, match="did not pass"):
        validate(tmp_path)
