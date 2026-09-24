"""The replay planner keeps missing frames in the denominator."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
PIL = pytest.importorskip("PIL")
TOOLS = Path(__file__).resolve().parents[1] / "tools"


def test_missing_and_blank_frames_are_listed_not_dropped(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("prepare_replay_evaluation", TOOLS / "prepare_replay_evaluation.py")
    prepare = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prepare)
    frames = []
    for i, blank in enumerate((False, True, False)):
        depth, mask, rgb, isaac = (tmp_path / f"{i}_{n}" for n in ("d.npy", "m.png", "rgb.png", "isaac.npy"))
        np.save(depth, np.full((8, 12), 0.3, dtype=np.float32))
        np.save(isaac, np.full((8, 12), 0.3, dtype=np.float32))
        cv2.imwrite(str(mask), np.full((8, 12), 255, dtype=np.uint8))
        image = (
            np.zeros((8, 12, 3), dtype=np.uint8)
            if blank
            else np.random.default_rng(i).integers(0, 255, (8, 12, 3), dtype=np.uint8)
        )
        cv2.imwrite(str(rgb), image)
        frames.append(
            {
                "rgb": str(rgb),
                "depth": str(depth),
                "mask": str(mask),
                "isaac_depth": str(isaac),
                "width": 12,
                "height": 8,
                "view_id": f"f{i}",
                "condition": "isaac_pose/palm/source",
            }
        )
    frames[2]["rgb"] = str(tmp_path / "absent.png")
    manifest = tmp_path / "render_manifest.json"
    manifest.write_text(json.dumps({"ok": True, "frames": frames, "runs": [], "lighting": {}}))
    plan = prepare.build(manifest, tmp_path / "plan.json", expected_frames=4)
    assert plan["registered_frames"] == 4 and plan["scored_frames"] == 1
    reasons = {m["view_id"]: m["reason"] for m in plan["missing"]}
    assert "Blank" in reasons["f1"] and "f2" in reasons and "*" in reasons
    with pytest.raises(FileExistsError):
        prepare.build(manifest, tmp_path / "plan.json")
