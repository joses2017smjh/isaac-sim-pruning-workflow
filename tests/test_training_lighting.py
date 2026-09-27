"""The rendered-lighting sampler is deterministic, stays in its ranges and never lands near an evaluation preset."""

from __future__ import annotations

import csv
import importlib.util
import math
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture
def tl(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    spec = importlib.util.spec_from_file_location("training_lighting", TOOLS / "training_lighting.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_draws_are_deterministic_per_frame_and_differ_between_frames(tl):
    rotation = [1.5395, 0.0, tl.ROTATION_REF_RZ]
    a = tl.sample("lpy_envy_00001", "box", "shot01", "l", rotation)
    b = tl.sample("lpy_envy_00001", "box", "shot01", "l", rotation)
    c = tl.sample("lpy_envy_00001", "box", "shot01", "r", rotation)
    assert a == b and a["config"] != c["config"]
    assert tl.lighting_label(a).startswith("rl1_") and len(tl.lighting_label(a)) == 20


def test_every_draw_respects_its_ranges_and_the_holdout(tl):
    regimes = {"low": 0, "high": 0}
    for i in range(400):
        yaw = (i % 5 - 2) * 0.4
        draw = tl.sample(
            f"lpy_envy_{i % 80:05d}",
            f"box_cam{i % 4 + 1}",
            f"shot{i % 6 + 1:02d}",
            "lr"[i % 2],
            [1.54, 0.0, tl.ROTATION_REF_RZ + yaw],
        )
        regimes[draw["regime"]] += 1
        spec, sun = tl.REGIMES[draw["regime"]], draw["config"]["sun"]
        assert spec["elevation_deg"][0] <= sun["elevation_deg"] <= spec["elevation_deg"][1]
        assert tl.SUN_ENERGY[0] <= sun["energy"] <= tl.SUN_ENERGY[1]
        assert spec["world_strength"][0] <= draw["config"]["world_strength"] <= spec["world_strength"][1]
        assert spec["cct_k"][0] <= draw["sun_cct_k"] <= spec["cct_k"][1]
        assert max(sun["color"]) == pytest.approx(1.0)
        assert min(draw["holdout_min_distance_deg"].values()) > tl.HOLDOUT_RADIUS_DEG
        # Recompute independently: no draw within 15 degrees of any preset sun, either frame.
        s = tl.unit(sun["azimuth_deg"], sun["elevation_deg"])
        for paz, pel in tl.PRESET_SUNS.values():
            for offset in (0.0, draw["pose_yaw_offset_deg"]):
                assert tl.angle_deg(s, tl.unit(paz + offset, pel)) > 15.0
    assert 0.5 < regimes["low"] / 400 < 0.7


def test_colour_temperature_is_warm_below_neutral_and_rejects_out_of_range(tl):
    warm, neutral = tl.cct_to_linear_rgb(2500), tl.cct_to_linear_rgb(6500)
    assert warm[0] == pytest.approx(1.0) and warm[2] < 0.4
    assert min(neutral) > 0.85
    with pytest.raises(ValueError):
        tl.planck_xy(1000)
    assert tl.yaw_offset_deg([0, 0, tl.ROTATION_REF_RZ + math.radians(50)]) == pytest.approx(50.0)


def test_agreement_requires_matching_geometry(tl):
    depth = np.full((10, 10), 1e10)
    depth[2:8, 2:8] = 2.0
    mask = depth < 1e6
    good = tl.agreement(depth + 0.0005 * mask, mask, depth, mask)
    assert good["ok"] and good["mask_iou"] == 1.0
    shifted = np.roll(mask, 1, axis=1)
    assert not tl.agreement(depth, shifted, depth, mask)["ok"]
    far = depth.copy()
    far[2:8, 2:8] = 2.01
    assert not tl.agreement(far, mask, depth, mask)["ok"]
    with pytest.raises(ValueError):
        tl.agreement(depth[:5], mask[:5], depth, mask)


def test_surviving_rows_keep_complete_frames_and_group_by_tree(tl, tmp_path):
    files = {}
    for name in ("a_rgb", "a_depth", "a_mask", "b_rgb", "b_mask"):
        files[name] = tmp_path / name
        files[name].write_bytes(b"x")
    manifest = tmp_path / "train.csv"
    with manifest.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["bark", "tree", "set_id", "shot", "view", "rgb_path", "depth_path", "mask_path"]
        )
        writer.writeheader()
        writer.writerow(
            {
                "bark": "b2",
                "tree": "t1",
                "set_id": "box",
                "shot": "shot01",
                "view": "l",
                "rgb_path": files["a_rgb"],
                "depth_path": files["a_depth"],
                "mask_path": files["a_mask"],
            }
        )
        writer.writerow(
            {
                "bark": "b2",
                "tree": "t2",
                "set_id": "box",
                "shot": "shot01",
                "view": "c",
                "rgb_path": files["b_rgb"],
                "depth_path": tmp_path / "missing",
                "mask_path": files["b_mask"],
            }
        )
    rows = tl.surviving_rows(manifest)
    assert [r["tree"] for r in rows] == ["t1"]
    assert list(tl.frames_by_tree(rows)) == ["t1"]
    assert tl.annotation_path("/cv", rows[0]).as_posix().endswith("ann/b2/t1/box/t1_shot01_l.json")
    assert tl.annotation_path("/cv", {**rows[0], "view": "c"}).as_posix().endswith("t1_shot01.json")
