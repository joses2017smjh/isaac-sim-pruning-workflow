"""Protect the appearance-loss diagnosis: the tracker's own correlation, the masks and nulls, the cache refusal.

The shadow model test needs open3d and the golden values need the recorded runs
plus an extracted geometry cache (PRUNING_SHADOW_CACHE); both skip otherwise.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
RUNS = TOOLS.parent / "artifacts/vision_robustness"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def diagnose():
    return _load("diagnose_appearance_loss")


@pytest.fixture(scope="module")
def extract():
    return _load("extract_shadow_casters")


def test_ncc_is_the_tracker_appearance_correlation(diagnose):
    rng = np.random.default_rng(3)
    previous = rng.integers(0, 256, (40, 50), dtype=np.uint8)
    shifted = np.roll(previous, (1, 1), axis=(0, 1)) + rng.normal(0, 25, previous.shape)
    current = np.clip(shifted, 0, 255).astype(np.uint8)
    tracker = diagnose.VisualServoTracker(diagnose.VisualServoConfig())
    tracker._previous_gray, tracker._pixel = previous, np.array([20.3, 17.6], np.float32)
    moved = np.array([21.45, 18.4], np.float32)
    expected = tracker._appearance(current, moved)
    a = diagnose.patch(previous.astype(np.float32), tracker._pixel)
    b = diagnose.patch(current.astype(np.float32), moved)
    assert 0.3 < expected < 0.99
    assert diagnose.ncc(a, b) == pytest.approx(expected, abs=1e-6)


def test_masked_ncc_drops_the_mask_and_its_dilation(diagnose):
    rng = np.random.default_rng(4)
    a = rng.normal(100, 20, (13, 13))
    b = 2 * a + 5
    flip = np.zeros((13, 13), bool)
    flip[6, 6] = True
    b[6, 6] = 255.0
    dilated = diagnose.dilate(flip)
    assert dilated.sum() == 9 and dilated[5:8, 5:8].all()
    assert diagnose.ncc(a, b) < 0.99
    assert diagnose.ncc(a, b, ~flip) == pytest.approx(1.0, abs=1e-12)
    assert diagnose.ncc(a, b, ~dilated) == pytest.approx(1.0, abs=1e-12)
    assert np.isnan(diagnose.ncc(a, b, np.zeros((13, 13), bool)))


def test_shift_null_bookkeeping(diagnose):
    corner = np.zeros((13, 13), bool)
    corner[0, 12] = True
    assert diagnose.shift_mask(corner, 1, 0, wrap=True)[0, 0]
    assert not diagnose.shift_mask(corner, 1, 0, wrap=False).any()
    assert diagnose.shift_mask(corner, -2, 3, wrap=False)[3, 10]

    rng = np.random.default_rng(5)
    a = rng.normal(100, 20, (13, 13))
    b = 2 * a + 5
    flip = np.zeros((13, 13), bool)
    flip[5:8, 5:8] = True
    b[flip] = 0.0
    result = diagnose.masked_ncc(a, b, flip, draws=200)
    assert result["undilated"]["n_kept"] == 160 and result["dilated"]["n_kept"] == 144
    assert result["dilated"]["ncc"] == pytest.approx(1.0, abs=1e-12)
    assert result["dilated"]["random_fraction_ge_true"] < 0.05
    shifts = result["dilated"]["shift_null"]
    assert {key: value["n_shifts"] for key, value in shifts.items()} == {
        "ring1_wrap": 8,
        "ring_ge2_wrap": 160,
        "ring1_zero_fill": 8,
        "ring_ge2_zero_fill": 160,
    }
    # Every ring-1 shift of the 5 x 5 dilated mask still covers the 3 x 3 flip; ring >= 2 shifts do not.
    assert shifts["ring1_wrap"]["max"] == pytest.approx(1.0, abs=1e-12)
    assert shifts["ring_ge2_wrap"]["max"] < 0.99 and shifts["ring_ge2_wrap"]["n_ge_true"] == 0


def test_cache_dir_refused_inside_the_repository_or_artifacts(extract, tmp_path, monkeypatch):
    for bad in (extract.REPO / "cache", extract.REPO, tmp_path / "artifacts" / "cache"):
        with pytest.raises(ValueError):
            extract.refuse_cache_dir(bad)
    assert extract.refuse_cache_dir(tmp_path / "cache") == (tmp_path / "cache").resolve()
    monkeypatch.setattr(sys, "argv", ["x", "--run-dir", str(tmp_path), "--cache-dir", str(extract.REPO / "c")])
    with pytest.raises(SystemExit) as stop:
        extract.main()
    assert stop.value.code != 0
    assert not (extract.REPO / "c").exists()


def test_cube_shadow_edge_on_a_plane(diagnose):
    pytest.importorskip("open3d")
    plane = (
        np.array([[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]], float),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    corners = np.array([[x, y, z] for x in (0, 1) for y in (-1, 1) for z in (1, 2)], float)
    scene = diagnose.build_scene({"plane": plane, "cube": (corners, diagnose.CUBE_FACES)})
    sun = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    # A 45 degree sun through the cube x in [0, 1], z in [1, 2] shadows the plane for x in (-2, 0), |y| < 1.
    points = np.array([[x, y, 0.0] for x, y in ((-1.9, 0), (-1.0, 0.5), (-0.1, -0.9), (0.1, 0), (-2.1, 0), (-1, 1.1))])
    up = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    visible, caster = diagnose.sun_visibility(scene, points, up, -up, sun)
    assert visible.tolist() == [False, False, False, True, True, True]
    assert caster.tolist() == ["cube"] * 3 + ["none"] * 3
    visible, _ = diagnose.sun_visibility(scene, points, -up, up, sun)
    assert not visible.any()  # facing away from the sun is never lit


GOLDEN = [
    (
        "tree1-listed-evening-20260926/run_01_evening_tree1_v14944_baseline",
        67,
        {"recorded_corr": 0.10267166, "to_lit": 70, "to_shadow": 0, "caster": "jaw_right", "kept": 62, "ncc": 0.921}
        # Shadow-to-lit counts at the four origin offsets, recomputed by an independent verifier's own code.
        | {"undilated": (99, 0.574), "offset_counts": [42, 55, 70, 82]},
    ),
    (
        "tree1-listed-evening-20260926/run_02_evening_tree1_v15004_baseline",
        67,
        {"to_lit": 21, "to_shadow": 0, "caster": "jaw_left", "kept": 124, "ncc": 0.958},
    ),
    (
        "tree1-listed-morning-20260926/run_02_morning_tree1_v15004_baseline",
        75,
        {"to_lit": 0, "to_shadow": 7, "caster": "jaw_right", "kept": 152, "ncc": 0.967},
    ),
]


@pytest.mark.parametrize(("run", "update", "expected"), GOLDEN)
def test_golden_values(diagnose, run, update, expected):
    pytest.importorskip("open3d")
    cache = os.environ.get("PRUNING_SHADOW_CACHE")
    if not cache or not Path(cache).is_dir() or not (RUNS / run / "frames.json").exists():
        pytest.skip("needs the recorded run and PRUNING_SHADOW_CACHE pointing to an extracted cache root")
    result = diagnose.analyse_run(RUNS / run, (update, update), Path(cache))
    (row,) = result["updates"]
    assert result["stop_update"] == update
    if "recorded_corr" in expected:
        assert row["recorded_corr"] == pytest.approx(expected["recorded_corr"], abs=1e-8)
    assert (row["n_shadow_to_lit"], row["n_lit_to_shadow"]) == (expected["to_lit"], expected["to_shadow"])
    assert row["flip_casters"] == {expected["caster"]: expected["to_lit"] + expected["to_shadow"]}
    dilated = row["masked_ncc"]["dilated"]
    assert dilated["n_kept"] == expected["kept"]
    assert dilated["ncc"] == pytest.approx(expected["ncc"], abs=1e-3)
    if "undilated" in expected:
        undilated = row["masked_ncc"]["undilated"]
        assert undilated["n_kept"] == expected["undilated"][0]
        assert undilated["ncc"] == pytest.approx(expected["undilated"][1], abs=1e-3)
    sensitivity = result["shadow_offset_sensitivity_at_stop"]
    default = next(
        s
        for s in sensitivity
        if (s["back_m"], s["along_normal_m"]) == (diagnose.SHADOW_BACK_M, diagnose.SHADOW_NORMAL_M)
    )
    assert (default["n_shadow_to_lit"], default["n_lit_to_shadow"]) == (expected["to_lit"], expected["to_shadow"])
    assert default["masked_ncc_dilated"] == pytest.approx(dilated["ncc"], abs=1e-12)
    assert all(set(s["casters"]) <= {expected["caster"]} for s in sensitivity)
    if "offset_counts" in expected:
        assert [s["n_shadow_to_lit"] for s in sensitivity] == expected["offset_counts"]
