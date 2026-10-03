"""Protect the closure-shadow projection: the sun-state split, flip counting, the decision rule and its bookkeeping,
the scene comparison, the in-memory cache stand-in and the output and cache refusals.

The ray-casting tests need open3d; the recording tests need artifacts/vision_robustness, and the golden test also
needs a geometry source (PRUNING_SHADOW_CACHE or PRUNING_ISAAC_PYTHON); each skips otherwise.
"""

from __future__ import annotations

import importlib.util
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
RUNS = TOOLS.parent / "artifacts/vision_robustness"


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("project_closure_shadow", TOOLS / "project_closure_shadow.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _box(tool, centre, half):
    return tool.jsm.box_corners((np.asarray(centre, float), np.eye(3), np.asarray(half, float))), tool.dal.CUBE_FACES


def test_classify_is_exclusive(tool):
    facing = np.array([True, True, True, True, False])
    other = np.array([False, True, False, True, False])
    jaw = np.array([False, False, True, True, True])
    state = tool.classify(facing, other, jaw)
    assert state["lit"].tolist() == [True, False, False, False, False]
    assert state["jaw_shadow"].tolist() == [False, False, True, False, False]
    assert state["other_shadow"].tolist() == [False, True, False, True, False]
    # every sun-facing point is in exactly one of the three states; facing away is in none
    total = state["lit"].astype(int) + state["jaw_shadow"] + state["other_shadow"]
    assert total.tolist() == [1, 1, 1, 1, 0]


def test_box_shadow_on_a_plane(tool):
    pytest.importorskip("open3d")
    plane = (
        np.array([[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]], float),
        np.array([[0, 1, 2], [0, 2, 3]]),
    )
    sun = np.array([1.0, 0.0, 1.0]) / np.sqrt(2)
    # A jaw-like box over x in [0, 1], |y| < 1, z in [1, 2]: a 45 degree sun shadows the plane for x in (-2, 0).
    jaw = tool.dal.build_scene({"jaw_right": _box(tool, [0.5, 0.0, 1.5], [0.5, 1.0, 0.5])})
    # An unrelated caster (a post) over x in [-6, -5], z in [1, 2] shadows x in (-8, -6).
    other = tool.dal.build_scene({"plane": plane, "post": _box(tool, [-5.5, 0.0, 1.5], [0.5, 1.0, 0.5])})
    points = np.array([[x, 0.0, 0.0] for x in (-1.9, -1.0, -0.1, 0.1, -3.0, -6.5)])
    up = np.tile([0.0, 0.0, 1.0], (len(points), 1))
    surface = {"P": points, "n": up, "view": -up}
    origins = tool.shadow_origins(surface)
    b_other, c_other = tool.blocked(other, origins, sun)
    b_jaw, c_jaw = tool.blocked(jaw, origins, sun)
    state = tool.classify(up @ sun > 0, b_other, b_jaw)
    assert state["jaw_shadow"].tolist() == [True, True, True, False, False, False]
    assert state["other_shadow"].tolist() == [False, False, False, False, False, True]
    assert state["lit"].tolist() == [False, False, False, True, True, False]
    assert c_jaw.tolist()[:4] == ["jaw_right"] * 3 + ["none"]
    assert c_other.tolist()[5] == "post"


def test_count_flips_attributes_the_shadowed_side(tool):
    on = np.array([True, True, True, True, False])
    previous = {
        "lit": np.array([True, False, True, False, True]),
        "caster": np.array(["none", "jaw_left", "none", "post", "none"], dtype=object),
    }
    current = {
        "lit": np.array([False, True, True, False, False]),
        "caster": np.array(["jaw_right", "none", "none", "post", "jaw_right"], dtype=object),
    }
    flips = tool.count_flips(on, previous, current)
    assert flips["to_shadow"].tolist() == [True, False, False, False, False]  # off-target element 4 is ignored
    assert flips["to_lit"].tolist() == [False, True, False, False, False]
    assert flips["casters"] == {"jaw_left": 1, "jaw_right": 1}
    assert flips["n_jaw"] == 2


def test_decision_rule_and_confusion_bookkeeping(tool):
    rows = [
        {"update": 72, "n_jaw_flips": 0, "recorded_corr": 0.999},
        {"update": 73, "n_jaw_flips": 34, "recorded_corr": 0.786},
        {"update": 74, "n_jaw_flips": 5, "recorded_corr": 0.86},
        {"update": 75, "n_jaw_flips": 4, "recorded_corr": 0.52},
        {"update": 76, "n_jaw_flips": 0, "recorded_corr": None},
    ]
    table = tool.confusion(rows, 0.85)
    assert (table["tp"], table["fp"], table["fn"], table["tn"]) == ([73], [74], [75], [72])
    assert (table["tp_n"], table["fp_n"], table["fn_n"], table["tn_n"]) == (1, 1, 1, 1)
    assert tool.confusion(rows, 0.90)["tp"] == [73, 74]
    assert tool.confusion(rows, 0.85, min_flips=1)["tp"] == [73, 75]
    assert tool.confusion(rows, 0.85, updates=range(73, 75))["tn"] == []
    cases = {"a": {"updates": rows}, "b": {"updates": rows[:2]}}
    assert tool.pooled(cases, ["a", "b"], 0.85) == {"tp": 2, "fp": 1, "fn": 1, "tn": 2}


def test_patch_distance_and_shifted_footprints(tool):
    us, vs = np.meshgrid(np.arange(0.0, 41.0), np.arange(0.0, 41.0))
    us, vs = us.ravel(), vs.ravel()
    p = np.array([20.0, 20.0])
    distance = tool.patch_distance(us, vs, p)
    assert distance[(us == 26) & (vs == 20)][0] == 0.0
    assert distance[(us == 29) & (vs == 20)][0] == 3.0
    assert distance[(us == 29) & (vs == 30)][0] == pytest.approx(5.0)
    on = np.ones(us.size, bool)
    shadow = us >= 29  # a shadow edge 3 px right of the patch
    summary = tool.window_summary(us, vs, on, shadow, p)
    assert summary["min_distance_px_patch_to_jaw_shadow"] == 3.0
    assert summary["shift_jaw_shadow_fraction_at_rint"] == 0.0
    assert summary["shift_jaw_shadow_fraction_max"] == pytest.approx(1 / 13)  # shifted 3 px right: one column
    assert summary["shift_n"] == 49 and summary["shift_n_with_jaw_shadow"] == 7


def test_scene_line_diff_allows_only_the_proxy_camera_and_spur_flag(tool):
    reference = [
        'def Xform "World"',
        "{",
        '    def Xform "Tree1"',
        "    {",
        '        def Xform "SelectedSpur"',
        "        {",
        "            bool physics:kinematicEnabled = 1",
        "            point3f[] points = [(0, 0, 0)]",
        "        }",
        "    }",
        '    def Xform "PruningJawProxy"',
        "    {",
        "        double3 xformOp:translate = (0, 0, 0)",
        "    }",
        '    def Camera "WristCamera"',
        "    {",
        "        matrix4d xformOp:transform = ( (1, 0, 0, 0) )",
        "    }",
        "}",
    ]
    run = list(reference)
    run[6] = "            bool physics:kinematicEnabled = 0"
    run[12] = "        double3 xformOp:translate = (1, 2, 3)"
    run[16] = "        matrix4d xformOp:transform = ( (0, 1, 0, 0) )"
    result = tool.scene_line_diff(reference, run)
    assert result["equivalent_geometry"]
    assert result["differing_line_kinds"] == {"spur_kinematic_flag": 2, "proxy": 2, "wrist_camera_transform": 2}
    run[7] = "            point3f[] points = [(0, 0, 1)]"
    result = tool.scene_line_diff(reference, run)
    assert not result["equivalent_geometry"]
    assert {entry["owners"][-1] for entry in result["disallowed_differences"]} == {"SelectedSpur"}


def _npz(arrays, level):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED, compresslevel=level) as archive:
        for name, array in arrays.items():
            member = io.BytesIO()
            np.lib.format.write_array(member, array, allow_pickle=False)
            archive.writestr(f"{name}.npy", member.getvalue())
    return buffer.getvalue()


def test_member_hashes_ignore_compression_and_mem_cache_feeds_committed_geometry(tool):
    arrays = {
        "static_vertices": np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], float),
        "static_faces": np.array([[0, 1, 2]], np.int64),
        "static_groups": np.array([0], np.int32),
        "robot_vertices": np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], float),
        "robot_faces": np.array([[0, 1, 2]], np.int64),
        "robot_groups": np.array([1], np.int32),
    }
    fast, small = _npz(arrays, 1), _npz(arrays, 9)
    assert tool.npz_member_sha256(fast) == tool.npz_member_sha256(small)
    meta = {
        "sun_direction_to_sun_w": [0.0, 0.0, 1.0],
        "groups": [
            {"index": 0, "kind": "static", "name": "tree1_SPUR", "is_tree": True},
            {"index": 1, "kind": "robot", "name": "wrist_3_link", "link": "ur5e__wrist_3_link"},
        ],
        "link_world_default": {"ur5e__base_link_inertia": np.eye(4).tolist()},
    }
    geometry = tool.dal.Geometry(tool.MemCache(small, json.dumps(meta)), report={})
    assert geometry.tree_groups == ["tree1_SPUR"]
    vertices, faces = geometry.static["tree1_SPUR"]
    assert len(vertices) == 3 and faces.tolist() == [[0, 1, 2]]  # unused vertex dropped, as from a cache file
    assert geometry.robot["wrist_3_link"][0] == "ur5e__wrist_3_link"


def test_output_and_cache_refusals(tool, tmp_path):
    for bad in (tool.REPO / "out.json", tool.REPO / "tools" / "x.json", tmp_path / "artifacts" / "out.json"):
        with pytest.raises(ValueError):
            tool.refuse_inside(bad, "an output")
    assert tool.refuse_inside(tmp_path / "out.json", "an output") == (tmp_path / "out.json").resolve()
    with pytest.raises(ValueError):
        tool.GeometrySource(cache_root=tool.REPO / "cache")
    existing = tmp_path / "existing.json"
    existing.write_text("keep\n")
    for argv in (
        ["--output", str(existing), "--isaac-python", "python"],
        ["--output", str(tool.REPO / "never.json"), "--isaac-python", "python"],
        ["--output", str(tmp_path / "new.json"), "--cache-root", str(tool.REPO / "cache")],
        ["--output", str(tmp_path / "new.json"), "--isaac-python", ""],
    ):
        with pytest.raises(SystemExit) as stop:
            tool.main(argv)
        assert stop.value.code != 0
    assert existing.read_text() == "keep\n"
    assert not (tool.REPO / "never.json").exists() and not (tmp_path / "new.json").exists()


def test_no_cache_and_no_extractor_is_a_gate_error(tool, tmp_path):
    scene = tmp_path / "scene.usda"
    scene.write_text("#usda 1.0\n")
    (tmp_path / "caches" / "other").mkdir(parents=True)
    (tmp_path / "caches" / "other" / "shadow_casters.json").write_text(json.dumps({"scene_sha256": "0" * 64}))
    with pytest.raises(tool.GateError):
        tool.GeometrySource(cache_root=tmp_path / "caches").load(scene)


def _needs_runs(*runs):
    if not all((RUNS / run / "frames.json").exists() for run in runs):
        pytest.skip("needs the recorded runs under artifacts/vision_robustness")


def test_recorded_scenes_differ_only_in_the_proxy_camera_and_spur(tool):
    reference, no_shadow = tool.REFERENCE_RUNS["eve14944"], tool.CASES["p_eve14944_jsb_r1"][0]
    _needs_runs(reference, no_shadow)
    result = tool.scene_line_diff(
        (RUNS / reference / "scene.usda").read_text().splitlines(),
        (RUNS / no_shadow / "scene.usda").read_text().splitlines(),
    )
    assert result["equivalent_geometry"]
    assert set(result["differing_line_kinds"]) == {"proxy", "wrist_camera_transform", "spur_kinematic_flag"}


def test_recorded_closure_schedule(tool):
    _needs_runs(tool.SCHEDULE_SOURCE)
    frames = json.loads((RUNS / tool.SCHEDULE_SOURCE / "frames.json").read_text())["frames"]
    schedule = tool.schedule_from(frames)
    assert [round(6 * schedule[k], 3) for k in range(71, 79)] == [0, 0, 1, 2, 3, 4, 5, 6]


def test_golden_morning_15004_closure_reproduces_the_committed_evidence(tool):
    _needs_runs(tool.CASES["repro_mor15004"][0])
    pytest.importorskip("open3d")
    cache, python = os.environ.get("PRUNING_SHADOW_CACHE"), os.environ.get("PRUNING_ISAAC_PYTHON")
    if not (cache and Path(cache).is_dir()) and not python:
        pytest.skip("needs PRUNING_SHADOW_CACHE (extracted caches) or PRUNING_ISAAC_PYTHON (pxr) for the geometry")
    source = tool.GeometrySource(python, cache if cache and Path(cache).is_dir() else None)
    case = tool.run_case("repro_mor15004", RUNS, source, {})
    rows = {row["update"]: row for row in case["updates"]}
    assert [(rows[k]["n_lit_to_shadow"], rows[k]["n_jaw_flips"]) for k in (73, 74, 75)] == [(13, 13), (11, 11), (7, 7)]
    assert rows[75]["masked_ncc"]["dilated"]["ncc"] == pytest.approx(0.967, abs=1e-3)
    assert all(row["split_equals_committed_full_scene"] for row in case["updates"])
