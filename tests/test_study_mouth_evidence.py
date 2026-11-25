"""tools/study_mouth_evidence.py on a hand-built scene: truth, the three channels and rigid displacement."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
IDENTITY = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
CAMERA_P = [0.0, -0.14, 0.045]


@pytest.fixture(scope="module")
def study():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("study_mouth_evidence", TOOLS / "study_mouth_evidence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quad(a, b, c, d):
    return np.array([a, b, c, d], float), np.array([[0, 1, 2], [0, 2, 3]])


def _join(quads):
    vertices, faces, offset = [], [], 0
    for V, F in quads:
        vertices.append(V)
        faces.append(F + offset)
        offset += len(V)
    return np.concatenate(vertices), np.concatenate(faces)


def _scene(study):
    """The tool at the origin, jaw frame = world: closing axis x, jaw along y, tool axis z, mouth at z = 70 mm.

    The target is three patches: one across the gap at the jaw tips (z 75-90 mm), one facing the ToF sensors
    inside the mouth box, and one facing the camera at the mouth centre.
    """
    across_gap = _quad([0, -0.02, 0.075], [0, 0.02, 0.075], [0, 0.02, 0.09], [0, -0.02, 0.09])
    facing_tof = _quad([0.008, -0.01, 0.0801], [0.016, -0.01, 0.0801], [0.016, 0.01, 0.0801], [0.008, 0.01, 0.0801])
    facing_camera = _quad([-0.01, 0.005, 0.06], [0.01, 0.005, 0.06], [0.01, 0.005, 0.08], [-0.01, 0.005, 0.08])
    far = _quad([1, 1, 1], [1.1, 1, 1], [1.1, 1.1, 1], [1, 1.1, 1])
    target = _join([across_gap, facing_tof, facing_camera])
    meshes = [("tree0_BRANCH", *far), ("TARGET", *target)]
    return study.Scene(meshes, index_cls=study.kmp.NumpyMeshIndex)


def _rig(study):
    return study.Rig(IDENTITY, 0.0, CAMERA_P, np.eye(3))


def test_ray_box_hits_misses_and_starts_inside(study):
    lo, hi = np.array([-1.0, -1.0, -1.0]), np.array([1.0, 1.0, 1.0])
    assert study.ray_box(np.array([-3.0, 0, 0]), np.array([1.0, 0, 0]), lo, hi) == pytest.approx((2.0, 4.0))
    assert study.ray_box(np.array([-3.0, 2, 0]), np.array([1.0, 0, 0]), lo, hi) is None
    assert study.ray_box(np.array([0.0, 0, 0]), np.array([0, 0, 1.0]), lo, hi) == pytest.approx((0.0, 1.0))


def test_truth_separates_between_the_jaws_from_at_the_mouth_centre(study):
    scene, rig = _scene(study), _rig(study)
    at_tips = study.beams(rig, study.jsm.JAW_OPEN_GAP_M, scene)
    assert at_tips["truth_between_jaws"] and not at_tips["truth_at_mouth_centre"]
    assert at_tips["truth_target_beams"] == 3 and at_tips["beams_blocked"] == 3
    assert not at_tips["beam_centre_blocked"] and at_tips["beam_hits"] == {"TARGET": 3}
    advanced = study.beams(rig.shifted([0, 0, 0.010]), study.jsm.JAW_OPEN_GAP_M, scene)
    assert advanced["truth_at_mouth_centre"] and advanced["beam_centre_blocked"]
    withdrawn = study.beams(rig.shifted([0, 0, -0.020]), study.jsm.JAW_OPEN_GAP_M, scene)
    assert not withdrawn["truth_between_jaws"] and withdrawn["beams_blocked"] == 0


def test_tof_channel_counts_first_returns_inside_the_mouth_box(study):
    scene, rig = _scene(study), _rig(study)
    reading = study.tof_channel(rig, scene)
    assert reading["mouth_zones"] > 0
    assert reading["returns_in_mouth"] >= 1 and set(reading["returns_in_mouth_by_label"]) == {"TARGET"}
    away = study.tof_channel(rig.shifted([0, 0, -0.060]), scene)
    assert away["returns_in_mouth"] == 0


def test_camera_channel_sees_the_surface_at_the_mouth_centre(study):
    scene, rig = _scene(study), _rig(study)
    reading = study.camera_channel(rig, scene)
    assert reading["surface_in_mouth"] and reading["surface_label"] == "TARGET"
    assert not study.camera_channel(rig.shifted([0.05, 0, 0]), scene)["surface_in_mouth"]


def test_a_shift_moves_jaws_sensors_and_camera_together(study):
    rig = _rig(study)
    moved = rig.shifted([0.01, 0.02, 0.03])
    delta = np.array([0.01, 0.02, 0.03])
    assert np.allclose(moved.mouth - rig.mouth, delta)
    assert np.allclose(moved.tof_origin("tof0") - rig.tof_origin("tof0"), delta)
    assert np.allclose(moved.cam_p - rig.cam_p, delta)
    # The mouth is ahead of the ToF sensors along the tool axis, and off each sensor's axis.
    ahead = (rig.mouth - rig.tof_origin("tof0"))[2]
    assert ahead == pytest.approx(0.1601525 + 0.070 - 0.14444246761)


def test_build_scene_splits_the_target_component_from_the_other_spurs(study):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]], float)
    export = {"v0": vertices, "t0": np.array([[0, 1, 2], [3, 4, 5]])}
    report = {
        "blender_scene": {
            "target_tree_index": 0,
            "translation_w_m": [0.0, 0.0, 0.0],
            "yaw_degrees": 0.0,
            "target": {"source_component": {"source_vertex_indices": [3, 4, 5]}},
        }
    }
    scene = study.build_scene(export, ["tree0:/root/tree0_SPUR/Shape"], report, study.kmp.NumpyMeshIndex)
    assert scene.labels == ["tree0_SPUR(other spurs)", "TARGET"] and scene.target_triangles == 1


def test_summary_counts_each_channel_against_both_truths(study):
    def row(axis, offset, between, centre, tof, blocked, centre_blocked, camera, hits):
        return {
            "axis": axis,
            "offset_mm": offset,
            "truth_between_jaws": between,
            "truth_at_mouth_centre": centre,
            "tof": {"returns_in_mouth": tof},
            "beams_blocked": blocked,
            "beam_centre_blocked": centre_blocked,
            "camera": {"surface_in_mouth": camera},
            "beam_hits": hits,
        }

    rows = [
        row("none", 0.0, True, False, 2, 3, False, True, {"TARGET": 3}),
        row("tool", -20.0, False, False, 1, 0, False, False, {}),
        row("lateral", 30.0, False, False, 0, 1, False, False, {"tree0_BRANCH": 1}),
    ]
    table = study.confusion(rows, "truth_between_jaws")
    assert table["A_tof_return_in_mouth_box"] == {
        "in_and_says_in": 1,
        "in_and_says_out": 0,
        "out_and_says_in": 1,
        "out_and_says_out": 1,
    }
    assert table["B_any_beam_blocked"]["out_and_says_in"] == 1
    assert study.confusion(rows, "truth_at_mouth_centre")["B_centre_beam_blocked"]["out_and_says_out"] == 3
    response = {(r["axis"], r["offset_mm"]): r for r in study.response(rows)}
    assert response[("lateral", 30.0)]["beam_blocked_by_other_wood"] == 1
    assert response[("none", 0.0)]["truth_between_jaws"] == 1


def test_depth_validation_flags_a_line_of_sight_the_mesh_set_cannot_explain(study, tmp_path):
    scene = _scene(study)
    # A camera level with the mouth, looking along +y at it: optical x = world x, y = world -z, z = world y.
    looking_at_mouth = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, -1.0, 0.0]])
    rig = study.Rig(IDENTITY, 0.0, [0.0, -0.14, 0.070], looking_at_mouth)
    optical = rig.cam_R.T @ (rig.mouth - rig.cam_p)
    u0, v0 = (int(np.floor(c)) for c in (study.kmp.K_CAM @ optical)[:2] / optical[2])
    assert (u0, v0) == (240, 160)
    reach = study.DEPTH_NEIGHBOURHOOD_PX
    cast = np.full((study.kmp.IMAGE_H, study.kmp.IMAGE_W), np.inf)
    for v in range(v0 - reach, v0 + reach + 1):
        for u in range(u0 - reach, u0 + reach + 1):
            ray = rig.cam_R @ np.linalg.solve(study.kmp.K_CAM, [u + 0.5, v + 0.5, 1.0])
            ray /= np.linalg.norm(ray)
            t, _, _ = scene.all.cast(rig.cam_p[None], ray[None])
            if np.isfinite(t[0]):
                cast[v, u] = t[0] * (ray @ rig.cam_R[:, 2])
    (tmp_path / "frames").mkdir()
    np.save(tmp_path / "frames/depth_00007.npy", cast)
    same = study.validate_depth(tmp_path, 7, rig, scene)
    assert same["neighbourhood_agree"] == same["neighbourhood_pixels"] == (2 * reach + 1) ** 2
    assert same["line_of_sight_verified"]
    # Something the mesh set does not hold, drawn in front of the mouth across the whole neighbourhood.
    np.save(tmp_path / "frames/depth_00008.npy", np.full_like(cast, 0.05))
    blocked = study.validate_depth(tmp_path, 8, rig, scene)
    assert blocked["neighbourhood_agree"] == 0 and not blocked["line_of_sight_verified"]


def test_ground_relief_joins_the_channels_and_reports_a_tool_beneath_it(study):
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 0, 0], [6, 0, 0], [5, 1, 0]], float)
    export = {"v0": vertices, "t0": np.array([[0, 1, 2], [3, 4, 5]])}
    report = {
        "blender_scene": {
            "target_tree_index": 0,
            "translation_w_m": [0.0, 0.0, 0.0],
            "yaw_degrees": 0.0,
            "target": {"source_component": {"source_vertex_indices": [3, 4, 5]}},
        }
    }
    names = ["tree0:/root/tree0_SPUR/Shape"]
    # A horizontal sheet 8 mm above the jaw mouth, wide enough to cover the camera too.
    sheet = _quad([-1, -1, 0.078], [1, -1, 0.078], [1, 1, 0.078], [-1, 1, 0.078])
    trees = study.build_scene(export, names, report, study.kmp.NumpyMeshIndex)
    full = study.build_scene(export, names, report, study.kmp.NumpyMeshIndex, ground=sheet)
    assert trees.ground is None and study.ground_relief(_rig(study), trees) is None
    assert full.labels[-1] == study.GROUND
    relief = study.ground_relief(_rig(study), full)
    assert relief["mouth"] == {"ground_above_mm": 8.0, "ground_below_mm": None, "beneath_the_surface": True}
    assert relief["camera"]["beneath_the_surface"] and relief["tool_point"]["beneath_the_surface"]
    above = study.ground_relief(_rig(study).shifted([0, 0, 0.2]), full)
    assert above["mouth"]["ground_below_mm"] == pytest.approx(192.0) and not above["mouth"]["beneath_the_surface"]
    # The sheet crosses the jaw gap, so the beam grid reports it; the truth, which is the target alone, does not.
    reading = study.beams(_rig(study), study.jsm.JAW_OPEN_GAP_M, full)
    assert reading["beam_hits"].get(study.GROUND, 0) == 0  # a beam parallel to the sheet does not cross it
    tilted = _quad([-1, -1, 0.0], [1, -1, 0.14], [1, 1, 0.14], [-1, 1, 0.0])
    crossing = study.build_scene(export, names, report, study.kmp.NumpyMeshIndex, ground=tilted)
    reading = study.beams(_rig(study), study.jsm.JAW_OPEN_GAP_M, crossing)
    assert reading["beam_hits"].get(study.GROUND, 0) > 0 and not reading["truth_between_jaws"]


def test_summary_separates_runs_whose_tool_is_beneath_the_ground(study):
    def run(name, beneath):
        row = {
            "axis": "none",
            "offset_mm": 0.0,
            "truth_between_jaws": True,
            "truth_at_mouth_centre": False,
            "tof": {"returns_in_mouth": 1, "returns_in_mouth_by_label": {"TARGET": 1}},
            "beams_blocked": 2,
            "beam_centre_blocked": False,
            "camera": {"surface_in_mouth": beneath, "surface_label": study.GROUND if beneath else None},
            "beam_hits": {"TARGET": 2},
            "camera_line_of_sight_verified": True,
        }
        return {
            "run": name,
            "tof_validation": {
                "zones": 4,
                "both_valid": 2,
                "validity_same": 4,
                "within_1mm_as_ray_length": 2,
                "within_1mm_as_axis_depth": 0,
            },
            "timeline": [{**row, "cut_phase": "closing"}],
            "displacements": [row],
            "camera_line_of_sight_verified": True,
            "ground_at_ready": {"relief": {}, "tool_beneath_the_ground_surface": beneath},
        }

    summary = study.summarise([run("a", False), run("b", False), run("c", True)])
    assert summary["ground_relief"]["runs_with_the_tool_beneath_the_surface"] == ["c"]
    above, beneath = summary["runs_with_the_tool_above_the_ground"], summary["runs_with_the_tool_beneath_the_ground"]
    assert (above["runs"], beneath["runs"], summary["poses"]) == (2, 1, 3)
    assert above["at_recorded_pose"]["camera_surface_when_it_says_in"] == {}
    assert beneath["at_recorded_pose"]["camera_surface_when_it_says_in"] == {study.GROUND: 1}
    assert above["closing_frames"] == {
        "frames": 2,
        "target_between_jaws": 2,
        "target_on_mouth_centre_segment": 0,
        "any_beam_blocked": 2,
        "tof_return_in_mouth_box": 2,
    }
    assert summary["caster_validation"]["met"]
