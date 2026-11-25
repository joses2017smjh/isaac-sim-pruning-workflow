"""tools/cast_jaw_beams.py on a hand-built scene: beam codes, the join check and the per-capture tallies."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
IDENTITY = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]


@pytest.fixture(scope="module")
def cast():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("cast_jaw_beams", TOOLS / "cast_jaw_beams.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quad(a, b, c, d):
    return np.array([a, b, c, d], float), np.array([[0, 1, 2], [0, 2, 3]])


def _scene(cast, ground=None):
    """Jaw frame = world. The target crosses the gap at the jaw tips (z 75 to 90 mm), a twig at the jaw's root."""
    target = _quad([0, -0.02, 0.075], [0, 0.02, 0.075], [0, 0.02, 0.09], [0, -0.02, 0.09])
    twig = _quad([0.004, 0.006, 0.055], [0.004, 0.01, 0.055], [0.004, 0.01, 0.065], [0.004, 0.006, 0.065])
    meshes = [("tree0_BRANCH", *twig), ("TARGET", *target)]
    if ground is not None:
        meshes.append((cast.study.GROUND, *ground))
    return cast.study.Scene(meshes, index_cls=cast.kmp.NumpyMeshIndex)


def _rig(cast, pose=IDENTITY):
    return cast.study.Rig(pose, 0.0, np.zeros(3), np.eye(3))


def test_beam_codes_name_what_each_beam_hits_first(cast):
    scene = _scene(cast)
    codes, target = cast.read_beams(_rig(cast), cast.jsm.JAW_OPEN_GAP_M, scene)
    # Heights major (60, 70, 80 mm), jaw positions minor (-8, 0, +8 mm): the twig at the root, the target at the tips.
    assert codes == "..W...TTT" and target == "000000111"
    back_codes, back_target = cast.read_beams(_rig(cast).shifted([0, 0, -0.02]), cast.jsm.JAW_OPEN_GAP_M, scene)
    assert back_target == "000000000" and cast.TARGET not in back_codes
    assert cast.blocked(codes) and cast.on_target(target) and not cast.blocked(".........")


def test_the_ground_is_a_code_of_its_own_and_sets_the_beneath_flags(cast):
    sheet = _quad([-1, -1, 0.0], [1, -1, 0.14], [1, 1, 0.14], [-1, 1, 0.0])  # crosses the gap at the mouth height
    scene = _scene(cast, ground=sheet)
    codes, target = cast.read_beams(_rig(cast), cast.jsm.JAW_OPEN_GAP_M, scene)
    assert cast.GROUND in codes and target == "000000111"
    assert cast.beneath_flags(_rig(cast), scene, has_camera=False) == "10-"
    assert cast.beneath_flags(_rig(cast), _scene(cast), has_camera=True) == "---"


def _capture(cast, poses, progress, ready, phases):
    records, frames = [], []
    for k, pose in enumerate(poses):
        records.append(
            {
                "index": k,
                "stamp_ns": 1000 + k,
                "tool_pose_wxyz": [float(v) for v in pose],
                "target": {"id": "tree0_SPUR_component_1"},
                "proposal": {"cut_phase": phases[k], "cut_detached": False, "closure_progress": progress[k]},
                "python": {"certificate": {"ready_to_close": ready[k]}},
            }
        )
        frames.append({"index": k, "tool_pose_wxyz": list(pose), "visual_jaw_closure_progress": progress[k]})
    report = {"blender_scene": {"visual_proxy_roll_rad": 0.0, "target": {"radius_m": 0.005}}}
    return records, frames, report


def test_a_capture_is_cast_cycle_by_cycle_and_tallied(cast):
    scene = _scene(cast)
    far = [0.0, 0.0, -0.05, 1.0, 0.0, 0.0, 0.0]
    records, frames, report = _capture(
        cast,
        poses=[far, IDENTITY, IDENTITY, IDENTITY],
        progress=[0.0, 0.0, 0.0, 0.5],
        ready=[False, True, True, True],
        phases=["approach", "align", "closing", "closing"],
    )
    reading = cast.cast_capture(records, frames, report, scene)
    assert reading["index"] == [0, 1, 2, 3] and reading["first_ready"] == 1
    assert reading["beams"][0] == "........." and reading["target_beams"][1] == "000000111"
    assert reading["gap_mm"] == [32.0, 32.0, 32.0, 21.0]
    assert len(reading["displaced_at_first_ready"]) == 3 * 2 * len(cast.study.OFFSETS_M)
    assert reading["pose_digest"][1] == cast.pose_digest(IDENTITY) != reading["pose_digest"][0]
    row = cast.summarise_capture("demo", records, reading)
    assert row["first_closing_index"] == 2 and not row["tool_beneath_the_ground"]
    assert row["ready"] == {
        "cycles": 3,
        "any_beam_blocked": 3,
        "target_between_jaws": 3,
        "blocked_by": {"W": 3, "T": 9},
    }
    assert row["closing"]["cycles"] == 2 and row["jaw_advancing"]["cycles"] == 1
    # 20 mm short of the spur the target is out of the gap in every ready or closing cycle.
    assert row["tool_short_ready_or_closing"]["target_between_jaws"] == 0
    summary = cast.totals([row])
    assert summary["reach_closing"] == 1 and summary["closing"]["captures_with_a_cycle_without_a_blocked_beam"] == 0


def test_a_record_and_a_frame_of_different_cycles_are_refused(cast):
    records, frames, report = _capture(cast, [IDENTITY], [0.0], [False], ["approach"])
    frames[0]["tool_pose_wxyz"] = [0.001, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    with pytest.raises(RuntimeError, match="not the same cycle"):
        cast.cast_capture(records, frames, report, _scene(cast))


def test_castable_needs_the_orchard_placement(cast):
    scene = {"translation_w_m": [0, 0, 0], "yaw_degrees": 0.0, "visual_proxy_roll_rad": 0.1}
    target = {"source_component": {"source_vertex_indices": [1, 2, 3]}}
    assert cast.castable({"blender_scene": {**scene, "target": target}})
    assert not cast.castable({"blender_scene": scene})
    assert not cast.castable({})
