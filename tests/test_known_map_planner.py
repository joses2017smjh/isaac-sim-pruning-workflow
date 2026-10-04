"""Protect the known-map planner: kinematics on a toy chain, hull-vs-mesh clearance on boxes, the IK step limits, the
final-pose grid and its count, the selection rule's three conditions, the acceptance bookkeeping and the output
refusals. Everything here is synthetic except the tests marked for recordings, which skip when
artifacts/vision_robustness (or the robot URDF) is absent.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"
RUNS = TOOLS.parent / "artifacts/vision_robustness"

TOY_URDF = """<robot name="toy">
  <link name="world"/>
  <link name="base"/>
  <link name="upper"/>
  <link name="lower"/>
  <link name="tip"/>
  <joint name="mount" type="fixed"><parent link="world"/><child link="base"/>
    <origin xyz="0 0 0.1" rpy="0 0 0"/></joint>
  <joint name="j1" type="revolute"><parent link="base"/><child link="upper"/>
    <origin xyz="0 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="j2" type="revolute"><parent link="upper"/><child link="lower"/>
    <origin xyz="0.5 0 0" rpy="0 0 0"/><axis xyz="0 0 1"/></joint>
  <joint name="tipj" type="fixed"><parent link="lower"/><child link="tip"/>
    <origin xyz="0.3 0 0" rpy="0 0 0"/></joint>
</robot>"""


@pytest.fixture(scope="module")
def tool():
    spec = importlib.util.spec_from_file_location("known_map_planner", TOOLS / "known_map_planner.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _toy(tool):
    parsed = tool.parse_urdf(TOY_URDF)
    return tool.Kinematics(
        parsed, base=(0.0, 0.0, 0.0), eef="tip", tool_in_base=(0.0, 0.0, 0.0), arm_joints=("j1", "j2")
    )


def _box(centre, half):
    corners = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float)
    faces = np.array(
        [[0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1]]
        + [[2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3]]
    )
    return np.asarray(centre, float) + corners * np.asarray(half, float), faces


# ------------------------------------------------------------------------------------------- kinematics
def test_toy_chain_forward_kinematics(tool):
    kin = _toy(tool)
    assert kin.root == "world"
    for q1, q2 in ((0.0, 0.0), (math.pi / 2, 0.0), (0.3, -1.1)):
        p, R = kin.tool_pose([q1, q2])
        expected = np.array(
            [0.5 * math.cos(q1) + 0.3 * math.cos(q1 + q2), 0.5 * math.sin(q1) + 0.3 * math.sin(q1 + q2), 0.1]
        )
        np.testing.assert_allclose(p, expected, atol=1e-12)
        np.testing.assert_allclose(R[:2, 0], [math.cos(q1 + q2), math.sin(q1 + q2)], atol=1e-12)
    fk = kin.fk([0.3, -1.1])
    np.testing.assert_allclose(fk["tip"], kin.link_transform([0.3, -1.1], "tip"), atol=0)
    np.testing.assert_allclose(fk["lower"][:3, 3], [0.5 * math.cos(0.3), 0.5 * math.sin(0.3), 0.1], atol=1e-12)


def test_toy_chain_jacobian_matches_analytic(tool):
    kin = _toy(tool)
    q1, q2 = 0.4, 0.7
    J = kin.jacobian_tool([q1, q2])
    analytic = np.array(
        [
            [-0.5 * math.sin(q1) - 0.3 * math.sin(q1 + q2), -0.3 * math.sin(q1 + q2)],
            [0.5 * math.cos(q1) + 0.3 * math.cos(q1 + q2), 0.3 * math.cos(q1 + q2)],
        ]
    )
    np.testing.assert_allclose(J[:2], analytic, atol=1e-5)
    np.testing.assert_allclose(J[5], [1.0, 1.0], atol=1e-5)


def test_ik_step_is_bounded_and_converges(tool):
    kin = _toy(tool)
    q0 = np.array([0.2, 0.9])
    p0, R0 = kin.tool_pose(q0)
    far = np.array([-0.6, 0.2, 0.1])
    q1, err, _ = kin.ik_step(q0, far, R0)
    assert err == pytest.approx(np.linalg.norm(far - p0))
    assert np.abs(q1 - q0).max() == pytest.approx(tool.IK_MAX_JOINT_STEP_RAD)
    near = p0 + np.array([1e-4, -1e-4, 0.0])
    q2, _, _ = kin.ik_step(q0, near, R0)
    assert 0 < np.abs(q2 - q0).max() < tool.IK_MAX_JOINT_STEP_RAD
    # a reachable pose: position and orientation of a known configuration
    qt = np.array([0.5, 0.4])
    pt, Rt = kin.tool_pose(qt)
    q, err = kin.ik([0.45, 0.45], pt, Rt, iters=80)
    assert err < 1e-6
    np.testing.assert_allclose(q, qt, atol=1e-5)


def test_mouth_line_takes_equal_steps_of_at_most_4_mm(tool):
    p0, R = np.zeros(3), np.eye(3)
    goal = np.array([0.1, 0.05, 0.07 + 0.02])
    steps = tool.mouth_line(p0, R, goal)
    assert len(steps) == math.ceil(np.linalg.norm(goal - tool.MOUTH_IN_TOOL) / tool.STEP_M)
    moves = np.linalg.norm(np.diff(np.vstack([p0, steps]), axis=0), axis=1)
    assert moves.max() <= tool.STEP_M + 1e-12
    np.testing.assert_allclose(moves, moves[0])
    np.testing.assert_allclose(steps[-1] + R @ tool.MOUTH_IN_TOOL, goal, atol=1e-12)


# ------------------------------------------------------------------------------------------- geometry kernels
def test_closest_point_on_triangle_regions(tool):
    a, b, c = np.array([0.0, 0, 0]), np.array([1.0, 0, 0]), np.array([0.0, 1, 0])
    cases = {
        (0.2, 0.2, 0.5): (0.2, 0.2, 0.0),  # face
        (-1.0, -1.0, 0.0): (0.0, 0.0, 0.0),  # vertex a
        (2.0, -0.5, 0.0): (1.0, 0.0, 0.0),  # vertex b
        (0.5, -1.0, 0.3): (0.5, 0.0, 0.0),  # edge ab
        (1.0, 1.0, 0.0): (0.5, 0.5, 0.0),  # edge bc
    }
    for p, expected in cases.items():
        np.testing.assert_allclose(tool.closest_point_on_triangle(np.array(p), a, b, c), expected, atol=1e-12)


def test_ray_cast_and_convex_depth_on_a_box(tool):
    V, F = _box([0, 0, 0], [0.1, 0.2, 0.3])
    index = tool.NumpyMeshIndex([(V, F)])
    t, owner, prim = index.cast(np.array([[-1.0, 0, 0], [-1.0, 5, 0]]), np.array([[1.0, 0, 0], [1.0, 0, 0]]))
    assert t[0] == pytest.approx(0.9) and owner[0] == 0 and prim[0] >= 0
    assert np.isinf(t[1]) and owner[1] == -1
    n, off = tool.convex_planes(V, F)
    depth = tool.halfspace_depth(np.array([[0.0, 0, 0], [0.05, 0, 0], [0.3, 0, 0]]), n, off)
    np.testing.assert_allclose(depth, [0.1, 0.05, -0.2], atol=1e-12)
    np.testing.assert_allclose(index.signed_distance(np.array([[0.05, 0, 0], [0.3, 0, 0]])), [-0.05, 0.2], atol=1e-12)


def test_brute_force_hull_of_a_cube_is_closed_and_outward(tool):
    V, _ = _box([0.3, -0.1, 0.2], [0.01, 0.02, 0.03])
    P, F = tool.brute_force_hull(V)
    n, off = tool.convex_planes(P, F)
    assert (tool.halfspace_depth(P.mean(0)[None], n, off) > 0).all()
    assert (np.abs(tool.halfspace_depth(P, n, off)) < 1e-12).all()


def _fake_export(tool, boxes):
    """An export dict like the streamed one: tree0 spur (target + one more box), tree1 spur, one post."""
    out, names = {}, []
    for i, (name, parts) in enumerate(boxes.items()):
        vs, fs, cs = [], [], []
        base = 0
        for label, (V, F) in enumerate(parts):
            vs.append(V)
            fs.append(F + base)
            cs.append(np.full(len(V), label))
            base += len(V)
        names.append(name)
        out[f"v{i}"], out[f"t{i}"], out[f"c{i}"] = np.concatenate(vs), np.concatenate(fs), np.concatenate(cs)
    out["names"] = np.asarray(names)
    return out


def _box_robot(tool, half=(0.05, 0.05, 0.05)):
    """A one-link 'robot' whose only collision hull is a box at the origin of the end-effector frame."""
    parsed = tool.parse_urdf(
        '<robot name="b"><link name="world"/><link name="box"/>'
        '<joint name="q" type="revolute"><parent link="world"/><child link="box"/><axis xyz="0 0 1"/></joint></robot>'
    )
    kin = tool.Kinematics(parsed, base=(0, 0, 0), eef="box", tool_in_base=(0, 0, 0), arm_joints=("q",))
    V, F = _box([0, 0, 0], half)
    grid = np.array(
        [[x, y, half[2]] for x in np.linspace(-half[0], half[0], 11) for y in np.linspace(-half[1], half[1], 11)]
    )

    def sampler(link, count):
        faces = [grid, grid * [1, 1, -1]]
        faces += [np.roll(grid, 1, axis=1) * s for s in (1, -1)] + [np.roll(grid, 2, axis=1) * s for s in (1, -1)]
        return np.vstack(faces + [V])

    return tool.RobotModel(kin, {"box": (V, F)}, sampler, tool.Backend.numpy())


def _orchard(tool, obstacle_centre, obstacle_half):
    target = _box([0.0, 0.5, 0.0], [0.004, 0.004, 0.02])
    obstacle = _box(obstacle_centre, obstacle_half)
    far = _box([0.0, -0.6, 0.0], [0.01, 0.01, 0.01])
    export = _fake_export(
        tool,
        {
            "tree0:/Root/tree0_SPUR/Shape": [target, obstacle],
            "tree1:/Root/tree1_SPUR/Shape": [far],
            "environment:/Root/post0/Cylinder": [_box([0.0, 0.0, -0.9], [0.01, 0.01, 0.01])],
        },
    )
    return tool.Orchard(export, np.zeros(3), np.eye(3), 0, np.arange(8), backend=tool.Backend.numpy())


def test_hull_clearance_between_separated_boxes(tool):
    robot = _box_robot(tool)
    orch = _orchard(tool, [0.08, 0.0, 0.0], [0.02, 0.02, 0.02])  # faces 1 cm apart along x
    res = tool.link_clearance(robot, orch, "box", np.eye(4))
    assert res["min_m"] == pytest.approx(0.01, abs=1e-6)
    assert res["nearest"] == "tree0_SPUR(non-target)" and res["component_first_vertex"] == 8
    assert res["target_m"] == pytest.approx(0.5 - 0.05 - 0.004, abs=1e-6)
    assert orch.meshes[orch.target_k][0] == tool.Orchard.TARGET_LABEL


def test_hull_clearance_is_negative_inside_and_follows_the_pose(tool):
    robot = _box_robot(tool)
    orch = _orchard(tool, [0.066, 0.0, 0.0], [0.02, 0.02, 0.02])  # penetrates 4 mm along x
    res = tool.link_clearance(robot, orch, "box", np.eye(4))
    assert res["min_m"] == pytest.approx(-0.004, abs=1e-6)
    moved = np.eye(4)
    moved[:3, 3] = [-0.014, 0.0, 0.0]  # back the box off: 1 cm gap
    assert tool.link_clearance(robot, orch, "box", moved)["min_m"] == pytest.approx(0.01, abs=1e-6)
    assert tool.clearance2(robot, orch, [0.0])["box"]["min_m"] == pytest.approx(-0.004, abs=1e-6)


def test_densify_is_seeded_and_keeps_vertices(tool):
    V, F = _box([0.0, 0.0, 0.7], [0.01, 0.01, 0.01])
    meshes = [("a", V, F, np.zeros(len(V), int))]
    p1, o1 = tool.densify(meshes)
    p2, _ = tool.densify(meshes)
    np.testing.assert_array_equal(p1, p2)
    assert (o1 == 0).all() and len(p1) > len(V)
    assert (p1[:, None, :] == V[None]).all(axis=2).any(axis=0).all()


# ------------------------------------------------------------------------------------------- search grid and rule
def test_orientation_grid_count_and_exclusion(tool):
    axis_530 = [-0.7111213953654838, 0.5039981689593108, -0.49019609008957793]
    grid = tool.orientation_grid(axis_530)
    assert len(grid) == 6016
    dirs = {row["direction_index"] for row in grid}
    assert len(dirs) == 376
    assert len(grid) == len(dirs) * len(tool.GRID_PSI_DEG) * len(tool.GRID_SHORT_M)
    for row in grid[:: len(grid) // 50]:
        angle = math.degrees(math.acos(min(1.0, abs(row["R"][:, 2] @ np.asarray(axis_530)))))
        assert angle >= tool.AXIS_EXCLUSION_DEG
        np.testing.assert_allclose(row["R"].T @ row["R"], np.eye(3), atol=1e-12)
    row = grid[777]
    ident = tool.grid_identity(row["R"])
    assert (ident["direction_index"], ident["psi_deg"]) == (row["direction_index"], row["psi_deg"])
    assert ident["residual"] < 1e-12


def test_committed_planned_orientations_lie_on_the_grid(tool):
    planned = {"530": (98, 315), "19444": (195, 90)}
    doc = json.loads((TOOLS.parent / tool.EVIDENCE["planned"]).read_text())
    for target in doc["targets"]:
        key = str(target["component_first_vertex"])
        if key in planned:
            ident = tool.grid_identity(tool.quat_wxyz_to_R(target["planned_final_tool_quat_wxyz"]))
            assert (ident["direction_index"], ident["psi_deg"]) == planned[key]
            assert ident["residual"] < 1e-3


def test_selection_rule_three_conditions(tool):
    ok = tool.selection_conditions(True, [169, 150, 140], 133)
    assert ok["qualifies"] and ok["approach_min_kept"] == 140 and ok["closure_kept"] == 133
    assert not tool.selection_conditions(False, [169, 150], 133)["qualifies"]  # (a)
    b = tool.selection_conditions(True, [169, 139, 160], 120)  # (b): one pose under 140
    assert not b["b_approach_keeps_patch"] and not b["qualifies"]
    c = tool.selection_conditions(True, [169, 160], 140)  # (c): closure keeps exactly 140
    assert not c["c_closure_loses_patch"] and not c["qualifies"]
    empty = tool.selection_conditions(True, [], None)  # projection undefined: fail closed
    assert not empty["b_approach_keeps_patch"] and not empty["c_closure_loses_patch"]


def test_kept_elements_counts_the_union_of_both_patches(tool):
    prev = np.zeros((13, 13), bool)
    cur = np.zeros((13, 13), bool)
    prev[:2] = True
    cur[1:3] = True
    assert tool.kept_elements(prev, cur) == 169 - 39


def test_classify_precedence(tool):
    base = {"home_min_link": "ur5e__forearm_link", "home_min_obj": "wire1_6:0"}
    assert (
        tool.classify({**base, "home_min_clear_m": -0.0006, "seed_los_blocked_home": True})["class"] == "home_overlap"
    )
    assert tool.classify({**base, "home_min_clear_m": 0.001, "seed_los_blocked_home": True})["class"] == "not_visible"
    assert tool.classify({**base, "home_min_clear_m": 0.001})["class"] == "marginal_home"
    path = {
        "first": {"contact": [59, "approach", "mock_pruner__base", "x:1", -0.001], "tof": [13, "approach", 0.0585, {}]}
    }
    call = tool.classify({**base, "home_min_clear_m": 0.01, "path": path})
    assert call["class"] == "tof_minimum_clearance" and call["step"] == 13 and call["recorded_frame_window"] == [14, 18]
    assert tool.classify({**base, "home_min_clear_m": 0.01, "path": {"first": {}}})["class"] == "geometry_clear"
    assert tool.classify({**base, "home_min_clear_m": -0.0347})["detail"].endswith("by 34.7 mm at the settled home")


def test_model_events_order_and_tof_sensor(tool):
    first = {
        "contact": [57, "approach", "mock_pruner__base", "tree0_BRANCH:0", -0.0002],
        "tof": [60, "approach", 0.0571, {"tof0": [0.0571, "tree0_BRANCH:0"], "tof1": [0.1066, "x:-1"]}],
        "los": [12, "wire0_2"],
    }
    events = tool.model_events(first)
    assert [e[1] for e in events] == ["los", "contact", "tof"]
    assert events[2][2:] == ("tof0", "tree0_BRANCH:0")


def test_recorded_passes_and_refusals_items(tool):
    replays = {
        k: {"replay": {"first": {}, "worst_clear_approach": [0.01], "worst_tof_approach": [0.07]}} for k in tool.PASSES
    }
    assert tool.item_recorded_passes(replays)["pass"]
    replays["8235"]["replay"]["first"] = {"contact": [68, "approach", "a", "b", 0.0004]}
    assert not tool.item_recorded_passes(replays)["pass"]
    refusals = {k: {"home_min_clear_m": -0.01, "home_min_link": "l", "home_min_obj": "o"} for k in tool.REFUSALS}
    assert tool.item_refused_layouts(refusals)["pass"]
    refusals["19145"]["home_min_clear_m"] = 0.0
    assert not tool.item_refused_layouts(refusals)["pass"]


def test_predictions_item_exempts_only_the_three_marginal_calls(tool):
    calls = tool.committed_calls()
    assert len(calls) == 40
    units = {k: {"call": {"class": v["class"], "detail": v["detail"]}} for k, v in calls.items()}
    assert tool.item_predictions(units)["pass"]
    for key in ("35837", "35957", "36017"):
        units[key]["call"]["class"] = "geometry_clear"
    item = tool.item_predictions(units)
    assert item["pass"] and item["required_matches"] == 37
    units["14944"]["call"]["class"] = "hazard_contact"
    assert not tool.item_predictions(units)["pass"]


# ------------------------------------------------------------------------------------------- refusals
def test_output_refusals(tool, tmp_path):
    with pytest.raises(ValueError, match="inside the repository"):
        tool.refuse_output(TOOLS.parent / "docs/evidence/x.json")
    with pytest.raises(ValueError, match="artifacts"):
        tool.refuse_output(tmp_path / "artifacts" / "x.json")
    existing = tmp_path / "exists.json"
    existing.write_text("{}")
    with pytest.raises(ValueError, match="overwrite"):
        tool.refuse_output(existing)
    assert tool.refuse_output(tmp_path / "new.json") == (tmp_path / "new.json").resolve()
    tool.write_json(tmp_path / "w.json", {"a": float("inf"), "b": np.float32(0.5), "c": np.arange(2)})
    assert json.loads((tmp_path / "w.json").read_text()) == {"a": "inf", "b": 0.5, "c": [0, 1]}
    with pytest.raises(FileExistsError):
        tool.write_json(tmp_path / "w.json", {})


def test_search_refuses_without_a_passing_acceptance(tool, tmp_path):
    failing = {"schema_version": 1, "kind": "acceptance", "all_items_pass": False, "items": {"x": {"pass": False}}}
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(failing))
    with pytest.raises(tool.GateError, match="did not pass"):
        tool.require_passing_acceptance(path)
    with pytest.raises(SystemExit):
        tool.main(["search-part", "--acceptance", str(path), "--output", str(tmp_path / "out.json")])
    assert not (tmp_path / "out.json").exists()
    with pytest.raises(SystemExit):
        tool.main(["acceptance", "--parts", str(path), "--output", str(TOOLS.parent / "out.json")])


def test_assembly_refuses_mixed_code(tool):
    part = {"kind": "acceptance_part", "part": "replays", "units": {}}
    a = {**part, "provenance": {"code_sha256": {"t": "1"}, "settings": {"ik_frame_updates": 80}, "inputs_sha256": {}}}
    b = {**part, "provenance": {"code_sha256": {"t": "2"}, "settings": {"ik_frame_updates": 80}, "inputs_sha256": {}}}
    with pytest.raises(tool.GateError, match="different code"):
        tool.assemble_acceptance([a, b])


# ------------------------------------------------------------------------------------------- recordings
needs_runs = pytest.mark.skipif(not RUNS.is_dir(), reason="artifacts/vision_robustness is absent")


@needs_runs
def test_recorded_kinematics_and_camera_mount(tool):
    if not tool.URDF_DEFAULT.is_file():
        pytest.skip("robot URDF is absent")
    kin = tool.Kinematics(tool.parse_urdf(tool.URDF_DEFAULT.read_text()))
    report, frames = tool.load_run(RUNS / tool.RECORDED["530"])
    for fr in frames[::20]:
        p, R = kin.tool_pose(fr["joint_position_rad"])
        assert np.linalg.norm(p - np.asarray(fr["tool_pose_wxyz"][:3])) < 2e-5
        assert math.degrees(tool.rotation_angle_rad(R, tool.quat_wxyz_to_R(fr["tool_pose_wxyz"][3:]))) < 0.01
    q0 = np.asarray(report["initial_control_state"]["joint_pos"][0])
    assert q0[1] == pytest.approx(-0.528, abs=1e-3)
    p0, R0 = kin.tool_pose(q0)
    mount, rot = tool.camera_for_home(p0, R0, report["target_position_m"], report["blender_scene"]["target"]["axis_w"])
    assert np.linalg.norm(mount - np.asarray(report["camera"]["wrist_position_in_tool_m"])) < 1e-5
    assert np.abs(rot - np.asarray(report["camera"]["wrist_rotation_in_tool_ros"])).max() < 1e-4


@needs_runs
def test_recorded_stop_facts(tool):
    report, frames = tool.load_run(RUNS / tool.RECORDED["530"])
    contact = tool.recorded_contact(report, frames)
    assert contact["frame"] == 59 and contact["link"] == "mock_pruner__base"
    assert contact["force_n"] == pytest.approx(25.12, abs=0.01)
    _, frames = tool.load_run(RUNS / tool.RECORDED["19444"])
    tof = tool.recorded_tof(frames)
    assert tof["frame"] == 43 and tof["sensor"] == "tof1" and tof["range_m"] == pytest.approx(0.0598, abs=1e-4)
    _, frames = tool.load_run(RUNS / tool.RECORDED["12142"])
    assert tool.recorded_tracking_loss(frames)["frame"] == 30


@needs_runs
def test_placement_from_component_matches_a_recorded_layout(tool):
    report, _ = tool.load_run(RUNS / tool.RECORDED["19444"])
    component = report["blender_scene"]["target"]["source_component"]
    t, R = tool.placement_from_component(component["center_m"])
    t_rec, R_rec = tool.placement_from_report(report)
    np.testing.assert_allclose(t, t_rec, atol=1e-9)
    np.testing.assert_allclose(R, R_rec, atol=1e-12)
