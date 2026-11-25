"""tools/orchard_ground.py and tools/audit_ground.py on a hand-built scene with a one-link robot."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture(scope="module")
def audit():
    sys.path.insert(0, str(TOOLS))
    spec = importlib.util.spec_from_file_location("audit_ground", TOOLS / "audit_ground.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _quad(a, b, c, d):
    return np.array([a, b, c, d], float), np.array([[0, 1, 2], [0, 2, 3]])


def _scenes(audit, ground_z=0.5):
    """A small target quad well away from everything, and a horizontal ground sheet at ``ground_z``."""
    target = _quad([5, 5, 5], [5.1, 5, 5], [5.1, 5.1, 5], [5, 5.1, 5])
    sheet = _quad([-3, -3, ground_z], [3, -3, ground_z], [3, 3, ground_z], [-3, 3, ground_z])
    index = audit.kmp.NumpyMeshIndex
    trees = audit.study.Scene([("TARGET", *target)], index_cls=index)
    full = audit.study.Scene([("TARGET", *target), (audit.orchard_ground.GROUND, *sheet)], index_cls=index)
    return trees, full


class _Kinematics:
    arm_joints = ("lift",)

    def fk(self, joints):
        T = np.eye(4)
        T[2, 3] = joints[0]
        return {"arm": T}

    def tool_pose(self, joints):
        return np.array([0.0, 0.0, joints[0] + 0.1]), np.eye(3)


class _Robot:
    """One link: a 10 cm stick below the tool point, raised and lowered by its single joint."""

    links = ["arm"]
    kin = _Kinematics()

    def local_points(self, link, count):
        return np.array([[0.0, 0.0, z] for z in np.linspace(0.0, 0.1, 5)])


def _frame(index, lift, camera=True):
    frame = {"index": index, "tool_pose_wxyz": [0.0, 0.0, lift + 0.1, 1.0, 0.0, 0.0, 0.0], "joint_position_rad": [lift]}
    if camera:
        frame["wrist_position_w_m"] = [0.0, -0.14, lift + 0.1]
        frame["wrist_rotation_w_ros"] = np.eye(3).tolist()
    return frame


def test_beneath_means_ground_above_and_none_below(audit):
    _, full = _scenes(audit)
    ground = audit.orchard_ground
    points = np.array([[0, 0, 0.2], [0, 0, 0.8], [9, 9, 0.2]], float)
    assert ground.beneath(full.ground, points).tolist() == [True, False, False]
    above, below = ground.vertical_gaps(full.ground, points)
    assert above[0] == pytest.approx(0.3) and below[1] == pytest.approx(0.3)
    assert not np.isfinite(above[1]) and not np.isfinite(below[0]) and not np.isfinite(above[2])
    moved = ground.placed((np.array([[1.0, 0.0, 0.0]]), np.array([[0, 0, 0]])), [0, 0, 2], audit.kmp.yaw_R(90.0))
    assert np.allclose(moved[0], [[0.0, 1.0, 2.0]])


def test_a_capture_is_classified_by_what_goes_beneath(audit, tmp_path):
    trees, full = _scenes(audit)
    robot = _Robot()
    # The stick's top (the tool point) stays above the sheet while its lower end dips beneath it.
    dipping = [_frame(0, 0.6), _frame(1, 0.45), _frame(2, 0.6)]
    row = audit.audit_capture(tmp_path, dipping, trees, full, robot)
    assert row["beneath_frames"] == {"tool_point": 0, "mouth": 0, "camera": 0}
    assert row["links_beneath_frames"] == {"arm": 1} and row["first_frame_beneath"] == 1
    assert row["kinematics"] == {"frames_with_joints": 3, "max_tool_point_error_mm": 0.0, "ok": True}
    assert audit.classify(row) == "links_only" and row["depth_first_frame"] is None
    buried = [_frame(0, 0.1), _frame(1, 0.1, camera=False)]
    row = audit.audit_capture(tmp_path, buried, trees, full, robot)
    assert row["beneath_frames"] == {"tool_point": 2, "mouth": 2, "camera": 1}
    assert audit.classify(row) == "tool" and row["anything_beneath_frames"] == 2
    clear = audit.audit_capture(tmp_path, [_frame(0, 0.9)], trees, full, robot)
    assert audit.classify(clear) == "above" and clear["first_frame_beneath"] is None


def test_kinematics_that_disagree_with_the_recorded_pose_are_flagged(audit, tmp_path):
    trees, full = _scenes(audit)
    frame = _frame(0, 0.9)
    frame["tool_pose_wxyz"][2] += 0.01
    row = audit.audit_capture(tmp_path, [frame], trees, full, _Robot())
    assert row["kinematics"]["max_tool_point_error_mm"] == pytest.approx(10.0) and not row["kinematics"]["ok"]


def test_the_depth_check_tells_the_two_mesh_sets_apart(audit, tmp_path):
    trees, full = _scenes(audit, ground_z=0.5)
    # A camera beneath the sheet looking straight up at it: every pixel sees the ground 0.3 m away.
    frame = {"index": 4, "wrist_position_w_m": [0.0, 0.0, 0.2], "wrist_rotation_w_ros": np.eye(3).tolist()}
    (tmp_path / "frames").mkdir()
    np.save(tmp_path / "frames/depth_00004.npy", np.full((audit.kmp.IMAGE_H, audit.kmp.IMAGE_W), 0.3))
    check = audit.depth_check(tmp_path, frame, trees, full)
    assert check["pixels"] == 24 * 16 == check["recorded_finite"]
    assert check["agree_with_ground"] == check["pixels"] == check["agree_on_the_ground"]
    assert check["agree_trees_only"] == 0
    assert audit.depth_check(tmp_path, {**frame, "index": 5}, trees, full) is None


def test_the_summary_counts_classes_and_targets(audit):
    def row(name, target, cls, frames=10, beneath=0, links=None):
        return {
            "capture": name,
            "target": target,
            "class": cls,
            "frames": frames,
            "anything_beneath_frames": beneath,
            "links_beneath_frames": links or {},
            "kinematics": {"frames_with_joints": frames, "max_tool_point_error_mm": 0.4, "ok": True},
            "depth_first_frame": {"pixels": 4, "recorded_finite": 4, "agree_trees_only": 1, "agree_with_ground": 4},
            "depth_last_frame": None,
        }

    summary = audit.summarise(
        [
            row("a", "t1", "above"),
            row("b", "t2", "tool", beneath=10, links={"arm": 10}),
            row("c", "t2", "links_only", beneath=3, links={"arm": 3}),
        ]
    )
    assert summary["by_class"] == {"above": 1, "tool": 1, "links_only": 1}
    assert summary["beneath_by_target"] == {"t2": 2} and summary["beneath_whole_run"] == 1
    assert summary["links_beneath"] == {"arm": 2} and summary["tool"] == ["b"] and summary["links_only"] == ["c"]
    assert summary["depth"]["tool"] == {
        "images": 1,
        "pixels": 4,
        "recorded_finite": 4,
        "agree_trees_only": 1,
        "agree_with_ground": 4,
    }
    assert summary["kinematics"] == {"captures_checked": 3, "captures_ok": 3, "max_tool_point_error_mm": 0.4}
