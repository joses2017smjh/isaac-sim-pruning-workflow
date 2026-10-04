#!/usr/bin/env python3
"""Known-map swept-geometry planner: the rebuild's acceptance test and the closure-hold candidate search.

docs/SCOPE_CLOSURE_HOLD_GENERALIZATION_2026-10-02.md fixed, before any code was written, the acceptance test a
rebuilt known-map model must pass and the selection rule its candidate search applies. This tool is that model.
CPU only: simulator geometry (URDF kinematics, the robot's convex collision hulls, the exported orchard meshes);
no dynamics, no perception, no GPU, and never a grader (tools/validate_vision_sequence.py alone grades runs).

Provenance. The model behind docs/evidence/contact_diagnosis_2026-09-27.json and
docs/evidence/tree1_swept_path_predictions_2026-09-27.json was never committed. Its scratch scripts (geom.py,
geom2.py, sweep.py, plan.py, feasibility.py, cand_aligned.py, cand_feaspath.py, final_check.py, screen.py and the
classification that wrote the predictions file) survive in the September 25-28 session transcripts. This file ports
them. Algorithms, constants, sampling seeds and thresholds are unchanged, except:
  * the orchard meshes are read from the hash-verified export by the Isaac venv python (which has pxr) and streamed
    through a pipe into memory; nothing holding geometry is written (the scripts kept a scratch cache);
  * the final-pose search can run lazily, in order of tool-axis angle from home until 60 margin-feasible poses are
    found, which yields the same smallest-re-orientation list as the full census (both modes are implemented);
  * per-frame IK updates are a parameter: 80 (the committed replay: damped least squares to convergence, warm
    started) by default, 12 for the controller's substeps reading of the scope's wording;
  * recorded-stop objects are identified with the replay's own clearance (clearance2), not the vertex-only clearance
    of the scripts' contact_nearest.py;
  * a planned path counts as clear for the search only if it also keeps camera line of sight and field of view on
    the approach (the committed "clear" ignored both; acceptance reports them separately).

Model (the scope's five bullets). Kinematics: URDF forward kinematics from each run's recorded settled home
(shoulder lift -0.528, against -0.65 commanded). Collision: each link's exact convex hull of its URDF collision mesh,
against the exported orchard meshes placed as the renderer places them (yaw 150 deg, the selected spur's centre at
the fixed target point); the selected spur is split off as a convex hull. Clearance is signed: hull-surface samples
to orchard triangles, and dense orchard surface samples (2.5 mm) against each hull (negative inside). Sensors: the two
8x8 ToF zones re-cast at the mock_pruner__tof0/tof1 offsets (65 deg diagonal field, [0.03, 3.4] m validity, stop below
0.06 m); the wrist camera's line of sight to the target (gap-aligned mount, rule of the renderer). Motion: the
baseline replay moves the mouth in a straight line to the target, at most 4 mm per frame, orientation held, each frame
solved by damped-least-squares IK (damping 0.05, 0.05 rad joint step limit). Search: 6016 final poses (a 400-point
Fibonacci sphere of tool axes, less those within 20 deg of the spur axis, x 8 rolls x mouth on target or 8 mm short).

Subcommands:
  acceptance-part  compute one part of the acceptance test (replays, refusals, predictions, reorientation,
                   jaw_projection) for some or all of its targets; numbers only
  acceptance       assemble parts into the acceptance JSON: every item with measured values, requirement and pass/fail
  search-part      candidate search over one shard of the jaw-fit-screened pool; refuses unless given an acceptance
                   JSON in which every item passes
  search           assemble search shards and apply the selection rule (a)-(c)

Outputs go to new files outside the repository and outside any artifacts/ directory; nothing is overwritten.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PACKAGE = REPO / "source/isaaclab_pruning"
SCHEMA_VERSION = 1
SCOPE_DOC = "docs/SCOPE_CLOSURE_HOLD_GENERALIZATION_2026-10-02.md"
RUNS_DEFAULT = REPO / "artifacts/vision_robustness"
EXPORT_DEFAULT = REPO / "artifacts/blender_scene/orchard_two_trees_v1"
ASSET_ID = "ur5e_mock_pruner_bdsdfede4c0_ur18e6f603_calib_3941312424972580002_urdf6b02ce9330be"
URDF_DEFAULT = REPO / f"artifacts/urdf/{ASSET_ID}/{ASSET_ID}_abs.urdf"
ISAAC_PYTHON_DEFAULT = os.environ.get(
    "PRUNING_ISAAC_PYTHON", "/nfs/hpc/share/sanchej7/Humanoid_Lite/venv-isaac60/bin/python"
)
EVIDENCE = {
    "scope": SCOPE_DOC,
    "diagnosis": "docs/evidence/contact_diagnosis_2026-09-27.json",
    "predictions": "docs/evidence/tree1_swept_path_predictions_2026-09-27.json",
    "planned": "docs/evidence/eval_targets_planned_pose_2026-09-27.json",
    "pool": "docs/evidence/eval_targets_2026-09-23.json",
    "tree0": "docs/evidence/eval_targets_tree0_2026-09-23.json",
    "tree1_listed": "docs/evidence/eval_targets_tree1_listed_2026-09-23.json",
    "seeded": "docs/evidence/eval_targets_tree1_seeded_2026-09-26.json",
    "seeded30": "docs/evidence/eval_targets_tree1_seeded30_2026-09-26.json",
}
EXPORT_FILES = ("tree0.usdc", "tree1.usdc", "environment.usdc")

# ------------------------------------------------------------------------------------------- model constants
BASE_W = np.array([0.0, 0.0, 0.70])
TOOL_IN_BASE = np.array([0.0, 0.0, 0.1601525])  # ur5e_pruner.yaml: the controlled tool point
MOUTH_IN_TOOL = np.array([0.0, 0.0, 0.070])  # vision_demo_controller.py: the jaw mouth
TOF_IN_BASE = {
    "tof0": np.array([0.04685226669, 0.0, 0.14444246761]),
    "tof1": np.array([-0.04685226669, 0.0, 0.14444246761]),
}
ARM_JOINTS = (
    "ur5e__shoulder_pan_joint",
    "ur5e__shoulder_lift_joint",
    "ur5e__elbow_joint",
    "ur5e__wrist_1_joint",
    "ur5e__wrist_2_joint",
    "ur5e__wrist_3_joint",
)
EEF_LINK = "mock_pruner__base"
TOF_DIAGONAL_FOV_DEG = 65.0
TOF_RANGE_M = (0.03, 3.4)
K_CAM = np.array([[320.0, 0.0, 240.0], [0.0, 320.0, 160.0], [0.0, 0.0, 1.0]])
IMAGE_W, IMAGE_H = 480, 320
TARGET_W = np.array([0.25123947, 0.63413405, 0.81517712])  # render_pruning_workflow.py TARGET_POSITION_M
ORCHARD_YAW_DEG = 150.0
STEP_M = 0.004
MAX_ROTATION_RAD = math.radians(1.5)
MIN_CLEAR_TOF_M = 0.06
CONTACT_M = 0.0005
LOS_MARGIN_M = 0.01
PEDESTAL_R_M, PEDESTAL_TOP_M = 0.14, 0.70
PEDESTAL_SKIP = ("ur5e__base_link_inertia", "ur5e__shoulder_link")
DENSE_SPACING_M = 0.0025
DENSE_CENTER = np.array([0.0, 0.0, 0.7])
DENSE_RADIUS_M = 2.0
BALL_MARGIN_M = 0.05
HULL_SAMPLES = 900
PEDESTAL_SAMPLES = 400
SAMPLE_SEED = 0
IK_DAMPING = 0.05
IK_MAX_JOINT_STEP_RAD = 0.05
IK_FRAME_UPDATES = 80
IK_POSE_UPDATES = 600
IK_POSE_TOL_M = 1e-5
IK_ROT_TOL_RAD = 1e-4
IK_PATH_FAIL_M = 2e-3
IK_REACH_M = 1e-3
HOLD_FRAMES = 10
MARGINAL_HOME_M = 0.002
GRID_DIRECTIONS = 400
GRID_PSI_DEG = tuple(range(0, 360, 45))
GRID_SHORT_M = (0.0, 0.008)
AXIS_EXCLUSION_DEG = 20.0
TOOL_LINKS = ("mock_pruner__base", "dovetail_female_mount__body", "ur5e__wrist_3_link")
FEAS_CLEAR_MARGIN_M = 0.005
FEAS_TOF_MARGIN_M = 0.065
MINROT_CHECKED = 60
MINROT_KEEP = 3
MINROT_ARM_MARGIN_M = 0.005
STANDOFFS_M = (0.06, 0.10)
PATH_STRIDE = 3
PATCH_SIZE_PX = 13
PATCH_ELEMENTS = PATCH_SIZE_PX * PATCH_SIZE_PX
MIN_UNMASKED_PATCH_ELEMENTS = 140
CLOSURE_FIRST_PROGRESS = 1.0 / 6.0
JAW_MASK_MARGIN_PX = 2.0
STOP_WINDOW_FRAMES = (0, 4)
EXEMPT_PREDICTIONS = (35837, 35957, 36017)
EXCLUDED_FROM_SEARCH = ((0, 530), (1, 19444))
MARKER = b"@@KNOWN_MAP_ORCHARD_STREAM@@\n"

#: Recorded runs that the acceptance test replays (baseline strategy, source light, the scripts' run table).
RECORDED = {
    "530": "targets-source-20260923/run_00_source_tree0_v530",
    "7524": "targets-source-20260923/run_02_source_tree0_v7524",
    "22988": "targets-source-20260923/run_08_source_tree0_v22988",
    "18669": "targets-source-20260923/run_07_source_tree0_v18669",
    "19384": "tree1-listed-baseline-20260923/run_05_source_tree1_v19384_baseline",
    "19444": "tree1-listed-baseline-20260923/run_06_source_tree1_v19444_baseline",
    "12142": "targets-source-20260923/run_06_source_tree0_v12142",
    "14944": "tree1-listed-baseline-20260923/run_01_source_tree1_v14944_baseline",
    "15004": "tree1-listed-baseline-20260923/run_02_source_tree1_v15004_baseline",
    "14884": "tree1-listed-baseline-20260923/run_00_source_tree1_v14884_baseline",
    "8235": "lighting-20260919/run_00_source_raw",
}
DIAGNOSED = ("530", "7524", "22988", "18669", "19384", "19444", "12142")
PASSES = ("14944", "15004", "14884", "8235")
#: The refused layouts and the register entry that places each one.
REFUSALS = {"10001": ("tree0", 0), "10061": ("tree0", 0), "23167": ("tree0", 0), "19145": ("tree1_listed", 1)}
#: Every recorded stop of the scope's list, with the object the committed diagnosis names for it.
STOPS = (
    {"name": "530_contact", "target": "530", "kind": "contact", "committed_object": "tree0_BRANCH:0"},
    {"name": "7524_contact", "target": "7524", "kind": "contact", "committed_object": "tree0_SPUR(non-target):7049"},
    {"name": "22988_contact", "target": "22988", "kind": "contact", "committed_object": "tree0_SPUR(non-target):28375"},
    {
        "name": "18669_home_contact",
        "target": "18669",
        "kind": "home_contact",
        "committed_object": "tree0_SPUR(non-target):19265",
    },
    {"name": "18669_tof", "target": "18669", "kind": "tof", "committed_object": "wire0_3:0"},
    {"name": "19384_tof", "target": "19384", "kind": "tof", "committed_object": "tree1_BRANCH:6646"},
    {"name": "19444_tof", "target": "19444", "kind": "tof", "committed_object": "tree1_BRANCH:6646"},
    {"name": "12142_line_of_sight", "target": "12142", "kind": "los", "committed_object": "wire0_2"},
)
#: The committed planned orientations (eval_targets_planned_pose_2026-09-27.json) and their re-orientation.
COMMITTED_PLANS = {"530": 55.5, "19444": 83.6}
#: Recorded planned-pose runs with the jaw self-mask on, whose patch telemetry validates the projection.
JAW_PROJECTION_RUNS = {
    "530_a": "jaw-hold-a-20261001/run_00_source_tree0_v530_planned_pose_jaw_hold",
    "530_b": "jaw-hold-b-20261001/run_00_source_tree0_v530_planned_pose_jaw_hold",
    "530_c": "jaw-hold-c-20261001/run_00_source_tree0_v530_planned_pose_jaw_hold",
    "19444_a": "jaw-hold-a-20261001/run_01_source_tree1_v19444_planned_pose_jaw_hold",
}
PARTS = ("replays", "refusals", "predictions", "reorientation", "jaw_projection")

SCOPE = (
    "Known-map CPU model rebuilt for docs/SCOPE_CLOSURE_HOLD_GENERALIZATION_2026-10-02.md: a port of the September 27 "
    "contact-diagnosis scripts recovered from the session transcripts (URDF kinematics from the recorded settled home, "
    "exact convex link hulls against the exported orchard meshes, the two 8x8 ToF zones re-cast, the camera line of "
    "sight, a frame-by-frame baseline replay, a 6016-pose final-orientation search). Simulator geometry only: no "
    "dynamics, orientation drift, contact forces or perception; never a grader. Numbers and hashes only, no geometry."
)


class GateError(RuntimeError):
    """A precondition failed; nothing is written."""


# ------------------------------------------------------------------------------------------- small math
def rpy_to_R(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


def T_from(xyz, R):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = xyz
    return T


def axis_angle_R(axis, angle):
    axis = np.asarray(axis, float) / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * K @ K


def quat_wxyz_to_R(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def R_to_quat_wxyz(R):
    q = np.empty(4)
    t = np.trace(R)
    if t > 0:
        s = math.sqrt(t + 1.0) * 2
        q[:] = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        if i == 0:
            s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            q[:] = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
        elif i == 1:
            s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            q[:] = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
        else:
            s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            q[:] = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    return q / np.linalg.norm(q)


def rotation_angle_rad(Ra, Rb):
    return math.acos(float(np.clip((np.trace(Rb @ Ra.T) - 1) / 2, -1, 1)))


def slerp_R(R0, R1, t):
    dR = R1 @ R0.T
    ang = math.acos(np.clip((np.trace(dR) - 1) / 2, -1, 1))
    if ang < 1e-9:
        return R0
    ax = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * math.sin(ang))
    return axis_angle_R(ax, ang * t) @ R0


def yaw_R(degrees):
    a = math.radians(degrees)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def fib_sphere(n):
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = math.pi * (1 + 5**0.5) * i
    return np.c_[np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)]


def frame_from_z(z, psi):
    z = z / np.linalg.norm(z)
    a = np.array([0, 0, 1.0]) if abs(z[2]) < 0.9 else np.array([1.0, 0, 0])
    x = np.cross(a, z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    return np.column_stack((x, y, z)) @ axis_angle_R([0, 0, 1], psi)


def orientation_grid(spur_axis_w, n_directions=GRID_DIRECTIONS):
    """The final-pose search grid, in the scripts' order: Fibonacci directions, then roll, then mouth offset.

    Directions within AXIS_EXCLUSION_DEG of the spur axis (either sense) are left out: there the jaw closing axis,
    perpendicular to both the tool axis and the spur, is undefined. 530's axis leaves 376 directions: 6016 poses.
    """
    axis = np.asarray(spur_axis_w, float)
    rows = []
    for index, z in enumerate(fib_sphere(n_directions)):
        if math.degrees(math.acos(min(1.0, abs(float(z @ axis))))) < AXIS_EXCLUSION_DEG:
            continue
        for psi in np.radians(np.arange(0, 360, 45)):
            R = frame_from_z(z, psi)
            for short in GRID_SHORT_M:
                rows.append(
                    {"direction_index": index, "z": z, "psi_deg": round(math.degrees(psi)), "short_m": short, "R": R}
                )
    return rows


def grid_identity(R, n_directions=GRID_DIRECTIONS):
    """(direction index, roll) of a rotation on the search grid, and the residual (max |element difference|)."""
    dirs = fib_sphere(n_directions)
    index = int(np.argmax(dirs @ R[:, 2]))
    best = min((float(np.abs(frame_from_z(dirs[index], math.radians(p)) - R).max()), p) for p in GRID_PSI_DEG)
    return {"direction_index": index, "psi_deg": best[1], "residual": best[0]}


# ------------------------------------------------------------------------------------------- URDF kinematics
def parse_urdf(text):
    """Joints (origin transform, axis, type), link order and collision mesh references of a URDF string."""
    root = ET.fromstring(text)

    def origin(element):
        o = element.find("origin")
        xyz = [float(v) for v in o.get("xyz", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0.0, 0.0, 0.0]
        return T_from(xyz, rpy_to_R(*rpy))

    joints, child_joint = {}, {}
    for j in root.findall("joint"):
        ax = j.find("axis")
        rec = {
            "name": j.get("name"),
            "type": j.get("type"),
            "parent": j.find("parent").get("link"),
            "child": j.find("child").get("link"),
            "T": origin(j),
            "axis": np.array([float(v) for v in ax.get("xyz").split()] if ax is not None else [0.0, 0.0, 1.0]),
        }
        joints[rec["name"]] = rec
        child_joint[rec["child"]] = rec
    collisions = []
    for link in root.findall("link"):
        for c in link.findall("collision"):
            mesh = c.find("geometry/mesh")
            if mesh is not None:
                collisions.append((link.get("name"), mesh.get("filename"), origin(c)))
    links = [link.get("name") for link in root.findall("link")]
    return {"joints": joints, "child_joint": child_joint, "links": links, "collisions": collisions}


class Kinematics:
    """Forward kinematics, the tool point and the damped-least-squares IK of the scripts' Robot class."""

    def __init__(self, parsed, base=BASE_W, eef=EEF_LINK, tool_in_base=TOOL_IN_BASE, arm_joints=ARM_JOINTS):
        self.joints, self.child_joint, self.links = parsed["joints"], parsed["child_joint"], parsed["links"]
        roots = [link for link in self.links if link not in self.child_joint]
        if len(roots) != 1:
            raise ValueError(f"expected one root link, found {roots}")
        self.root, self.eef = roots[0], eef
        self.base = T_from(np.asarray(base, float), np.eye(3))
        self.tool_in_base = np.asarray(tool_in_base, float)
        self.arm_joints = tuple(arm_joints)
        missing = [name for name in self.arm_joints if name not in self.joints]
        if missing:
            raise ValueError(f"URDF lacks joints {missing}")
        self._chains = {}

    def _chain(self, link):
        if link not in self._chains:
            chain, cur = [], link
            while cur != self.root:
                joint = self.child_joint[cur]
                chain.append(joint)
                cur = joint["parent"]
            self._chains[link] = chain[::-1]
        return self._chains[link]

    def fk(self, q):
        """World transforms of every link for arm joints q (ARM_JOINTS order); the root sits at the base pose."""
        qd = dict(zip(self.arm_joints, q))
        cache = {self.root: self.base}
        for link in self.links:
            if link in cache:
                continue
            T = self.base
            for joint in self._chain(link):
                if joint["child"] in cache:
                    T = cache[joint["child"]]
                    continue
                T = T @ joint["T"]
                if joint["type"] == "revolute":
                    T = T @ T_from(np.zeros(3), axis_angle_R(joint["axis"], qd[joint["name"]]))
                cache[joint["child"]] = T
        return cache

    def link_transform(self, q, link):
        qd = dict(zip(self.arm_joints, q))
        T = self.base
        for joint in self._chain(link):
            T = T @ joint["T"]
            if joint["type"] == "revolute":
                T = T @ T_from(np.zeros(3), axis_angle_R(joint["axis"], qd[joint["name"]]))
        return T

    def tool_pose(self, q):
        T = self.link_transform(q, self.eef)
        return T[:3, :3] @ self.tool_in_base + T[:3, 3], T[:3, :3]

    def jacobian_tool(self, q, eps=1e-6):
        p0, R0 = self.tool_pose(q)
        J = np.zeros((6, len(self.arm_joints)))
        for i in range(len(self.arm_joints)):
            dq = np.array(q, float).copy()
            dq[i] += eps
            p, R = self.tool_pose(dq)
            J[:3, i] = (p - p0) / eps
            dR = R @ R0.T
            J[3:, i] = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * eps)
        return J

    def ik_step(self, q, p_des, R_des, damping=IK_DAMPING, max_dq=IK_MAX_JOINT_STEP_RAD):
        """One SVD damped-least-squares update with the uniform joint trust region of
        control_diagnostics.bounded_damped_joint_delta; returns (q, position error, rotation error) before it."""
        p, R = self.tool_pose(q)
        e_p = p_des - p
        dR = R_des @ R.T
        angle = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2)))
        if angle < 1e-9:
            e_r = np.zeros(3)
        else:
            vee = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]])
            e_r = angle / (2 * math.sin(angle)) * vee
        J = self.jacobian_tool(q)
        u, s, vh = np.linalg.svd(J, full_matrices=J.shape[0] == J.shape[1])
        gain = s / (s * s + damping**2)
        dq = vh.T @ (gain * (u.T @ np.r_[e_p, e_r]))
        scale = min(1.0, max_dq / max(np.abs(dq).max(), 1e-12))
        return np.asarray(q) + dq * scale, float(np.linalg.norm(e_p)), angle

    def ik(self, q0, p_des, R_des, iters=200, tol=IK_POSE_TOL_M):
        q = np.array(q0, float)
        for _ in range(iters):
            q, ep, er = self.ik_step(q, p_des, R_des)
            if ep < tol and er < IK_ROT_TOL_RAD:
                break
        p, _ = self.tool_pose(q)
        return q, float(np.linalg.norm(p - p_des))


# ------------------------------------------------------------------------------------------- geometry kernels
def _dot(a, b):
    return np.einsum("...i,...i->...", a, b)


def closest_point_on_triangle(p, a, b, c):
    """Closest points of points p to triangles (a, b, c), broadcast over leading axes (Ericson's region test)."""
    p, a, b, c = (np.asarray(x, float) for x in (p, a, b, c))
    ab, ac = b - a, c - a
    ap, bp, cp = p - a, p - b, p - c
    d1, d2, d3, d4, d5, d6 = _dot(ab, ap), _dot(ac, ap), _dot(ab, bp), _dot(ac, bp), _dot(ab, cp), _dot(ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    with np.errstate(divide="ignore", invalid="ignore"):
        denom = va + vb + vc
        v, w = vb / denom, vc / denom
        out = a + ab * v[..., None] + ac * w[..., None]
        w_bc = (d4 - d3) / ((d4 - d3) + (d5 - d6))
        out = np.where(((va <= 0) & (d4 - d3 >= 0) & (d5 - d6 >= 0))[..., None], b + (c - b) * w_bc[..., None], out)
        out = np.where(((vb <= 0) & (d2 >= 0) & (d6 <= 0))[..., None], a + ac * (d2 / (d2 - d6))[..., None], out)
        out = np.where(((d6 >= 0) & (d5 <= d6))[..., None], c, out)
        out = np.where(((vc <= 0) & (d1 >= 0) & (d3 <= 0))[..., None], a + ab * (d1 / (d1 - d3))[..., None], out)
        out = np.where(((d3 >= 0) & (d4 <= d3))[..., None], b, out)
        out = np.where(((d1 <= 0) & (d2 <= 0))[..., None], a, out)
    return out


def ray_triangle_t(o, d, a, b, c, eps=1e-12):
    """Möller-Trumbore hit distance of rays (o, d) against triangles, broadcast; inf on a miss."""
    e1, e2 = b - a, c - a
    pvec = np.cross(d, e2)
    det = _dot(e1, pvec)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / det
        tvec = o - a
        u = _dot(tvec, pvec) * inv
        qvec = np.cross(tvec, e1)
        v = _dot(d, qvec) * inv
        t = _dot(e2, qvec) * inv
        hit = (np.abs(det) > eps) & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 0)
    return np.where(hit, t, np.inf)


def convex_planes(vertices, faces):
    """Outward unit normals and offsets of a closed convex triangle mesh (n . x <= offset inside)."""
    V, F = np.asarray(vertices, float), np.asarray(faces, int)
    tri = V[F]
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    keep = np.linalg.norm(n, axis=1) > 1e-15
    n, tri = n[keep], tri[keep]
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    centre = V.mean(axis=0)
    off = _dot(n, tri[:, 0])
    flip = _dot(n, np.broadcast_to(centre, n.shape)) > off
    n[flip] *= -1
    off[flip] *= -1
    return n, off


def halfspace_depth(points, normals, offsets):
    """Depth of points inside a convex polytope (min over faces of offset - n.x); negative outside."""
    return (offsets[None, :] - np.asarray(points, float) @ normals.T).min(axis=1)


def brute_force_hull(points):
    """Convex hull of a few points by facet enumeration (tests and tiny inputs only); outward triangles."""
    P = np.asarray(points, float)
    faces = []
    n = len(P)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                normal = np.cross(P[j] - P[i], P[k] - P[i])
                if np.linalg.norm(normal) < 1e-12:
                    continue
                side = (P - P[i]) @ normal
                if (side <= 1e-12).all():
                    faces.append((i, j, k))
                elif (side >= -1e-12).all():
                    faces.append((i, k, j))
    return P, np.asarray(faces, dtype=np.int64).reshape(-1, 3)


class NumpyMeshIndex:
    """Brute-force closest point, distance, convex signed distance and ray cast over a few triangle meshes.

    The reference backend for tests and small inputs; signed distance assumes a closed convex mesh, which is the
    only use (the robot's link hulls)."""

    def __init__(self, meshes):
        tris, owner, prim = [], [], []
        self.planes = []
        for k, (V, F) in enumerate(meshes):
            V, F = np.asarray(V, float), np.asarray(F, int).reshape(-1, 3)
            tris.append(V[F])
            owner.append(np.full(len(F), k))
            prim.append(np.arange(len(F)))
            self.planes.append(convex_planes(V, F) if len(F) else None)
        self.tri = np.concatenate(tris) if tris else np.zeros((0, 3, 3))
        self.owner = np.concatenate(owner) if owner else np.zeros(0, int)
        self.prim = np.concatenate(prim) if prim else np.zeros(0, int)

    def closest(self, points):
        P = np.asarray(points, float)
        cp = closest_point_on_triangle(P[:, None, :], self.tri[None, :, 0], self.tri[None, :, 1], self.tri[None, :, 2])
        d = np.linalg.norm(cp - P[:, None, :], axis=2)
        j = np.argmin(d, axis=1)
        rows = np.arange(len(P))
        return d[rows, j], self.owner[j], self.prim[j]

    def distance(self, points):
        return self.closest(points)[0]

    def signed_distance(self, points):
        d = self.distance(points)
        inside = halfspace_depth(points, *self.planes[0]) > 0
        return np.where(inside, -d, d)

    def cast(self, origins, dirs):
        orig, D = np.asarray(origins, float), np.asarray(dirs, float)
        t = ray_triangle_t(orig[:, None], D[:, None], self.tri[None, :, 0], self.tri[None, :, 1], self.tri[None, :, 2])
        j = np.argmin(t, axis=1)
        rows = np.arange(len(orig))
        th = t[rows, j]
        hit = np.isfinite(th)
        return th, np.where(hit, self.owner[j], -1), np.where(hit, self.prim[j], -1)


class Open3dMeshIndex:
    """The scripts' open3d RaycastingScene queries (float32, as they ran)."""

    def __init__(self, meshes):
        import open3d as o3d

        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        self.lookup = {}
        for k, (V, F) in enumerate(meshes):
            g = self.scene.add_triangles(
                o3d.core.Tensor(np.asarray(V, np.float32)), o3d.core.Tensor(np.asarray(F, np.uint32))
            )
            self.lookup[int(g)] = k

    def _owner(self, gids):
        return np.array([self.lookup.get(int(g), -1) for g in gids], dtype=np.int64)

    def closest(self, points):
        pts = np.asarray(points, np.float32)
        ans = self.scene.compute_closest_points(self.o3d.core.Tensor(pts))
        d = np.linalg.norm(ans["points"].numpy() - pts, axis=1)
        return d, self._owner(ans["geometry_ids"].numpy()), ans["primitive_ids"].numpy().astype(np.int64)

    def distance(self, points):
        return self.scene.compute_distance(self.o3d.core.Tensor(np.asarray(points, np.float32))).numpy()

    def signed_distance(self, points):
        return self.scene.compute_signed_distance(self.o3d.core.Tensor(np.asarray(points, np.float32))).numpy()

    def cast(self, origins, dirs):
        rays = np.c_[origins, dirs].astype(np.float32)
        ans = self.scene.cast_rays(self.o3d.core.Tensor(rays))
        t = ans["t_hit"].numpy()
        return t, self._owner(ans["geometry_ids"].numpy()), ans["primitive_ids"].numpy().astype(np.int64)


class NumpyBall:
    def __init__(self, points):
        self.points = np.asarray(points, float)

    def query(self, centre, radius):
        return np.flatnonzero(np.linalg.norm(self.points - centre, axis=1) <= radius)


class KDBall:
    def __init__(self, points):
        from scipy.spatial import cKDTree

        self.tree = cKDTree(points)

    def query(self, centre, radius):
        return np.asarray(self.tree.query_ball_point(centre, radius), dtype=np.int64)


def trimesh_hull(points):
    import trimesh

    hull = trimesh.Trimesh(np.asarray(points, float), process=False).convex_hull
    return np.asarray(hull.vertices), np.asarray(hull.faces)


class Backend:
    """Factories for mesh queries, ball queries and convex hulls: open3d/scipy/trimesh, or numpy for tests."""

    def __init__(self, index=Open3dMeshIndex, ball=KDBall, hull=trimesh_hull):
        self.index, self.ball, self.hull = index, ball, hull

    @classmethod
    def numpy(cls):
        return cls(index=NumpyMeshIndex, ball=NumpyBall, hull=lambda pts: brute_force_hull(pts))


# ------------------------------------------------------------------------------------------- orchard
def mesh_label(name):
    """The scripts' short label of an export mesh key: the prim under the root (tree0_SPUR, wire0_3, post2)."""
    return name.split(":")[1].split("/")[2] if name.startswith("tree") else name.split("/")[2]


def densify(meshes, center=DENSE_CENTER, radius=DENSE_RADIUS_M, spacing=DENSE_SPACING_M, seed=SAMPLE_SEED):
    """Dense surface samples (about 2 per spacing^2, plus the vertices) within radius of the robot base."""
    rng = np.random.default_rng(seed)
    pts, owner = [], []
    for k, (_label, V, F, _C) in enumerate(meshes):
        tri = V[F]
        keep = np.linalg.norm(tri.mean(1) - center, axis=1) < radius + 0.5
        tri = tri[keep]
        if not len(tri):
            continue
        a = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
        n = np.maximum(1, np.ceil(a / spacing**2 * 2.0)).astype(int)
        idx = np.repeat(np.arange(len(tri)), n)
        r1 = np.sqrt(rng.random(len(idx)))
        r2 = rng.random(len(idx))
        t = tri[idx]
        p = (1 - r1)[:, None] * t[:, 0] + (r1 * (1 - r2))[:, None] * t[:, 1] + (r1 * r2)[:, None] * t[:, 2]
        p = p[np.linalg.norm(p - center, axis=1) < radius]
        pts.append(np.r_[p, V[np.linalg.norm(V - center, axis=1) < radius]])
        owner.append(np.full(len(pts[-1]), k))
    return np.concatenate(pts), np.concatenate(owner)


class Orchard:
    """The placed orchard: labelled triangle meshes, the selected spur as a convex hull, dense samples, queries."""

    TARGET_LABEL = "TARGET_SPUR_hull"

    def __init__(self, export, translation, rotation, target_tree, target_vertices, backend=None, dense=True):
        backend = backend or Backend()
        names = [str(n) for n in export["names"]]
        target_vertices = np.asarray(target_vertices, dtype=np.int64)
        spur = f"tree{int(target_tree)}_SPUR"
        if sum(mesh_label(n) == spur for n in names) != 1:
            raise GateError(f"expected exactly one {spur} mesh in the export")
        self.meshes = []
        for i, n in enumerate(names):
            V = np.asarray(export[f"v{i}"], float) @ rotation.T + translation
            F = np.asarray(export[f"t{i}"], np.int64)
            C = np.asarray(export[f"c{i}"])
            short = mesh_label(n)
            if short == spur:
                tv = np.zeros(len(V), bool)
                tv[target_vertices] = True
                ftarget = tv[F].all(axis=1)
                self.meshes.append((f"{short}(non-target)", V, F[~ftarget], C))
                hv, hf = backend.hull(V[tv])
                self.meshes.append(
                    (self.TARGET_LABEL, np.asarray(hv, float), np.asarray(hf, np.int64), np.zeros(len(hv), int))
                )
            else:
                self.meshes.append((short, V, F, C))
        self.target_k = [k for k, m in enumerate(self.meshes) if m[0] == self.TARGET_LABEL][0]
        self.index = backend.index([(m[1], m[2]) for m in self.meshes])
        self.nt_map = [k for k in range(len(self.meshes)) if k != self.target_k]
        self.index_nt = backend.index([(self.meshes[k][1], self.meshes[k][2]) for k in self.nt_map])
        self.index_t = backend.index([(self.meshes[self.target_k][1], self.meshes[self.target_k][2])])
        self._first = {}
        if dense:
            self.dense, self.dense_owner = densify(self.meshes)
            self.ball = backend.ball(self.dense)

    def _component_first(self, k, vertex):
        label, _V, _F, C = self.meshes[k]
        if label == self.TARGET_LABEL:
            return -1
        key = (k, int(C[vertex]))
        if key not in self._first:
            self._first[key] = int(np.flatnonzero(C[vertex] == C)[0])
        return self._first[key]

    def describe(self, k, prim):
        """(label, component first vertex) of triangle prim of mesh k; -1 for the selected spur's hull."""
        k = int(k)
        label, _V, F, _C = self.meshes[k]
        return label, self._component_first(k, F[int(prim)][0])

    def describe_nt(self, g, prim):
        return self.describe(self.nt_map[int(g)], prim)

    def cast(self, origins, dirs):
        return self.index.cast(origins, dirs)


def placement_from_report(report):
    b = report["blender_scene"]
    return np.asarray(b["translation_w_m"], float), yaw_R(b["yaw_degrees"])


def placement_from_component(source_center, yaw_degrees=ORCHARD_YAW_DEG, target=TARGET_W):
    """spawn_blender_demo_scene: the spur's centre lands on the fixed target point after the orchard yaw."""
    R = yaw_R(yaw_degrees)
    return np.asarray(target, float) - R @ np.asarray(source_center, float), R


# ------------------------------------------------------------------------------------------- robot model
class RobotModel:
    """Kinematics plus each link's convex collision hull, its surface samples, ray scene and bounding sphere."""

    def __init__(self, kin, hulls, sampler, backend=None):
        backend = backend or Backend()
        self.kin = kin
        self.hulls = {link: (np.asarray(V, float), np.asarray(F, np.int64)) for link, (V, F) in hulls.items()}
        self.links = list(self.hulls)
        self._sampler = sampler
        self._samples = {}
        self.hull_index = {link: backend.index([mesh]) for link, mesh in self.hulls.items()}
        self.sphere = {}
        for link, (V, _F) in self.hulls.items():
            c = (V.min(0) + V.max(0)) / 2
            self.sphere[link] = (c, float(np.linalg.norm(V - c, axis=1).max()))

    def local_points(self, link, count):
        key = (link, count)
        if key not in self._samples:
            self._samples[key] = np.asarray(self._sampler(link, count), float)
        return self._samples[key]


def load_robot(urdf_path, backend=None):
    """Exact convex hulls of the URDF collision meshes (PhysX approximation convexHull), sampled as the scripts did."""
    import trimesh

    parsed = parse_urdf(Path(urdf_path).read_text())
    meshes, hashes = {}, {}
    for link, filename, Tc in parsed["collisions"]:
        path = filename[len("file://") :] if filename.startswith("file://") else filename
        hashes[path] = sha256(path)
        hull = trimesh.load(path, force="mesh").convex_hull
        v = (np.c_[hull.vertices, np.ones(len(hull.vertices))] @ Tc.T)[:, :3]
        meshes[link] = trimesh.Trimesh(v, hull.faces, process=True)

    def sampler(link, count):
        pts, _ = trimesh.sample.sample_surface_even(meshes[link], count, seed=SAMPLE_SEED)
        return np.r_[pts, meshes[link].vertices]

    hulls = {link: (np.asarray(m.vertices), np.asarray(m.faces)) for link, m in meshes.items()}
    return RobotModel(Kinematics(parsed), hulls, sampler, backend), hashes


def link_clearance(robot, orch, link, T, n_samples=HULL_SAMPLES, margin=BALL_MARGIN_M):
    """The scripts' clearance2 for one posed link hull: signed clearance to non-target geometry and to the target.

    (a) hull surface samples to non-target triangles (unsigned) and to the selected spur's hull; (b) dense orchard
    samples near the link to the hull (signed, negative inside). The smaller wins; its object is named."""
    pts = (robot.local_points(link, n_samples) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
    d, g, prim = orch.index_nt.closest(pts)
    i = int(np.argmin(d))
    best = float(d[i])
    label, first = orch.describe_nt(g[i], prim[i])
    dt = float(np.min(orch.index_t.distance(pts)))
    c, r = robot.sphere[link]
    idx = orch.ball.query(T[:3, :3] @ c + T[:3, 3], r + margin)
    if len(idx):
        idx = np.asarray(idx)
        local = (orch.dense[idx] - T[:3, 3]) @ T[:3, :3]
        sd = np.asarray(robot.hull_index[link].signed_distance(local.astype(np.float32)), float)
        own = orch.dense_owner[idx]
        is_t = own == orch.target_k
        if is_t.any():
            dt = min(dt, float(sd[is_t].min()))
        if (~is_t).any():
            sdn = np.where(is_t, np.inf, sd)
            j = int(np.argmin(sdn))
            if sdn[j] < best:
                best = float(sdn[j])
                _d2, g2, p2 = orch.index_nt.closest(orch.dense[idx[j]][None].astype(np.float32))
                label, first = orch.describe_nt(g2[0], p2[0])
    return {"min_m": best, "nearest": label, "component_first_vertex": first, "target_m": dt}


def clearance2(robot, orch, q, links=None):
    fk = robot.kin.fk(q)
    return {link: link_clearance(robot, orch, link, fk[link]) for link in (links or robot.links)}


def pedestal_ground(robot, q):
    """Min clearance of non-base link hull samples to the pedestal (r 0.14 m, top 0.70 m) and the ground plane."""
    fk = robot.kin.fk(q)
    worst = (np.inf, None)
    for link in robot.links:
        if link in PEDESTAL_SKIP:
            continue
        T = fk[link]
        p = robot.local_points(link, PEDESTAL_SAMPLES) @ T[:3, :3].T + T[:3, 3]
        r = np.linalg.norm(p[:, :2], axis=1)
        dz, dr = np.maximum(p[:, 2] - PEDESTAL_TOP_M, 0), np.maximum(r - PEDESTAL_R_M, 0)
        inside = (p[:, 2] <= PEDESTAL_TOP_M) & (r <= PEDESTAL_R_M)
        dped = np.where(inside, -np.minimum(PEDESTAL_TOP_M - p[:, 2], PEDESTAL_R_M - r), np.hypot(dz, dr))
        m = min(float(p[:, 2].min()), float(dped.min()))
        if m < worst[0]:
            worst = (m, link)
    return worst


# ------------------------------------------------------------------------------------------- sensors and camera
def tof_directions(fov_deg=TOF_DIAGONAL_FOV_DEG):
    f = math.hypot(8, 8) / (2 * math.tan(math.radians(fov_deg) / 2))
    d = np.array([[(u + 0.5 - 4) / f, (v + 0.5 - 4) / f, 1.0] for v in range(8) for u in range(8)])
    return d / np.linalg.norm(d, axis=1, keepdims=True)


TOF_DIRS = tof_directions()


def gap_aligned_camera_mount(closing_axis_tool, radial_distance_m=0.14, z_offset_m=-0.025):
    """blender_demo_scene.gap_aligned_camera_mount (copied by the scripts; checked against it at run time)."""
    axis = np.asarray(closing_axis_tool, dtype=float)
    lateral = np.cross([0, 0, 1], axis)
    lateral /= np.linalg.norm(lateral)
    if lateral[1] > 0 or (abs(lateral[1]) < 1e-8 and lateral[0] > 0):
        lateral *= -1
    offset = lateral * radial_distance_m
    offset[2] = z_offset_m
    return offset


def optical_rotation_from_forward(direction):
    forward = np.asarray(direction, float) / np.linalg.norm(direction)
    right = np.cross(np.array([0.0, 1.0, 0.0]), forward)
    right /= np.linalg.norm(right)
    return np.column_stack((right, np.cross(forward, right), forward))


def camera_for_home(p0, R0, target, target_axis):
    """The renderer's wrist mount for a home tool pose: gap-aligned to the home closing axis, aimed at the target."""
    closing_w = np.cross(R0[:, 2], np.asarray(target_axis))
    closing_w /= np.linalg.norm(closing_w)
    mount = gap_aligned_camera_mount(R0.T @ closing_w)
    tgt_tool = R0.T @ (np.asarray(target) - p0)
    tgt_tool[2] = 0.20
    return mount, optical_rotation_from_forward(tgt_tool - mount)


class Scene:
    """One placed layout: robot, orchard, target, spur axis and the settled home (the scripts' Env)."""

    def __init__(self, robot, orch, target, axis, q_home):
        self.robot, self.orch = robot, orch
        self.target = np.asarray(target, float)
        self.axis = np.asarray(axis, float)
        self.q_home = np.asarray(q_home, float)

    def tof(self, Tb):
        out = {}
        for name, off in TOF_IN_BASE.items():
            o = Tb[:3, :3] @ off + Tb[:3, 3]
            t, g, p = self.orch.cast(np.repeat(o[None], 64, 0), TOF_DIRS @ Tb[:3, :3].T)
            valid = np.isfinite(t) & (t >= TOF_RANGE_M[0]) & (t <= TOF_RANGE_M[1])
            if valid.any():
                j = int(np.flatnonzero(valid)[np.argmin(t[valid])])
                label = self.orch.describe(g[j], p[j])
                out[name] = (float(t[j]), f"{label[0]}:{label[1]}")
            else:
                out[name] = (float("inf"), None)
            out[name + "_n_below_0p03"] = int((np.isfinite(t) & (t < TOF_RANGE_M[0])).sum())
        return out

    def camera(self, Tb, mount, cam_rot):
        R = Tb[:3, :3]
        p_tool = R @ TOOL_IN_BASE + Tb[:3, 3]
        cam = p_tool + R @ mount
        opt = (R @ cam_rot).T @ (self.target - cam)
        px = (K_CAM @ opt)[:2] / opt[2] if opt[2] > 0 else np.array([np.nan, np.nan])
        dist = float(np.linalg.norm(self.target - cam))
        t, g, pr = self.orch.cast(cam[None], ((self.target - cam) / dist)[None])
        blocked = bool(np.isfinite(t[0]) and int(g[0]) != self.orch.target_k and t[0] < dist - LOS_MARGIN_M)
        return {
            "cam_in_fov": bool(opt[2] > 0 and 0 <= px[0] < IMAGE_W and 0 <= px[1] < IMAGE_H),
            "los_blocked": blocked,
            "los_first": self.orch.describe(g[0], pr[0])[0] if np.isfinite(t[0]) else None,
            "mouth_to_target": float(np.linalg.norm(p_tool + R @ MOUTH_IN_TOOL - self.target)),
        }

    def frame(self, q, mount, cam_rot, links=True):
        fk = self.robot.kin.fk(q)
        Tb = fk[EEF_LINK]
        out = {}
        if links:
            res = {link: link_clearance(self.robot, self.orch, link, fk[link]) for link in self.robot.links}
            link = min(res, key=lambda k: res[k]["min_m"])
            out.update(
                min_clear=res[link]["min_m"],
                min_link=link,
                min_obj=f"{res[link]['nearest']}:{res[link]['component_first_vertex']}",
            )
            tl = min(res, key=lambda k: res[k]["target_m"])
            out.update(target_clear=res[tl]["target_m"], target_link=tl)
            pg = pedestal_ground(self.robot, q)
            out.update(pedestal_ground=pg[0], pedestal_ground_link=pg[1])
        tof = self.tof(Tb)
        out["tof"] = tof
        out["tof_min"] = min(tof["tof0"][0], tof["tof1"][0])
        out.update(self.camera(Tb, mount, cam_rot))
        return out


# ------------------------------------------------------------------------------------------- paths
def mouth_line(p_start_tool, R, mouth_goal, step=STEP_M):
    """Tool positions moving the mouth straight to mouth_goal in ceil(L / step) equal steps (orientation R held)."""
    m0 = p_start_tool + R @ MOUTH_IN_TOOL
    d = mouth_goal - m0
    n = max(1, int(math.ceil(float(np.linalg.norm(d)) / step)))
    return [p_start_tool + d * (k / n) for k in range(1, n + 1)]


def straight_path(scene):
    p0, R0 = scene.robot.kin.tool_pose(scene.q_home)
    approach = mouth_line(p0, R0, scene.target)
    return approach, list(reversed(approach[:-1])) + [p0], R0


def run_path(scene, approach, retreat, R, stride=1, ik_updates=IK_FRAME_UPDATES, early_stop=False):
    """The scripts' run_path: per frame IK (warm started), links, pedestal, ToF and camera; first events and worsts."""
    kin = scene.robot.kin
    p0, _ = kin.tool_pose(scene.q_home)
    mount, cam_rot = camera_for_home(p0, R, scene.target, scene.axis)
    seq = [("approach", p) for p in approach] + [("retreat", p) for p in retreat]
    q = np.array(scene.q_home, float)
    first, rows = {}, []
    worst = {"approach": (np.inf, None, None, None), "retreat": (np.inf, None, None, None)}
    tofw = {"approach": (np.inf, None, None), "retreat": (np.inf, None, None)}
    tworst = {"approach": (np.inf, None, None), "retreat": (np.inf, None, None)}
    for k, (phase, p) in enumerate(seq):
        q, err = kin.ik(q, p, R, iters=ik_updates)
        last_approach = phase == "approach" and k == len(approach) - 1
        f = scene.frame(q, mount, cam_rot, links=(k % stride == 0) or last_approach or k == len(seq) - 1)
        f.update(k=k, phase=phase, ik_err=err)
        rows.append(f)
        if "min_clear" in f:
            c = min(f["min_clear"], f["pedestal_ground"])
            if c < worst[phase][0]:
                ped = f["min_clear"] > f["pedestal_ground"]
                worst[phase] = (
                    c,
                    f["pedestal_ground_link"] if ped else f["min_link"],
                    "pedestal/ground" if ped else f["min_obj"],
                    k,
                )
            if c <= CONTACT_M and "contact" not in first:
                first["contact"] = (k, phase, worst[phase][1], worst[phase][2], round(c, 4))
            if f["target_clear"] < tworst[phase][0]:
                tworst[phase] = (f["target_clear"], f["target_link"], k)
            if f["target_clear"] <= CONTACT_M and "target_contact" not in first:
                first["target_contact"] = (k, phase, f["target_link"], round(f["target_clear"], 4))
        if f["tof_min"] < tofw[phase][0]:
            tofw[phase] = (f["tof_min"], f["tof"], k)
        if f["tof_min"] < MIN_CLEAR_TOF_M and "tof" not in first:
            first["tof"] = (k, phase, round(f["tof_min"], 4), f["tof"])
        if phase == "approach" and f["los_blocked"] and "los" not in first:
            first["los"] = (k, f["los_first"])
        if phase == "approach" and not f["cam_in_fov"] and "fov" not in first:
            first["fov"] = k
        if early_stop and {"contact", "tof", "target_contact"} & set(first):
            break
    appr = [r for r in rows if r["phase"] == "approach"]
    summary = {
        "frames_needed": len(approach) + HOLD_FRAMES + len(retreat),
        "approach_frames": len(approach),
        "first": first,
        "worst_clear_approach": worst["approach"],
        "worst_clear_retreat": worst["retreat"],
        "worst_tof_approach": tofw["approach"],
        "worst_tof_retreat": tofw["retreat"],
        "worst_target_clear_approach": tworst["approach"],
        "final_mouth_to_target": appr[-1]["mouth_to_target"] if appr else None,
        "max_ik_err": max(r["ik_err"] for r in rows),
        "los_blocked_frames": sum(r["los_blocked"] for r in appr),
        "fov_out_frames": sum(not r["cam_in_fov"] for r in appr),
        "camera_mount_tool": [round(float(v), 6) for v in mount],
        "completed": len(rows) == len(seq),
        "clear_no_margin": bool(len(rows) == len(seq) and not {"contact", "tof", "target_contact"} & set(first)),
    }
    return summary, rows


def planned_legs(scene, Rf, short, standoff):
    """cand_feaspath: rotate (<= 1.5 deg) while moving (<= 4 mm) to a standoff along the final tool axis, then in."""
    p0, R0 = scene.robot.kin.tool_pose(scene.q_home)
    z = Rf[:, 2]
    goal_mouth = scene.target - short * z
    Sm = goal_mouth - standoff * z
    St = Sm - Rf @ MOUTH_IN_TOOL
    ang = rotation_angle_rad(R0, Rf)
    n1 = max(int(math.ceil(np.linalg.norm(St - p0) / STEP_M)), int(math.ceil(ang / MAX_ROTATION_RAD)), 1)
    leg1 = [(p0 + (St - p0) * (k / n1), slerp_R(R0, Rf, k / n1)) for k in range(1, n1 + 1)]
    n2 = int(math.ceil(standoff / STEP_M))
    leg2 = [(St + (goal_mouth - Sm) * (k / n2), Rf) for k in range(1, n2 + 1)]
    approach = leg1 + leg2
    return approach, list(reversed(approach[:-1])) + [(p0, R0)], math.degrees(ang), n1, n2


def solve_poses(scene, approach, ik_updates=IK_FRAME_UPDATES):
    """Tool poses the replay's IK reaches along a planned approach (warm started from home); no collision queries."""
    kin = scene.robot.kin
    q = np.array(scene.q_home, float)
    poses = []
    for p, R in approach:
        q, _err = kin.ik(q, p, R, iters=ik_updates)
        poses.append(kin.tool_pose(q))
    return poses


def run_planned(scene, approach, retreat, stride=PATH_STRIDE, ik_updates=IK_FRAME_UPDATES):
    """cand_aligned.run2: the planned path with the home camera mount; stops at contact, ToF, target contact or IK."""
    kin = scene.robot.kin
    p0, R0 = kin.tool_pose(scene.q_home)
    mount, cam_rot = camera_for_home(p0, R0, scene.target, scene.axis)
    seq = [("approach", *x) for x in approach] + [("retreat", *x) for x in retreat]
    q = np.array(scene.q_home, float)
    first, worst, wobj = {}, {"approach": np.inf, "retreat": np.inf}, {}
    tworst, tofw, los, fov = np.inf, np.inf, 0, 0
    stop = {"contact", "tof", "target_contact", "ik"}
    k = -1
    for k, (ph, p, R) in enumerate(seq):
        q, err = kin.ik(q, p, R, iters=ik_updates)
        if err > IK_PATH_FAIL_M and "ik" not in first:
            first["ik"] = (k, err)
        f = scene.frame(q, mount, cam_rot, links=(k % stride == 0) or k == len(approach) - 1 or k == len(seq) - 1)
        if "min_clear" in f:
            c = min(f["min_clear"], f["pedestal_ground"])
            if c < worst[ph]:
                worst[ph] = c
                wobj[ph] = (f["min_link"], f["min_obj"], k)
            tworst = min(tworst, f["target_clear"])
            if c <= CONTACT_M and "contact" not in first:
                first["contact"] = (k, ph, f["min_link"], f["min_obj"], round(c, 4))
            if f["target_clear"] <= CONTACT_M and "target_contact" not in first:
                first["target_contact"] = (k, ph, f["target_link"])
        tofw = min(tofw, f["tof_min"])
        if f["tof_min"] < MIN_CLEAR_TOF_M and "tof" not in first:
            first["tof"] = (k, ph, round(f["tof_min"], 4), f["tof"])
        if ph == "approach":
            los += f["los_blocked"]
            fov += not f["cam_in_fov"]
            if f["los_blocked"] and "los" not in first:
                first["los"] = (k, f["los_first"])
            if not f["cam_in_fov"] and "fov" not in first:
                first["fov"] = k
        if stop & set(first):
            break
    done = k == len(seq) - 1 and not (stop & set(first))
    out = {
        "first": first,
        "worst_clear": {p: (worst[p], wobj.get(p)) for p in worst},
        "worst_target": tworst,
        "worst_tof": tofw,
        "frames_needed": len(approach) + HOLD_FRAMES + len(retreat),
        "completed_clear": done,
        "clear_margin_1cm": bool(done and worst["approach"] >= 0.01 and worst["retreat"] >= 0.01 and tofw >= 0.07),
        "los_blocked_frames": int(los),
        "fov_out_frames": int(fov),
        "camera_mount_tool": [round(float(v), 6) for v in mount],
    }
    return out


# ------------------------------------------------------------------------------------------- final-pose search
def tool_side_eval(scene, Tb, rel):
    """Signed clearance of the tool-side hulls (non-target and target) and the nearest valid ToF zone at a pose."""
    orch, robot = scene.orch, scene.robot
    worst, wl, wo, tw = np.inf, None, None, np.inf
    for link in TOOL_LINKS:
        T = Tb @ rel[link]
        pts = (robot.local_points(link, HULL_SAMPLES) @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        d, g, prim = orch.index_nt.closest(pts)
        i = int(np.argmin(d))
        best = float(d[i])
        obj = orch.describe_nt(g[i], prim[i])
        tw = min(tw, float(np.min(orch.index_t.distance(pts))))
        c, r = robot.sphere[link]
        idx = orch.ball.query(T[:3, :3] @ c + T[:3, 3], r + BALL_MARGIN_M)
        if len(idx):
            idx = np.asarray(idx)
            local = (orch.dense[idx] - T[:3, 3]) @ T[:3, :3]
            sd = np.asarray(robot.hull_index[link].signed_distance(local.astype(np.float32)), float)
            own = orch.dense_owner[idx]
            is_t = own == orch.target_k
            if is_t.any():
                tw = min(tw, float(sd[is_t].min()))
            sdn = np.where(is_t, np.inf, sd)
            j = int(np.argmin(sdn))
            if sdn[j] < best:
                best = float(sdn[j])
                obj = (orch.meshes[int(own[j])][0], None)
        if best < worst:
            worst, wl, wo = best, link, obj
    tof, tobj = np.inf, None
    for name, off in TOF_IN_BASE.items():
        o = Tb[:3, :3] @ off + Tb[:3, 3]
        t, g, p = orch.cast(np.repeat(o[None], 64, 0), TOF_DIRS @ Tb[:3, :3].T)
        v = np.isfinite(t) & (t >= TOF_RANGE_M[0]) & (t <= TOF_RANGE_M[1])
        if v.any():
            j = int(np.flatnonzero(v)[np.argmin(t[v])])
            if t[j] < tof:
                tof, tobj = float(t[j]), orch.describe(g[j], p[j])
    return worst, wl, wo, tw, tof, tobj


def final_pose_search(scene, census=False, log=None):
    """feasibility.py (as edited on September 27): the smallest-re-orientation reachable final poses.

    Margin-feasible poses (tool-side clearance >= 5 mm to non-target geometry, > 0 to the selected spur, ToF >= 65 mm)
    are taken in order of tool-axis angle from home; each is solved by IK from home (600 updates), must reach within
    1 mm with |elbow| <= pi, and is kept when the whole arm clears the orchard by > 5 mm and the pedestal and ground;
    the search stops at 3 kept poses or after 60 checked. The census evaluates all poses first (same list)."""
    kin = scene.robot.kin
    fkh = kin.fk(scene.q_home)
    Tbh = fkh[EEF_LINK]
    rel = {link: np.linalg.inv(Tbh) @ fkh[link] for link in TOOL_LINKS}
    p0, R0 = kin.tool_pose(scene.q_home)
    grid = orientation_grid(scene.axis)
    order = sorted(range(len(grid)), key=lambda i: float(np.arccos(np.clip(grid[i]["z"] @ R0[:, 2], -1, 1))))

    def evaluate(row):
        R = row["R"]
        mouth = scene.target - row["short_m"] * R[:, 2]
        p_tool = mouth - R @ MOUTH_IN_TOOL
        Tb = np.eye(4)
        Tb[:3, :3], Tb[:3, 3] = R, p_tool - R @ TOOL_IN_BASE
        w, wl, wo, tw, tof, tobj = tool_side_eval(scene, Tb, rel)
        row.update(clear=w, clear_link=wl, clear_obj=None if wo is None else f"{wo[0]}:{wo[1]}", target_clear=tw)
        row.update(tof=tof, tof_obj=None if tobj is None else f"{tobj[0]}:{tobj[1]}")
        row["feasible"] = bool(w > 0 and tw > 0 and tof >= MIN_CLEAR_TOF_M)
        row["margin_feasible"] = bool(row["feasible"] and w >= FEAS_CLEAR_MARGIN_M and tof >= FEAS_TOF_MARGIN_M)

    t0 = time.time()
    census_out = None
    if census:
        for row in grid:
            evaluate(row)
        census_out = {
            "n_poses": len(grid),
            "n_feasible_tool_side": sum(r["feasible"] for r in grid),
            "n_feasible_margin": sum(r["margin_feasible"] for r in grid),
        }
    shortlist, evaluated = [], 0
    for i in order:
        if "feasible" not in grid[i]:
            evaluate(grid[i])
            evaluated += 1
        if grid[i]["margin_feasible"]:
            shortlist.append(grid[i])
            if len(shortlist) >= MINROT_CHECKED:
                break
    kept, checked = [], []
    for row in shortlist:
        R = row["R"]
        p_tool = scene.target - row["short_m"] * R[:, 2] - R @ MOUTH_IN_TOOL
        q, err = kin.ik(scene.q_home, p_tool, R, iters=IK_POSE_UPDATES, tol=IK_POSE_TOL_M)
        if err > IK_REACH_M or abs(q[2]) > math.pi:
            checked.append({"direction_index": row["direction_index"], "psi_deg": row["psi_deg"], "reached": False})
            continue
        full = clearance2(scene.robot, scene.orch, q)
        fl = min(full, key=lambda k: full[k]["min_m"])
        pg = pedestal_ground(scene.robot, q)
        rec = {
            "direction_index": row["direction_index"],
            "psi_deg": row["psi_deg"],
            "short_m": row["short_m"],
            "quat_wxyz": R_to_quat_wxyz(R).tolist(),
            "R": R.tolist(),
            "tool_axis_w": R[:, 2].tolist(),
            "tool_z_vs_home_deg": math.degrees(math.acos(np.clip(R[:, 2] @ R0[:, 2], -1, 1))),
            "total_rotation_deg": math.degrees(rotation_angle_rad(R0, R)),
            "tool_clear_m": row["clear"],
            "tof_m": row["tof"],
            "full_arm_min_m": full[fl]["min_m"],
            "full_arm_link": fl,
            "full_arm_obj": f"{full[fl]['nearest']}:{full[fl]['component_first_vertex']}",
            "pedestal_ground_m": pg[0],
            "q": q.tolist(),
            "reached": True,
        }
        rec["kept"] = bool(rec["full_arm_min_m"] > MINROT_ARM_MARGIN_M and rec["pedestal_ground_m"] > 0)
        checked.append(rec)
        if rec["kept"]:
            kept.append(rec)
            if len(kept) >= MINROT_KEEP:
                break
    if log:
        log(
            f"final-pose search: {evaluated} poses evaluated lazily, {len(checked)} checked, {len(kept)} kept, "
            f"{time.time() - t0:.0f} s"
        )
    return {
        "n_grid_poses": len(grid),
        "census": census_out,
        "n_evaluated_lazily": evaluated,
        "n_margin_feasible_shortlisted": len(shortlist),
        "n_checked": len(checked),
        "checked": [{k: v for k, v in c.items() if k != "q"} for c in checked],
        "kept": kept,
    }


def plan_paths(scene, kept, ik_updates=IK_FRAME_UPDATES, log=None):
    """cand_feaspath + final_check: up to 3 kept poses x standoffs 0.06/0.10 at stride 3; clear ones at stride 1."""
    out = []
    for i, m in enumerate(kept[:MINROT_KEEP]):
        Rf = np.asarray(m["R"], float)
        for s in STANDOFFS_M:
            approach, retreat, ang, n1, n2 = planned_legs(scene, Rf, m["short_m"], s)
            r3 = run_planned(scene, approach, retreat, stride=PATH_STRIDE, ik_updates=ik_updates)
            rec = {
                "pose_index": i,
                "direction_index": m["direction_index"],
                "psi_deg": m["psi_deg"],
                "short_m": m["short_m"],
                "standoff_m": s,
                "total_rotation_deg": ang,
                "frames_leg1": n1,
                "frames_leg2": n2,
                "stride3": r3,
                "stride1": None,
            }
            if r3["completed_clear"]:
                rec["stride1"] = run_planned(scene, approach, retreat, stride=1, ik_updates=ik_updates)
            rec["clear"] = bool(r3["completed_clear"] and rec["stride1"] and rec["stride1"]["completed_clear"])
            rec["clear_with_view"] = bool(
                rec["clear"] and rec["stride1"]["los_blocked_frames"] == 0 and rec["stride1"]["fov_out_frames"] == 0
            )
            out.append(rec)
            if log:
                log(
                    f"  pose{i} s{s}: stride3 {'clear' if r3['completed_clear'] else r3['first']}"
                    f" -> clear={rec['clear']}"
                )
    return out


# ------------------------------------------------------------------------------------------- jaw self-mask projection
def _jsm():
    if str(PACKAGE) not in sys.path:
        sys.path.insert(0, str(PACKAGE))
    from isaaclab_pruning.perception import jaw_self_mask

    return jaw_self_mask


def patch_masked(mask, pixel):
    """Patch elements a mask covers, sampled with the tracker's getRectSubPix footprint (any weight counts)."""
    import cv2

    return cv2.getRectSubPix(np.asarray(mask, np.float32), (PATCH_SIZE_PX, PATCH_SIZE_PX), tuple(map(float, pixel))) > 0


def kept_elements(masked_previous, masked_current):
    return int(PATCH_ELEMENTS - np.count_nonzero(np.asarray(masked_previous) | np.asarray(masked_current)))


def jaw_patch_mask(p_tool, R, roll, progress, radius, mount, cam_rot, pixel=None, target=None):
    """(masked patch elements, tracked pixel) for the two-box jaw at a tool pose, seen by the tool-fixed camera."""
    jsm = _jsm()
    cam = p_tool + R @ mount
    Rc = R @ cam_rot
    world_from_optical = T_from(cam, Rc)
    if pixel is None:
        opt = Rc.T @ (np.asarray(target, float) - cam)
        pixel = (K_CAM @ opt)[:2] / opt[2]
    pose = np.r_[p_tool, R_to_quat_wxyz(R)]
    boxes = jsm.jaw_boxes(pose, roll, float(progress), radius)
    mask = jsm.jaw_mask(boxes, K_CAM, world_from_optical, (IMAGE_H, IMAGE_W), margin_px=JAW_MASK_MARGIN_PX)
    return patch_masked(mask, pixel), np.asarray(pixel, float)


def closing_roll(Rf, spur_axis):
    """Attachment roll of the jaw at the planned orientation (closing_axis_tool_at, then proxy_roll_rad)."""
    closing_w = np.cross(Rf[:, 2], np.asarray(spur_axis, float))
    closing_tool = Rf.T @ (closing_w / np.linalg.norm(closing_w))
    return float(np.arctan2(closing_tool[1], closing_tool[0])), closing_tool


def predicted_patch_series(scene, approach_poses, Rf, radius):
    """Kept patch elements at every predicted approach pose (open jaw, consecutive patches) and at closure start.

    The camera keeps the baseline home mount; the jaw straddles the spur at the planned orientation; the tracked
    pixel is the target's projection. Closure start = the final pose, open jaw then closure progress 1/6."""
    p0, R0 = scene.robot.kin.tool_pose(scene.q_home)
    mount, cam_rot = camera_for_home(p0, R0, scene.target, scene.axis)
    roll, _ = closing_roll(Rf, scene.axis)
    previous, _ = jaw_patch_mask(p0, R0, roll, 0.0, radius, mount, cam_rot, target=scene.target)
    kept = []
    for p, R in approach_poses:
        current, _ = jaw_patch_mask(p, R, roll, 0.0, radius, mount, cam_rot, target=scene.target)
        kept.append(kept_elements(previous, current))
        previous = current
    p, R = approach_poses[-1]
    open_mask, pixel = jaw_patch_mask(p, R, roll, 0.0, radius, mount, cam_rot, target=scene.target)
    closing, _ = jaw_patch_mask(p, R, roll, CLOSURE_FIRST_PROGRESS, radius, mount, cam_rot, pixel=pixel)
    return {"approach_kept": kept, "closure_kept": kept_elements(open_mask, closing), "closure_pixel": pixel.tolist()}


def selection_conditions(path_clear, approach_kept, closure_kept, min_kept=MIN_UNMASKED_PATCH_ELEMENTS):
    """The scope's rule: (a) a clear path; (b) >= min_kept elements at each approach pose; (c) < min_kept at closure."""
    a = bool(path_clear)
    b = bool(len(approach_kept) > 0 and min(approach_kept) >= min_kept)
    c = bool(closure_kept is not None and closure_kept < min_kept)
    return {
        "a_clear_path": a,
        "b_approach_keeps_patch": b,
        "c_closure_loses_patch": c,
        "approach_min_kept": int(min(approach_kept)) if len(approach_kept) else None,
        "closure_kept": None if closure_kept is None else int(closure_kept),
        "qualifies": a and b and c,
    }


# ------------------------------------------------------------------------------------------- inputs and outputs
def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if REPO in path.parents else str(path)


def refuse_output(path, repo_root=REPO):
    """The resolved output path; refuses the repository, any artifacts/ directory and an existing file."""
    resolved = Path(path).resolve()
    repo = Path(repo_root).resolve()
    if resolved == repo or repo in resolved.parents:
        raise ValueError(f"Refusing an output inside the repository: {resolved}")
    if "artifacts" in resolved.parts:
        raise ValueError(f"Refusing an output inside an artifacts/ directory: {resolved}")
    if resolved.exists():
        raise ValueError(f"Refusing to overwrite {resolved}")
    return resolved


def finite(value):
    """JSON-safe: non-finite floats become strings, numpy scalars and arrays become Python values."""
    if isinstance(value, dict):
        return {str(k): finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite(v) for v in value]
    if isinstance(value, np.ndarray):
        return finite(value.tolist())
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return value if math.isfinite(value) else ("inf" if value > 0 else "-inf" if value < 0 else "nan")
    return value


def write_json(path, document):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(finite(document), stream, indent=1, allow_nan=False)
        stream.write("\n")


def provenance(inputs, settings):
    tool = Path(__file__).resolve()
    status = git("status", "--porcelain", "--", relative(tool))
    return {
        "code_revision": git("rev-parse", "HEAD"),
        "code_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "tool_committed_unmodified": status == "",
        "code_sha256": {
            relative(tool): sha256(tool),
            "source/isaaclab_pruning/isaaclab_pruning/perception/jaw_self_mask.py": sha256(
                PACKAGE / "isaaclab_pruning/perception/jaw_self_mask.py"
            ),
            "source/isaaclab_pruning/isaaclab_pruning/sim/blender_component.py": sha256(
                PACKAGE / "isaaclab_pruning/sim/blender_component.py"
            ),
        },
        "inputs_sha256": dict(sorted(inputs.items())),
        "settings": settings,
        "python": sys.version.split()[0],
        "host": os.uname().nodename,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def load_json(path):
    return json.loads(Path(path).read_text())


def load_run(run_dir):
    run_dir = Path(run_dir)
    return load_json(run_dir / "report.json"), load_json(run_dir / "frames.json")["frames"]


_EXTRACT_HELPER = r"""
import io, json, sys
import numpy as np
from pxr import Usd, UsdGeom
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

export, marker = sys.argv[1], sys.argv[2].encode() + b"\n"
out, names = {}, []
for f in ("tree0.usdc", "tree1.usdc", "environment.usdc"):
    stage = Usd.Stage.Open(f"{export}/{f}")
    cache = UsdGeom.XformCache()
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        path = str(prim.GetPath())
        if "/ground/" in path:
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
        counts = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
        idx = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
        M = np.asarray(cache.GetLocalToWorldTransform(prim), dtype=np.float64)
        ptsw = (np.c_[pts, np.ones(len(pts))] @ M)[:, :3]
        tris, off = [], 0
        for c in counts:
            face = idx[off:off + c]
            for k in range(1, c - 1):
                tris.append((face[0], face[k], face[k + 1]))
            off += c
        tris = np.asarray(tris, dtype=np.int64)
        e = np.r_[tris[:, [0, 1]], tris[:, [1, 2]]]
        g = coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(pts), len(pts)))
        _, lab = connected_components(g, directed=False)
        names.append(f.split(".")[0] + ":" + path)
        i = len(names) - 1
        out[f"v{i}"], out[f"t{i}"], out[f"c{i}"] = ptsw, tris, lab
buffer = io.BytesIO()
np.savez(buffer, names=np.asarray(names), **out)
blob = buffer.getvalue()
sys.stdout.buffer.write(marker + json.dumps({"npz_bytes": len(blob)}).encode() + b"\n" + blob)
"""


def stream_export(export_dir, isaac_python, inputs):
    """The export's meshes (file-local world frame, fan triangulated, component labels), in memory only.

    The three USD files are verified against the export manifest and the pool register first; the helper runs under
    the Isaac venv python (pxr) and streams npz bytes through a pipe, so nothing holding geometry is written."""
    export_dir = Path(export_dir).resolve()
    manifest = load_json(export_dir / "manifest.json")
    files = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    pool = load_json(REPO / EVIDENCE["pool"])
    inputs[relative(export_dir / "manifest.json")] = sha256(export_dir / "manifest.json")
    for name in EXPORT_FILES:
        digest = sha256(export_dir / name)
        if files.get(name) != digest:
            raise GateError(f"export file {name} does not match its manifest")
        if name in pool["export_sha256"] and pool["export_sha256"][name] != digest:
            raise GateError(f"export file {name} does not match the pool register")
        inputs[relative(export_dir / name)] = digest
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    marker = MARKER.decode().strip()
    done = subprocess.run(
        [isaac_python, "-c", _EXTRACT_HELPER, str(export_dir), marker], capture_output=True, env=env, timeout=3600
    )
    if done.returncode != 0 or MARKER not in done.stdout:
        raise GateError(f"export streaming failed ({done.returncode}): {done.stderr.decode(errors='replace')[-800:]}")
    stream = done.stdout[done.stdout.index(MARKER) + len(MARKER) :]
    line, _, blob = stream.partition(b"\n")
    if len(blob) != json.loads(line)["npz_bytes"]:
        raise GateError("streamed export geometry is truncated")
    with np.load(io.BytesIO(blob), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


class Context:
    """Shared, read-only state of one process: robot model, streamed export, registers and recorded-run access."""

    def __init__(self, args, log):
        self.args, self.log = args, log
        self.inputs = {}
        for path in EVIDENCE.values():
            self.inputs[path] = sha256(REPO / path)
        self.inputs[relative(args.urdf)] = sha256(args.urdf)
        self.robot, mesh_hashes = load_robot(args.urdf)
        self.inputs.update({relative(p): h for p, h in mesh_hashes.items()})
        t0 = time.time()
        self.export = stream_export(args.export_dir, args.isaac_python, self.inputs)
        self.names = [str(n) for n in self.export["names"]]
        log(f"streamed {len(self.names)} export meshes in {time.time() - t0:.0f} s")
        self.registers = {key: load_json(REPO / path) for key, path in EVIDENCE.items() if key != "scope"}
        ref = load_json(Path(args.runs_root) / RECORDED["14944"] / "report.json")
        self.inputs[f"runs/{RECORDED['14944']}/report.json"] = sha256(
            Path(args.runs_root) / RECORDED["14944"] / "report.json"
        )
        self.q_home_ref = np.asarray(ref["initial_control_state"]["joint_pos"][0], float)

    def spur_index(self, tree):
        return [i for i, n in enumerate(self.names) if mesh_label(n) == f"tree{tree}_SPUR"][0]

    def run(self, key):
        run_dir = Path(self.args.runs_root) / RECORDED[key]
        for name in ("report.json", "frames.json"):
            self.inputs[f"runs/{RECORDED[key]}/{name}"] = sha256(run_dir / name)
        return load_run(run_dir)

    def scene_from_report(self, report):
        b = report["blender_scene"]
        tree = b.get("target_tree_index", 1 if b.get("tree_path", "").endswith("Tree1") else 0)
        t, R = placement_from_report(report)
        orch = Orchard(self.export, t, R, tree, b["target"]["source_component"]["source_vertex_indices"])
        q_home = np.asarray(report["initial_control_state"]["joint_pos"][0], float)
        return Scene(self.robot, orch, report["target_position_m"], b["target"]["axis_w"], q_home), tree

    def scene_from_component(self, tree, first_vertex, source_center, axis):
        """A layout no run recorded (screen.py's synthetic report): placement from the component, reference home."""
        k = self.spur_index(tree)
        C = self.export[f"c{k}"]
        verts = np.flatnonzero(C[int(first_vertex)] == C)
        t, R = placement_from_component(source_center)
        axis_w = R @ np.asarray(axis, float)
        axis_w /= np.linalg.norm(axis_w)
        orch = Orchard(self.export, t, R, tree, verts)
        local_centre = np.asarray(self.export[f"v{k}"])[verts].mean(0)
        check = float(np.linalg.norm(local_centre - np.asarray(source_center, float)))
        return Scene(self.robot, orch, TARGET_W, axis_w, self.q_home_ref), check

    def register_entry(self, register, first_vertex):
        hits = [e for e in self.registers[register]["targets"] if e["component_first_vertex"] == int(first_vertex)]
        if len(hits) != 1:
            raise GateError(f"{first_vertex} is not listed once in {EVIDENCE[register]}")
        return hits[0]


# ------------------------------------------------------------------------------------------- recorded stops
def contact_names(report):
    return {k: v["body_names"][0] for k, v in report["contact_coverage"]["sensors"].items()}


def frame_forces(frame):
    return {
        k: float(np.linalg.norm(np.asarray(v, float).reshape(-1, 3), axis=1).max())
        for k, v in frame["contact_forces_w_n"].items()
    }


def recorded_contact(report, frames, threshold=5.0):
    names = contact_names(report)
    for fr in frames:
        fired = {names[k]: f for k, f in frame_forces(fr).items() if f > threshold}
        if fired:
            link = max(fired, key=fired.get)
            return {"frame": fr["index"], "link": link, "force_n": fired[link]}
    return None


def recorded_home_contact(report, frames):
    names = contact_names(report)
    fired = {names[k]: f for k, f in frame_forces(frames[0]).items() if f > 0}
    if not fired:
        return None
    link = max(fired, key=fired.get)
    return {"frame": 0, "link": link, "force_n": fired[link]}


def recorded_tof(frames):
    """First frame where a valid, finite, positive ToF zone reads under 0.06 m (as render_pruning_workflow's guard)."""
    for fr in frames:
        best = None
        for key, sensor in (("tof_left", "tof0"), ("tof_right", "tof1")):
            ranges = np.array([[np.nan if x is None else x for x in row] for row in fr[f"{key}_m"]], float)
            valid = np.asarray(fr[f"{key}_valid"], bool) & np.isfinite(ranges) & (ranges > 0)
            if valid.any():
                v, u = np.unravel_index(np.argmin(np.where(valid, ranges, np.inf)), ranges.shape)
                if ranges[v, u] < MIN_CLEAR_TOF_M and (best is None or ranges[v, u] < best["range_m"]):
                    best = {
                        "frame": fr["index"],
                        "sensor": sensor,
                        "range_m": float(ranges[v, u]),
                        "zone_vu": [int(v), int(u)],
                    }
        if best:
            return best
    return None


def recorded_tracking_loss(frames):
    started = False
    for fr in frames:
        state = ((fr.get("live_vision") or {}).get("measurement") or {}).get("state")
        if state == "tracking":
            started = True
        elif started and state is not None:
            return {"frame": fr["index"], "state": state}
    return None


def recorded_tof_hit(scene, frame, sensor, zone_vu):
    """Re-cast the recorded zone from the recorded joints; the object it hits and the per-sensor re-cast error."""
    Tb = scene.robot.kin.fk(frame["joint_position_rad"])[EEF_LINK]
    key = "tof_left_m" if sensor == "tof0" else "tof_right_m"
    rec = np.array([[np.nan if x is None else x for x in row] for row in frame[key]], float)
    o = Tb[:3, :3] @ TOF_IN_BASE[sensor] + Tb[:3, 3]
    t, g, p = scene.orch.cast(np.repeat(o[None], 64, 0), TOF_DIRS @ Tb[:3, :3].T)
    t = t.reshape(8, 8)
    m = np.isfinite(rec) & np.isfinite(t)
    v, u = zone_vu
    j = v * 8 + u
    label = scene.orch.describe(g[j], p[j]) if np.isfinite(t[v, u]) else ("miss", -1)
    return {
        "object": f"{label[0]}:{label[1]}",
        "recast_m": float(t[v, u]),
        "recorded_m": float(rec[v, u]),
        "median_abs_recast_error_m": float(np.median(np.abs(t[m] - rec[m]))) if m.any() else None,
    }


def recorded_los_first(scene, frame):
    cam = np.asarray(frame["wrist_position_w_m"], float)
    dist = float(np.linalg.norm(scene.target - cam))
    t, g, pr = scene.orch.cast(cam[None], ((scene.target - cam) / dist)[None])
    return {
        "object": scene.orch.describe(g[0], pr[0])[0] if np.isfinite(t[0]) else None,
        "range_m": float(t[0]),
        "camera_to_target_m": dist,
    }


def model_events(first):
    """(step, kind, link or sensor, object) of each first event of a replay."""
    out = []
    if "contact" in first:
        k, _ph, link, obj, _c = first["contact"]
        out.append((int(k), "contact", link, obj))
    if "target_contact" in first:
        out.append(
            (int(first["target_contact"][0]), "target_contact", first["target_contact"][2], Orchard.TARGET_LABEL)
        )
    if "tof" in first:
        k, _ph, _m, tof = first["tof"]
        sensor = min(("tof0", "tof1"), key=lambda s: tof[s][0])
        out.append((int(k), "tof", sensor, tof[sensor][1]))
    if "los" in first:
        out.append((int(first["los"][0]), "los", "camera", first["los"][1]))
    if "fov" in first:
        out.append((int(first["fov"]), "fov", "camera", None))
    return sorted(out, key=lambda e: e[0])


# ------------------------------------------------------------------------------------------- acceptance parts
def part_replays(ctx, targets, settings):
    """Baseline replays of the diagnosed and passing runs, with the recorded stop facts and model validation."""
    units = {}
    for key in targets:
        t0 = time.time()
        report, frames = ctx.run(key)
        scene, _tree = ctx.scene_from_report(report)
        approach, retreat, R0 = straight_path(scene)
        summary, rows = run_path(scene, approach, retreat, R0, stride=1, ik_updates=settings["ik_frame_updates"])
        p0, _ = scene.robot.kin.tool_pose(scene.q_home)
        mount, cam_rot = camera_for_home(p0, R0, scene.target, scene.axis)
        cam_rec = report.get("camera") or {}
        unit = {
            "run": RECORDED[key],
            "recorded_stopped_reason": report.get("stopped_reason"),
            "replay": summary,
            "replay_seed_los_blocked": rows[0]["los_blocked"],
            "model_events": model_events(summary["first"]),
            "camera_mount_error_m": float(np.linalg.norm(mount - np.asarray(cam_rec["wrist_position_in_tool_m"])))
            if cam_rec.get("wrist_position_in_tool_m")
            else None,
            "camera_rotation_error": float(np.abs(cam_rot - np.asarray(cam_rec["wrist_rotation_in_tool_ros"])).max())
            if cam_rec.get("wrist_rotation_in_tool_ros")
            else None,
        }
        errs, angs = [], []
        for fr in frames[::7]:
            p, R = scene.robot.kin.tool_pose(fr["joint_position_rad"])
            errs.append(float(np.linalg.norm(p - np.asarray(fr["tool_pose_wxyz"][:3]))))
            angs.append(math.degrees(rotation_angle_rad(R, quat_wxyz_to_R(fr["tool_pose_wxyz"][3:]))))
        unit["fk_check"] = {"frames": len(errs), "max_position_error_m": max(errs), "max_rotation_error_deg": max(angs)}
        recorded = {}
        contact = recorded_contact(report, frames)
        if contact:
            T = scene.robot.kin.fk(frames[contact["frame"]]["joint_position_rad"])[contact["link"]]
            c = link_clearance(scene.robot, scene.orch, contact["link"], T)
            contact.update(object=f"{c['nearest']}:{c['component_first_vertex']}", hull_clearance_m=c["min_m"])
            recorded["contact"] = contact
        home = recorded_home_contact(report, frames)
        if home:
            T = scene.robot.kin.fk(frames[0]["joint_position_rad"])[home["link"]]
            c = link_clearance(scene.robot, scene.orch, home["link"], T)
            home.update(object=f"{c['nearest']}:{c['component_first_vertex']}", hull_clearance_m=c["min_m"])
            recorded["home_contact"] = home
        tof = recorded_tof(frames)
        if tof:
            tof.update(recorded_tof_hit(scene, frames[tof["frame"]], tof["sensor"], tof["zone_vu"]))
            recorded["tof"] = tof
        loss = recorded_tracking_loss(frames)
        if loss:
            loss.update(recorded_los_first(scene, frames[loss["frame"]]))
            recorded["tracking_loss"] = loss
        unit["recorded"] = recorded
        unit["seconds"] = round(time.time() - t0, 1)
        units[key] = unit
        frames_at = {k: v.get("frame") for k, v in recorded.items()}
        ctx.log(f"replay {key}: first={summary['first']} recorded={frames_at}")
    return units


def home_check(scene):
    home = clearance2(scene.robot, scene.orch, scene.q_home)
    hl = min(home, key=lambda k: home[k]["min_m"])
    return {
        "home_min_clear_m": home[hl]["min_m"],
        "home_min_link": hl,
        "home_min_obj": f"{home[hl]['nearest']}:{home[hl]['component_first_vertex']}",
        "home_target_clear_m": min(v["target_m"] for v in home.values()),
    }


def part_refusals(ctx, targets, settings):
    units = {}
    for key in targets:
        register, tree = REFUSALS[key]
        entry = ctx.register_entry(register, int(key))
        scene, check = ctx.scene_from_component(tree, int(key), entry["source_center_m"], entry["axis"])
        units[key] = {"register": EVIDENCE[register], "tree": tree, "center_check_m": check, **home_check(scene)}
        ctx.log(f"refusal {key}: home {units[key]['home_min_clear_m']:.4f} {units[key]['home_min_link']}")
    return units


def classify(unit):
    """The classification that wrote tree1_swept_path_predictions_2026-09-27.json, in its precedence."""
    home = (unit["home_min_clear_m"], unit["home_min_link"], unit["home_min_obj"])
    if home[0] < 0:
        return {
            "class": "home_overlap",
            "detail": f"{home[1]} into {home[2]} by {-home[0] * 1000:.1f} mm at the settled home",
        }
    if unit.get("seed_los_blocked_home"):
        return {"class": "not_visible", "detail": "the seed pixel's line of sight is blocked at home"}
    if home[0] < MARGINAL_HOME_M:
        return {"class": "marginal_home", "detail": f"{home[1]} within {home[0] * 1000:.1f} mm of {home[2]} at home"}
    first = (unit.get("path") or {}).get("first") or {}
    events = []
    if "contact" in first:
        events.append((first["contact"][0], "hazard_contact", f"{first['contact'][2]} on {first['contact'][3]}"))
    if "tof" in first:
        events.append((first["tof"][0], "tof_minimum_clearance", f"ToF {first['tof'][2]:.4f} m"))
    if events:
        k, cls, what = min(events)
        return {"class": cls, "step": k, "recorded_frame_window": [k + 1, k + 5], "detail": what}
    return {
        "class": "geometry_clear",
        "detail": "no contact, ToF or line-of-sight event on the straight path; "
        "may still fail on perception or the drop check",
    }


def part_predictions(ctx, targets, settings):
    units = {}
    for key in targets:
        register = next(
            r
            for r in ("seeded", "seeded30")
            if any(e["component_first_vertex"] == int(key) for e in ctx.registers[r]["targets"])
        )
        entry = ctx.register_entry(register, int(key))
        scene, check = ctx.scene_from_component(
            entry["target_tree_index"], int(key), entry["source_center_m"], entry["axis"]
        )
        unit = {"register": EVIDENCE[register], "center_check_m": check, **home_check(scene)}
        if unit["home_min_clear_m"] >= 0:
            approach, retreat, R0 = straight_path(scene)
            summary, rows = run_path(scene, approach, retreat, R0, stride=1, ik_updates=settings["ik_frame_updates"])
            unit.update(path=summary, seed_los_blocked_home=rows[0]["los_blocked"])
        unit["call"] = classify(unit)
        units[key] = unit
        ctx.log(f"prediction {key}: {unit['call']['class']} ({unit['call']['detail']})")
    return units


def part_reorientation(ctx, targets, settings):
    units = {}
    planned = {str(t["component_first_vertex"]): t for t in ctx.registers["planned"]["targets"]}
    for key in targets:
        t0 = time.time()
        report, _frames = ctx.run(key)
        scene, _tree = ctx.scene_from_report(report)
        search = final_pose_search(scene, census=settings["census"], log=ctx.log)
        paths = plan_paths(scene, search["kept"], ik_updates=settings["ik_frame_updates"], log=ctx.log)
        unit = {"run": RECORDED[key], "search": search, "paths": paths}
        if key in COMMITTED_PLANS and planned.get(key, {}).get("planned_final_tool_quat_wxyz"):
            Rc = quat_wxyz_to_R(planned[key]["planned_final_tool_quat_wxyz"])
            ident = grid_identity(Rc)
            p0, R0 = scene.robot.kin.tool_pose(scene.q_home)
            match = [
                p
                for p in paths
                if p["direction_index"] == ident["direction_index"] and p["psi_deg"] == ident["psi_deg"]
            ]
            unit["committed_plan"] = {
                "grid_identity": ident,
                "total_rotation_deg": math.degrees(rotation_angle_rad(R0, Rc)),
                "matching_paths": [
                    {
                        k: p[k]
                        for k in (
                            "pose_index",
                            "short_m",
                            "standoff_m",
                            "total_rotation_deg",
                            "clear",
                            "clear_with_view",
                        )
                    }
                    for p in match
                ],
            }
            clear = [p for p in match if p["clear"]]
            if clear:
                pick = min(clear, key=lambda p: (p["standoff_m"] != 0.06, p["standoff_m"]))
                approach, _retreat, *_ = planned_legs(scene, Rc, pick["short_m"], pick["standoff_m"])
                poses = solve_poses(scene, approach, settings["ik_frame_updates"])
                radius = float(report["blender_scene"]["target"]["radius_m"])
                try:
                    series = predicted_patch_series(scene, poses, Rc, radius)
                except ValueError as exc:  # a jaw corner behind the camera: the silhouette is undefined
                    series = {"approach_kept": [], "closure_kept": None, "error": str(exc)}
                unit["committed_plan"]["patch_projection"] = {
                    "standoff_m": pick["standoff_m"],
                    "short_m": pick["short_m"],
                    **series,
                    "conditions": selection_conditions(
                        pick["clear_with_view"], series["approach_kept"], series["closure_kept"]
                    ),
                }
        unit["seconds"] = round(time.time() - t0, 1)
        units[key] = unit
        ctx.log(f"reorientation {key}: kept {len(search['kept'])} poses, clear paths {sum(p['clear'] for p in paths)}")
    return units


def part_jaw_projection(ctx, targets, settings):
    """Validate the patch projection on recorded jaw self-mask runs: recorded poses, cameras, pixels and closure."""
    units = {}
    for key in targets:
        run_dir = Path(ctx.args.runs_root) / JAW_PROJECTION_RUNS[key]
        for name in ("report.json", "frames.json"):
            ctx.inputs[f"runs/{JAW_PROJECTION_RUNS[key]}/{name}"] = sha256(run_dir / name)
        report, frames = load_run(run_dir)
        b = report["blender_scene"]
        roll, radius = float(b["visual_proxy_roll_rad"]), float(b["target"]["radius_m"])
        rows, prev = [], None
        for fr in frames:
            m = (fr.get("live_vision") or {}).get("measurement") or {}
            pose = np.asarray(fr["tool_pose_wxyz"], float)
            W = T_from(np.asarray(fr["wrist_position_w_m"], float), np.asarray(fr["wrist_rotation_w_ros"], float))
            jsm = _jsm()
            boxes = jsm.jaw_boxes(pose, roll, float(fr.get("visual_jaw_closure_progress") or 0.0), radius)
            try:
                mask = jsm.jaw_mask(boxes, K_CAM, W, (IMAGE_H, IMAGE_W), margin_px=JAW_MASK_MARGIN_PX)
            except ValueError:
                mask = None
            if prev is not None and mask is not None and prev["mask"] is not None and "jaw_mask_patch_unmasked" in m:
                if prev["pixel"] is not None:
                    predicted = kept_elements(
                        patch_masked(prev["mask"], prev["pixel"]), patch_masked(mask, m["pixel_xy"])
                    )
                    rows.append(
                        {
                            "frame": fr["index"],
                            "phase": fr["phase"],
                            "progress": fr.get("visual_jaw_closure_progress"),
                            "recorded_kept": int(m["jaw_mask_patch_unmasked"]),
                            "predicted_kept": predicted,
                            "state": m.get("state"),
                        }
                    )
            prev = {"mask": mask, "pixel": m.get("pixel_xy")}
        diffs = [abs(r["predicted_kept"] - r["recorded_kept"]) for r in rows]
        approach = [
            r["recorded_kept"] for r in rows if r["phase"] in ("vision_approach", "align") and not r["progress"]
        ]
        closing = [r for r in rows if r["progress"] and r["progress"] > 0]
        units[key] = {
            "run": JAW_PROJECTION_RUNS[key],
            "frames_compared": len(rows),
            "exact_matches": sum(d == 0 for d in diffs),
            "max_abs_difference": max(diffs) if diffs else None,
            "recorded_approach_min_kept": min(approach) if approach else None,
            "recorded_first_closure_kept": closing[0]["recorded_kept"] if closing else None,
            "rows": rows,
        }
        ctx.log(
            f"jaw projection {key}: {units[key]['exact_matches']}/{len(rows)} exact,"
            f" max diff {units[key]['max_abs_difference']}"
        )
    return units


PART_TARGETS = {
    "replays": tuple(RECORDED),
    "refusals": tuple(REFUSALS),
    "predictions": None,  # filled from the two seeded registers
    "reorientation": DIAGNOSED,
    "jaw_projection": tuple(JAW_PROJECTION_RUNS),
}
PART_FUNCTIONS = {
    "replays": part_replays,
    "refusals": part_refusals,
    "predictions": part_predictions,
    "reorientation": part_reorientation,
    "jaw_projection": part_jaw_projection,
}


def committed_calls():
    doc = load_json(REPO / EVIDENCE["predictions"])
    calls = {}
    for register in doc["registers"].values():
        for t in register["targets"]:
            calls[str(t["component_first_vertex"])] = t
    return calls


# ------------------------------------------------------------------------------------------- acceptance items
def item_recorded_stops(replays):
    checks = []
    for stop in STOPS:
        unit = replays.get(stop["target"])
        if unit is None:
            checks.append({**stop, "pass": False, "why": "replay missing"})
            continue
        events = [tuple(e) for e in unit["model_events"]]
        rec_key = {"contact": "contact", "home_contact": "home_contact", "tof": "tof", "los": "tracking_loss"}[
            stop["kind"]
        ]
        recorded = unit["recorded"].get(rec_key)
        kind = {"contact": "contact", "home_contact": "contact", "tof": "tof", "los": "los"}[stop["kind"]]
        model = next((e for e in events if e[1] == kind), None)
        row = {**stop, "recorded": recorded, "model_event": model}
        if recorded is None or model is None:
            checks.append({**row, "pass": False, "why": "no recorded stop" if recorded is None else "no model event"})
            continue
        rec_link = recorded.get("link") or recorded.get("sensor") or "camera"
        lead = recorded["frame"] - model[0]
        explained = {s["kind"] for s in STOPS if s["target"] == stop["target"] and s["name"] != stop["name"]}
        earlier = [
            e
            for e in events
            if e[0] < model[0] and not (e[1] == "contact" and "home_contact" in explained and e[0] == 0)
        ]
        row.update(
            recorded_frame=recorded["frame"],
            model_step=model[0],
            frames_early=lead,
            link_match=model[2] == rec_link,
            object_match_recorded=model[3] == recorded.get("object"),
            object_match_committed=model[3] == stop["committed_object"],
            earlier_model_events=earlier,
        )
        row["pass"] = bool(
            row["link_match"]
            and row["object_match_recorded"]
            and row["object_match_committed"]
            and STOP_WINDOW_FRAMES[0] <= lead <= STOP_WINDOW_FRAMES[1]
            and not earlier
        )
        checks.append(row)
    return {
        "required": "every listed stop reproduced on the same link (sensor) and object, 0-4 frames early, as the "
        "replay's earliest event (18669's home contact explains the k=0 contact before its ToF stop); recorded frame = "
        "first frame over the stop's threshold in frames.json, recorded object = the model's nearest object to that "
        "link at the recorded joints (ToF: the re-cast hit of the recorded zone; line of sight: first hit from the "
        "recorded camera), and it must equal the object the committed diagnosis names",
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


def item_recorded_passes(replays):
    checks = []
    for key in PASSES:
        unit = replays.get(key)
        first = (unit or {}).get("replay", {}).get("first")
        checks.append(
            {
                "target": key,
                "first_events": first,
                "worst_clear_approach": (unit or {}).get("replay", {}).get("worst_clear_approach"),
                "worst_tof_approach_m": ((unit or {}).get("replay", {}).get("worst_tof_approach") or [None])[0],
                "pass": unit is not None and first == {},
            }
        )
    return {
        "required": "the baseline replays of 14944, 15004, 14884 and Stage A 8235 have no contact, target contact, "
        "ToF, line-of-sight or field-of-view event",
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


def item_refused_layouts(refusals):
    checks = []
    for key in REFUSALS:
        unit = refusals.get(key)
        checks.append(
            {
                "target": key,
                **({k: unit[k] for k in ("home_min_clear_m", "home_min_link", "home_min_obj")} if unit else {}),
                "pass": unit is not None and unit["home_min_clear_m"] < 0,
            }
        )
    return {
        "required": "each refused layout (10001, 10061, 23167, 19145) has a link hull overlapping orchard geometry "
        "(signed clearance < 0) at the settled home",
        "checks": checks,
        "pass": all(c["pass"] for c in checks),
    }


def item_reorientation(reorient):
    per_target, clear_set = {}, set()
    for key in DIAGNOSED:
        unit = reorient.get(key)
        if unit is None:
            per_target[key] = {"missing": True}
            continue
        clear = [p for p in unit["paths"] if p["clear"]]
        if clear:
            clear_set.add(key)
        per_target[key] = {
            "kept_final_poses": len(unit["search"]["kept"]),
            "smallest_reorientation_deg": unit["search"]["kept"][0]["total_rotation_deg"]
            if unit["search"]["kept"]
            else None,
            "census": unit["search"]["census"],
            "paths_checked": len(unit["paths"]),
            "clear_paths": len(clear),
            "clear_paths_with_view": sum(p["clear_with_view"] for p in clear),
            "clear_rotations_deg": sorted({round(p["total_rotation_deg"], 1) for p in clear}),
            "committed_plan": {k: v for k, v in (unit.get("committed_plan") or {}).items() if k != "patch_projection"},
        }
    committed_ok = {}
    for key, angle in COMMITTED_PLANS.items():
        plan = (reorient.get(key) or {}).get("committed_plan") or {}
        rot = plan.get("total_rotation_deg")
        committed_ok[key] = bool(
            plan.get("grid_identity", {}).get("residual", 1.0) < 1e-3
            and any(p["clear"] for p in plan.get("matching_paths", []))
            and rot is not None
            and round(rot, 1) == angle
        )
    missing = [k for k in DIAGNOSED if k not in reorient]
    return {
        "required": "among the seven diagnosed targets, a clear re-oriented path (no contact, target contact, ToF or "
        "IK event at stride 3 and again at stride 1) exists for exactly 530 and 19444, and each committed planned "
        "orientation (530 at 55.5 deg, 19444 at 83.6 deg; on the search grid) is among the clear paths",
        "targets_with_clear_path": sorted(clear_set, key=int),
        "committed_orientation_clear": committed_ok,
        "per_target": per_target,
        "pass": not missing and clear_set == set(COMMITTED_PLANS) and all(committed_ok.values()),
    }


def item_predictions(predictions):
    calls = committed_calls()
    checks, required_ok = [], True
    for key, committed in calls.items():
        unit = predictions.get(key)
        call = (unit or {}).get("call") or {}
        exempt = int(key) in EXEMPT_PREDICTIONS
        row = {
            "target": int(key),
            "committed_class": committed["class"],
            "model_class": call.get("class"),
            "class_match": call.get("class") == committed["class"],
            "exempt": exempt,
            "committed_detail": committed.get("detail"),
            "model_detail": call.get("detail"),
            "committed_step": committed.get("step"),
            "model_step": call.get("step"),
        }
        row["pass"] = bool(row["class_match"] or exempt)
        required_ok &= row["pass"]
        checks.append(row)
    return {
        "required": "the class of all 40 committed per-target calls reproduced, except 35837, 35957 and 36017 "
        "(marginal at home within 0.1 mm of a trellis wire; either class accepted); steps and details are reported, "
        "not required",
        "matches": sum(c["class_match"] for c in checks),
        "required_matches": sum(c["class_match"] for c in checks if not c["exempt"]),
        "required_calls": sum(not c["exempt"] for c in checks),
        "checks": sorted(checks, key=lambda c: c["target"]),
        "pass": len(checks) == 40 and required_ok and all(str(k) in predictions for k in calls),
    }


def model_validation(replays, jaw):
    stops = [u["recorded"].get("contact") for u in replays.values() if u["recorded"].get("contact")]
    tof = [u["recorded"]["tof"] for u in replays.values() if u["recorded"].get("tof")]
    return {
        "kinematics_max_tool_error_m": max(
            (u["fk_check"]["max_position_error_m"] for u in replays.values()), default=None
        ),
        "kinematics_max_tool_error_deg": max(
            (u["fk_check"]["max_rotation_error_deg"] for u in replays.values()), default=None
        ),
        "hull_clearance_at_recorded_contact_frames_m": sorted(s["hull_clearance_m"] for s in stops),
        "tof_recast_median_abs_error_m": sorted(
            t["median_abs_recast_error_m"] for t in tof if t["median_abs_recast_error_m"] is not None
        ),
        "camera_mount_error_max_m": max(
            (u["camera_mount_error_m"] for u in replays.values() if u["camera_mount_error_m"] is not None), default=None
        ),
        "jaw_projection_vs_recorded_telemetry": {
            k: {
                kk: v[kk]
                for kk in (
                    "frames_compared",
                    "exact_matches",
                    "max_abs_difference",
                    "recorded_approach_min_kept",
                    "recorded_first_closure_kept",
                )
            }
            for k, v in (jaw or {}).items()
        },
    }


def selection_calibration(reorient):
    out = {}
    for key in COMMITTED_PLANS:
        proj = ((reorient.get(key) or {}).get("committed_plan") or {}).get("patch_projection")
        out[key] = (
            None
            if proj is None
            else {
                "approach_min_kept": min(proj["approach_kept"]),
                "closure_kept": proj["closure_kept"],
                "conditions": proj["conditions"],
            }
        )
    agrees = bool(
        out.get("530")
        and out.get("19444")
        and out["530"]["conditions"]["b_approach_keeps_patch"]
        and out["530"]["conditions"]["c_closure_loses_patch"]
        and not out["19444"]["conditions"]["b_approach_keeps_patch"]
    )
    return {
        "note": "not an acceptance item: the scope says condition (b) is the check 19444 fails and 530 passes, and "
        "530's closure started the hold; the model's projection of both committed plans is reported here",
        "per_target": out,
        "agrees_with_recorded_outcomes": agrees,
    }


def assemble_acceptance(parts, sensitivity=None):
    units = {name: {} for name in PARTS}
    provenance_rows = []
    for doc in parts:
        if doc.get("kind") != "acceptance_part":
            raise GateError("not an acceptance part")
        units[doc["part"]].update(doc["units"])
        provenance_rows.append(doc["provenance"])
    shas = {json.dumps(p["code_sha256"], sort_keys=True) for p in provenance_rows}
    settings = {json.dumps(p["settings"], sort_keys=True) for p in provenance_rows}
    if len(shas) != 1 or len(settings) != 1:
        raise GateError("parts come from different code or settings")
    inputs = {}
    for p in provenance_rows:
        for k, v in p["inputs_sha256"].items():
            if inputs.setdefault(k, v) != v:
                raise GateError(f"parts disagree on input {k}")
    expected_predictions = set(committed_calls())
    coverage = {
        "replays": set(RECORDED) <= set(units["replays"]),
        "refusals": set(REFUSALS) <= set(units["refusals"]),
        "predictions": expected_predictions <= set(units["predictions"]),
        "reorientation": set(DIAGNOSED) <= set(units["reorientation"]),
    }
    items = {
        "recorded_stops": item_recorded_stops(units["replays"]),
        "recorded_passes": item_recorded_passes(units["replays"]),
        "refused_layouts": item_refused_layouts(units["refusals"]),
        "reorientation_result": item_reorientation(units["reorientation"]),
        "committed_predictions": item_predictions(units["predictions"]),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "kind": "acceptance",
        "scope": SCOPE,
        "scope_doc": SCOPE_DOC,
        "code_revision": provenance_rows[0]["code_revision"],
        "code_tree_dirty": any(p["code_tree_dirty"] for p in provenance_rows),
        "tool_committed_unmodified": all(p["tool_committed_unmodified"] for p in provenance_rows),
        "code_sha256": provenance_rows[0]["code_sha256"],
        "settings": provenance_rows[0]["settings"],
        "inputs_sha256": dict(sorted(inputs.items())),
        "part_jobs": sorted({str(p.get("slurm_job_id")) for p in provenance_rows}),
        "coverage": coverage,
        "items": items,
        "all_items_pass": all(i["pass"] for i in items.values()) and all(coverage.values()),
        "model_validation": model_validation(units["replays"], units["jaw_projection"]),
        "selection_rule_calibration": selection_calibration(units["reorientation"]),
        "units": units,
    }
    if sensitivity:
        sens = {name: {} for name in PARTS}
        for doc in sensitivity:
            sens[doc["part"]].update(doc["units"])
        result["sensitivity"] = {
            "settings": sensitivity[0]["provenance"]["settings"],
            "recorded_stops_pass": item_recorded_stops(sens["replays"])["pass"] if sens["replays"] else None,
            "recorded_passes_pass": item_recorded_passes(sens["replays"])["pass"] if sens["replays"] else None,
            "recorded_stops": [
                {k: c.get(k) for k in ("name", "model_step", "frames_early", "pass")}
                for c in item_recorded_stops(sens["replays"])["checks"]
            ]
            if sens["replays"]
            else None,
        }
    return result


# ------------------------------------------------------------------------------------------- candidate search
def measure_pool(ctx):
    """The jaw-fit screen of eval_targets_2026-09-23.json, re-measured from the streamed spur meshes."""
    if str(PACKAGE) not in sys.path:
        sys.path.insert(0, str(PACKAGE))
    from isaaclab_pruning.sim.blender_component import component_geometry

    register = ctx.registers["pool"]
    max_radius = float(register["max_radius_m"])
    pool, report = [], {}
    for tree in (0, 1):
        k = ctx.spur_index(tree)
        V, C = np.asarray(ctx.export[f"v{k}"]), np.asarray(ctx.export[f"c{k}"])
        groups = {}
        for vertex, label in enumerate(C):
            groups.setdefault(int(label), []).append(vertex)
        measured = []
        for members in groups.values():
            if len(members) < 6:
                continue
            try:
                measured.append(component_geometry(V, sorted(members), object_name=f"tree{tree}_SPUR"))
            except ValueError:
                continue
        accepted = [m for m in measured if 0 < m["max_radius_m"] <= max_radius]
        rejected = sorted(m["component_first_vertex"] for m in measured if not 0 < m["max_radius_m"] <= max_radius)
        expected = register["populations"][f"tree{tree}"]
        listed_out = register["screened_out_component_first_vertexes"][f"tree{tree}"]
        listed_out = sorted(e["component_first_vertex"] if isinstance(e, dict) else int(e) for e in listed_out)
        report[f"tree{tree}"] = {
            "measured": len(measured),
            "screened_in": len(accepted),
            "matches_register": len(measured) == expected["measured_components"]
            and len(accepted) == expected["screened_in"]
            and rejected == listed_out,
        }
        for m in accepted:
            pool.append(
                {
                    "tree": tree,
                    "first_vertex": m["component_first_vertex"],
                    "center": m["center_m"],
                    "axis": m["axis"],
                    "radius_m": m["max_radius_m"],
                }
            )
    if not all(r["matches_register"] for r in report.values()):
        raise GateError(f"the re-measured pool does not match {EVIDENCE['pool']}: {report}")
    pool.sort(key=lambda e: (e["tree"], e["first_vertex"]))
    return [e for e in pool if (e["tree"], e["first_vertex"]) not in EXCLUDED_FROM_SEARCH], report


def search_target(ctx, entry, settings):
    """One pool target: home and seed checks, the final-pose search, planned paths and the patch projection."""
    scene, check = ctx.scene_from_component(entry["tree"], entry["first_vertex"], entry["center"], entry["axis"])
    unit = {"center_check_m": check, **home_check(scene)}
    if unit["home_min_clear_m"] < 0:
        unit["excluded"] = "home_overlap"
        return unit
    p0, R0 = scene.robot.kin.tool_pose(scene.q_home)
    mount, cam_rot = camera_for_home(p0, R0, scene.target, scene.axis)
    seed = scene.camera(scene.robot.kin.fk(scene.q_home)[EEF_LINK], mount, cam_rot)
    unit["seed_los_blocked_home"] = seed["los_blocked"]
    unit["seed_in_fov_home"] = seed["cam_in_fov"]
    if seed["los_blocked"] or not seed["cam_in_fov"]:
        unit["excluded"] = "not_visible_at_home"
        return unit
    search = final_pose_search(scene, census=False)
    unit["search"] = {k: v for k, v in search.items() if k != "checked"}
    if not search["kept"]:
        unit["excluded"] = "no_reachable_gate_clear_final_pose"
        return unit
    paths = plan_paths(scene, search["kept"], ik_updates=settings["ik_frame_updates"])
    unit["paths"] = paths
    candidates = []
    for p in paths:
        if not p["clear"]:
            continue
        Rf = np.asarray(search["kept"][p["pose_index"]]["R"], float)
        approach, _retreat, *_ = planned_legs(scene, Rf, p["short_m"], p["standoff_m"])
        poses = solve_poses(scene, approach, settings["ik_frame_updates"])
        try:
            series = predicted_patch_series(scene, poses, Rf, entry["radius_m"])
        except ValueError as exc:  # a jaw corner behind the camera: the silhouette is undefined, fail closed
            series = {"approach_kept": [], "closure_kept": None, "error": str(exc)}
        cond = selection_conditions(p["clear_with_view"], series["approach_kept"], series["closure_kept"])
        candidates.append(
            {
                "pose_index": p["pose_index"],
                "standoff_m": p["standoff_m"],
                "short_m": p["short_m"],
                "total_rotation_deg": p["total_rotation_deg"],
                "quat_wxyz": search["kept"][p["pose_index"]]["quat_wxyz"],
                "conditions": cond,
                "closure_pixel": series.get("closure_pixel"),
                "error": series.get("error"),
            }
        )
    unit["variants"] = candidates
    unit["qualifies"] = any(c["conditions"]["qualifies"] for c in candidates)
    return unit


def require_passing_acceptance(path):
    doc = load_json(path)
    if doc.get("kind") != "acceptance" or doc.get("schema_version") != SCHEMA_VERSION:
        raise GateError(f"{path} is not an acceptance JSON of this tool")
    failing = [name for name, item in doc.get("items", {}).items() if not item.get("pass")]
    if not doc.get("all_items_pass") or failing or len(doc.get("items", {})) != 5:
        raise GateError(
            f"the acceptance test did not pass every item ({failing or 'coverage'}); the search selects nothing"
        )
    if doc.get("code_sha256", {}).get(relative(Path(__file__).resolve())) != sha256(Path(__file__).resolve()):
        raise GateError("the acceptance JSON was made by different code than this file")
    return doc


def assemble_search(acceptance, parts):
    units, provenance_rows, pool_report = {}, [], None
    for doc in parts:
        if doc.get("kind") != "search_part":
            raise GateError("not a search part")
        units.update(doc["units"])
        provenance_rows.append(doc["provenance"])
        pool_report = doc["pool"]
    expected = pool_report["targets_after_exclusion"]
    if len(units) != expected:
        raise GateError(f"search shards cover {len(units)} of {expected} targets")
    candidates = []
    for key, unit in sorted(units.items(), key=lambda kv: (int(kv[0].split("_")[0][4:]), int(kv[0].split("_")[1]))):
        for v in unit.get("variants", []):
            if v["conditions"]["qualifies"]:
                candidates.append({"target": key, **v})
                break
    excluded = {}
    for unit in units.values():
        reason = unit.get("excluded") or ("qualifies" if unit.get("qualifies") else "no_variant_meets_a_b_c")
        excluded[reason] = excluded.get(reason, 0) + 1
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "search",
        "scope": SCOPE,
        "selection_rule": {
            "a": "a clear re-oriented path to a gate-clear final pose: no contact, target contact, ToF or IK event at "
            "stride 3 and at stride 1, the settled home clear of the orchard, and no camera line-of-sight block or "
            "field-of-view loss on the approach",
            "b": f"the open jaw leaves at least {MIN_UNMASKED_PATCH_ELEMENTS} of the {PATCH_ELEMENTS} patch elements "
            "at every predicted approach pose (consecutive patches, the tracker's getRectSubPix footprint, 2 px mask "
            "margin)",
            "c": "at the planned final pose with closure progress 1/6 the jaw leaves fewer than "
            f"{MIN_UNMASKED_PATCH_ELEMENTS}",
        },
        "acceptance_code_revision": acceptance["code_revision"],
        "code_sha256": provenance_rows[0]["code_sha256"],
        "inputs_sha256": provenance_rows[0]["inputs_sha256"],
        "part_jobs": sorted({str(p.get("slurm_job_id")) for p in provenance_rows}),
        "pool": pool_report,
        "outcome_counts": excluded,
        "candidates": candidates,
        "go": bool(candidates),
        "units": units,
    }


# ------------------------------------------------------------------------------------------- CLI
def _log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", file=sys.stderr, flush=True)


def _targets(part, value):
    if value:
        return [v.strip() for v in value.split(",") if v.strip()]
    if part == "predictions":
        return [str(k) for k in committed_calls()]
    return list(PART_TARGETS[part])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        p.add_argument("--output", required=True, type=Path, help="New JSON file outside the repository")
        p.add_argument("--runs-root", type=Path, default=RUNS_DEFAULT, help="Directory of the recorded batches")
        p.add_argument("--export-dir", type=Path, default=EXPORT_DEFAULT, help="The hash-verified orchard export")
        p.add_argument("--urdf", type=Path, default=URDF_DEFAULT, help="The content-addressed robot URDF")
        p.add_argument("--isaac-python", default=ISAAC_PYTHON_DEFAULT, help="Python with pxr that streams the export")
        p.add_argument("--ik-frame-updates", type=int, default=IK_FRAME_UPDATES, help="IK updates per replay frame")

    p = sub.add_parser("acceptance-part", help="compute one acceptance part")
    common(p)
    p.add_argument("--part", required=True, choices=PARTS)
    p.add_argument("--targets", help="comma-separated subset of the part's targets")
    p.add_argument("--census", action="store_true", help="evaluate all final poses, not only the shortlist")
    p = sub.add_parser("acceptance", help="assemble acceptance parts into the acceptance JSON")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--parts", required=True, nargs="+", type=Path)
    p.add_argument("--sensitivity", nargs="*", type=Path, default=[], help="replay parts run with other IK settings")
    p = sub.add_parser("search-part", help="search one shard of the jaw-fit-screened pool")
    common(p)
    p.add_argument("--acceptance", required=True, type=Path, help="acceptance JSON in which every item passes")
    p.add_argument("--shard", default="1/1", help="K/N: this process searches every N-th target from the K-th")
    p = sub.add_parser("search", help="assemble search shards and apply the selection rule")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--acceptance", required=True, type=Path)
    p.add_argument("--parts", required=True, nargs="+", type=Path)
    args = parser.parse_args(argv)

    try:
        output = refuse_output(args.output)
    except ValueError as exc:
        parser.error(str(exc))
    try:
        if args.command == "acceptance":
            result = assemble_acceptance([load_json(p) for p in args.parts], [load_json(p) for p in args.sensitivity])
            write_json(output, result)
            for name, item in result["items"].items():
                print(f"{name}: {'PASS' if item['pass'] else 'FAIL'}")
            print(f"all items pass: {result['all_items_pass']}; wrote {output}")
            return 0
        if args.command == "search":
            acceptance = require_passing_acceptance(args.acceptance)
            result = assemble_search(acceptance, [load_json(p) for p in args.parts])
            write_json(output, result)
            print(f"{len(result['candidates'])} qualifying targets; wrote {output}")
            return 0
        if args.command == "search-part":
            require_passing_acceptance(args.acceptance)
        settings = {"ik_frame_updates": args.ik_frame_updates}
        options = {"census": bool(getattr(args, "census", False)), "targets": getattr(args, "targets", None)}
        ctx = Context(args, _log)
        if args.command == "acceptance-part":
            units = PART_FUNCTIONS[args.part](ctx, _targets(args.part, args.targets), {**settings, **options})
            write_json(
                output,
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "acceptance_part",
                    "part": args.part,
                    "options": options,
                    "provenance": provenance(ctx.inputs, settings),
                    "units": units,
                },
            )
        else:
            k, n = (int(v) for v in args.shard.split("/"))
            if not 1 <= k <= n:
                parser.error("--shard must be K/N with 1 <= K <= N")
            pool, pool_report = measure_pool(ctx)
            units = {}
            for entry in pool[k - 1 :: n]:
                key = f"tree{entry['tree']}_{entry['first_vertex']}"
                t0 = time.time()
                units[key] = search_target(ctx, entry, settings)
                units[key]["seconds"] = round(time.time() - t0, 1)
                _log(f"{key}: {units[key].get('excluded') or ('QUALIFIES' if units[key].get('qualifies') else 'no')}")
            write_json(
                output,
                {
                    "schema_version": SCHEMA_VERSION,
                    "kind": "search_part",
                    "shard": args.shard,
                    "pool": {**pool_report, "targets_after_exclusion": len(pool)},
                    "provenance": provenance(ctx.inputs, settings),
                    "units": units,
                },
            )
        print(f"wrote {output}")
        return 0
    except GateError as exc:
        parser.exit(1, f"gate failed, nothing written: {exc}\n")


if __name__ == "__main__":
    sys.exit(main())
