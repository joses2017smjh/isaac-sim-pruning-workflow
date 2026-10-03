#!/usr/bin/env python3
"""Project the visual jaw surrogate's cast sun shadow onto the tracker's appearance patch, through closure.

Design study for docs/SCOPE_AGREEMENT_ARM_CLOSED_LOOP_2026-10-02.md ("What would be built", item 1, the closure
frames 73-77). Every shadowed evening-14944 recording stops at frame 67, so no recording shows whether the closing
jaw's shadow crosses the tracker's 13 x 13 patch during closure. This tool predicts it by CPU geometry and validates
the same method on recordings where the outcome is known.

Model: the committed shadow model of tools/diagnose_appearance_loss.py, imported unchanged. Element (r, c) of the
patch of update i is the bark point hit by the camera ray of frame i-1 through p + (c - 6, r - 6) (pixel centres at
+0.5), on target when the first hit is tree geometry within 1 cm of the recorded depth. It is sunlit at frame k when
its camera-facing normal faces the sun and a ray toward the sun (1 mm back along the view ray, 0.2 mm along the
normal) is unblocked by the trees, orchard and FK-posed UR5e links, or by the two jaw boxes posed from the recorded
tool pose, proxy roll and closure progress (perception/jaw_self_mask.jaw_boxes, checked equal to the committed
diagnose_appearance_loss.jaw_cubes). Additions:
  * the shadow is split into the part cast by the jaw boxes alone and the rest;
  * a +-60 px window gives the distance from the patch to the nearest jaw-shadowed tree pixel and the jaw-shadow
    fraction of every footprint shifted by up to 3 px; the four committed shadow-ray offsets are re-evaluated;
  * a closure-progress sweep at a frozen tool pose, applied to the shadowed runs after their stop;
  * pixel checks: within one run during closure (tool and camera static, so the grey change from frame 72 isolates
    the jaw), and across renders (a shadow-kept run against its shadow-removed twin, warped into the same camera
    through the ray-cast bark point and the recorded poses).

Geometry: per run from its own scene.usda. Either a tools/extract_shadow_casters.py cache whose scene_sha256 matches
(--cache-root, outside the repository and any artifacts/ directory), or the committed extract() run in memory by the
Isaac venv python (--isaac-python, which has pxr) and streamed through a pipe, so nothing holding mesh data is
written. The output holds numbers and hashes only, never geometry.

Limits: the direct-sun hard shadow of the two-box surrogate only (no sky occlusion, interreflection or soft edges);
it predicts where and when the patch's sun state changes, not the NCC value or pass/fail at the 0.35 gate;
simulator renders, one per run; tools/validate_vision_sequence.py alone grades runs.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib.util
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TOOLS = REPO / "tools"
sys.path.insert(0, str(REPO / "source/isaaclab_pruning"))
from isaaclab_pruning.perception import jaw_self_mask as jsm


def _load_tool(name):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


dal = _load_tool("diagnose_appearance_loss")

SCHEMA_VERSION = 1
RUNS_DEFAULT = REPO / "artifacts/vision_robustness"
EVIDENCE = REPO / "docs/evidence/appearance_loss_diagnosis_2026-09-28.json"
SCOPE_DOC = "docs/SCOPE_AGREEMENT_ARM_CLOSED_LOOP_2026-10-02.md"
HALF = dal.HALF
WINDOW_HALF = 60
SHIFT_MAX = 3
MASKED_NCC_DRAWS = 500
JAW_BODY_EXCLUSION_MARGIN_PX = 3.0
PAIR_VISIBLE_M = 3e-3
SWEEP = np.round(np.linspace(0.0, 1.0, 61), 6)
CLOSURE_REFERENCE_FRAME = 72
CLOSURE_UPDATES = range(73, 78)
PREDICTION_FRAMES = range(71, 78)
GREY_THRESHOLDS = (20.0, 40.0)
MIN_JAW_FLIPS = 5
DIPS = (0.85, 0.90)
MARKER = b"@@SHADOW_CASTERS_STREAM@@\n"

SCOPE = (
    f"Design study for {SCOPE_DOC}: does the closing jaw's cast sun shadow reach the tracker's 13 x 13 "
    "appearance patch of evening 14944 during closure (frames 73-77), which no shadowed recording reaches? "
    "The model is the direct-sun hard shadow of the two-box visual jaw surrogate (the committed "
    "tools/diagnose_appearance_loss.py shadow model); it predicts where and when the patch's sun state changes, "
    "not the NCC value or pass/fail at the 0.35 gate. Simulator renders, one per run; no run is graded here."
)

#: Reference runs whose committed diagnosis caches the scene comparison and the reproduction use.
REFERENCE_RUNS = {
    "eve14944": "tree1-listed-evening-20260926/run_01_evening_tree1_v14944_baseline",
    "eve15004": "tree1-listed-evening-20260926/run_02_evening_tree1_v15004_baseline",
    "mor15004": "tree1-listed-morning-20260926/run_02_morning_tree1_v15004_baseline",
    "src14944_r1": "tree1-listed-repeat-r1-20260926/run_01_source_tree1_v14944_baseline",
    "src15004_r1": "tree1-listed-repeat-r1-20260926/run_02_source_tree1_v15004_baseline",
}
#: name -> (run, reference, first update, last update, roles)
CASES = {
    "repro_eve14944": (REFERENCE_RUNS["eve14944"], "eve14944", 55, 67, ("repro",)),
    "repro_mor15004": (REFERENCE_RUNS["mor15004"], "mor15004", 70, 75, ("repro",)),
    "v1_eve14944_jsa_r1": (
        "jaw-shadow-eve-a-r1-20260930/run_00_evening_tree1_v14944_baseline",
        "eve14944",
        55,
        67,
        ("settled",),
    ),
    "v1_eve14944_jsa_r2": (
        "jaw-shadow-eve-a-r2-20260930/run_00_evening_tree1_v14944_baseline",
        "eve14944",
        55,
        67,
        ("settled",),
    ),
    "v1_eve14944_dl_r1": (
        "depth-loop-eve-r1-20261001/run_00_evening_tree1_v14944_baseline_depth_appearance",
        "eve14944",
        55,
        67,
        ("settled",),
    ),
    "v1_eve14944_dl_r2": (
        "depth-loop-eve-r2-20261001/run_00_evening_tree1_v14944_baseline_depth_appearance",
        "eve14944",
        55,
        67,
        ("settled",),
    ),
    "p_eve14944_jsb_r1": (
        "jaw-shadow-eve-b-r1-20260930/run_00_evening_tree1_v14944_jaw_no_shadow",
        "eve14944",
        55,
        77,
        ("closure",),
    ),
    "p_eve14944_jsb_r2": (
        "jaw-shadow-eve-b-r2-20260930/run_00_evening_tree1_v14944_jaw_no_shadow",
        "eve14944",
        55,
        77,
        ("closure",),
    ),
    "v2_eve15004_dl_r1": (
        "depth-loop-eve-r1-20261001/run_01_evening_tree1_v15004_baseline_depth_appearance",
        "eve15004",
        55,
        77,
        ("closure",),
    ),
    "v2_eve15004_dl_r2": (
        "depth-loop-eve-r2-20261001/run_01_evening_tree1_v15004_baseline_depth_appearance",
        "eve15004",
        55,
        77,
        ("closure",),
    ),
    "v2c_eve15004_jsb_r1": (
        "jaw-shadow-eve-b-r1-20260930/run_01_evening_tree1_v15004_jaw_no_shadow",
        "eve15004",
        55,
        77,
        ("closure",),
    ),
    "v3_mor15004_dl_r1": (
        "depth-loop-mor-r1-20261001/run_00_morning_tree1_v15004_baseline_depth_appearance",
        "mor15004",
        55,
        77,
        ("closure",),
    ),
    "v3_mor15004_dl_r2": (
        "depth-loop-mor-r2-20261001/run_00_morning_tree1_v15004_baseline_depth_appearance",
        "mor15004",
        55,
        77,
        ("closure",),
    ),
    "v3c_mor15004_jsb_r1": (
        "jaw-shadow-mor-b-r1-20260930/run_00_morning_tree1_v15004_jaw_no_shadow",
        "mor15004",
        55,
        77,
        ("closure",),
    ),
    "v4_src14944_dl": (
        "depth-loop-src-20261001/run_00_source_tree1_v14944_baseline_depth_appearance",
        "src14944_r1",
        55,
        77,
        ("closure",),
    ),
    "v4_src15004_dl": (
        "depth-loop-src-20261001/run_01_source_tree1_v15004_baseline_depth_appearance",
        "src15004_r1",
        55,
        77,
        ("closure",),
    ),
}
#: name -> (shadow-kept run A, jaw-shadow-removed run B of the same target and light, first frame, last frame)
PAIRS = {
    "pair_eve14944_r1": (
        "jaw-shadow-eve-a-r1-20260930/run_00_evening_tree1_v14944_baseline",
        "jaw-shadow-eve-b-r1-20260930/run_00_evening_tree1_v14944_jaw_no_shadow",
        55,
        72,
    ),
    "pair_eve14944_r2": (
        "jaw-shadow-eve-a-r2-20260930/run_00_evening_tree1_v14944_baseline",
        "jaw-shadow-eve-b-r2-20260930/run_00_evening_tree1_v14944_jaw_no_shadow",
        55,
        72,
    ),
    "pair_eve15004_r1": (
        "depth-loop-eve-r1-20261001/run_01_evening_tree1_v15004_baseline_depth_appearance",
        "jaw-shadow-eve-b-r1-20260930/run_01_evening_tree1_v15004_jaw_no_shadow",
        55,
        77,
    ),
    "pair_eve15004_r2": (
        "depth-loop-eve-r2-20261001/run_01_evening_tree1_v15004_baseline_depth_appearance",
        "jaw-shadow-eve-b-r2-20260930/run_01_evening_tree1_v15004_jaw_no_shadow",
        55,
        77,
    ),
    "pair_mor15004_r1": (
        "depth-loop-mor-r1-20261001/run_00_morning_tree1_v15004_baseline_depth_appearance",
        "jaw-shadow-mor-b-r1-20260930/run_00_morning_tree1_v15004_jaw_no_shadow",
        55,
        77,
    ),
}
#: The no-shadow run whose recorded closure schedule is applied to the shadowed runs frozen after their stop.
SCHEDULE_SOURCE = "jaw-shadow-eve-b-r1-20260930/run_00_evening_tree1_v14944_jaw_no_shadow"
GROUPS = {
    "V1_evening_14944_approach_shadow_kept": [
        "v1_eve14944_jsa_r1",
        "v1_eve14944_jsa_r2",
        "v1_eve14944_dl_r1",
        "v1_eve14944_dl_r2",
    ],
    "V2_evening_15004_shadow_kept": ["v2_eve15004_dl_r1", "v2_eve15004_dl_r2"],
    "V3_morning_15004_shadow_kept": ["v3_mor15004_dl_r1", "v3_mor15004_dl_r2"],
    "V4_source_light_controls": ["v4_src14944_dl", "v4_src15004_dl"],
    "render_controls_jaw_shadow_removed": [
        "p_eve14944_jsb_r1",
        "p_eve14944_jsb_r2",
        "v2c_eve15004_jsb_r1",
        "v3c_mor15004_jsb_r1",
    ],
}
SHADOW_KEPT = ["V1_evening_14944_approach_shadow_kept", "V2_evening_15004_shadow_kept", "V3_morning_15004_shadow_kept"]
PREDICTION_CASES = ["p_eve14944_jsb_r1", "p_eve14944_jsb_r2"]
SETTLED_CASES = GROUPS["V1_evening_14944_approach_shadow_kept"]
DECISION_RULES = {
    "predicted_jaw_shadow_change": (
        f"at least {MIN_JAW_FLIPS} on-target patch elements change sun state between frames i-1 and i with a jaw box "
        "as the caster, at the default shadow-ray offset (variant: at least 1)"
    ),
    "observed_dip": f"recorded patch correlation below {DIPS[0]} (variant {DIPS[1]})",
    "render_controls": (
        "runs whose render has the jaw shadow removed are expected to show modelled flips without dips; they test the "
        "render counterfactual, not the model's location, and are tabulated apart"
    ),
    "disclosure": (
        "the thresholds were fixed after the recorded correlations had been seen and before the model outputs were "
        "read; the variants are reported so no single threshold carries the result"
    ),
}

# Isaac venv helper: run the committed extract() and stream (marker, JSON header line, deterministic npz bytes).
_EXTRACT_HELPER = """
import hashlib, importlib.util, io, json, sys
spec = importlib.util.spec_from_file_location("extract_shadow_casters", sys.argv[1])
esc = importlib.util.module_from_spec(spec)
spec.loader.exec_module(esc)
arrays, meta = esc.extract(sys.argv[2])
buffer = io.BytesIO()
esc.write_npz(buffer, arrays)
blob = buffer.getvalue()
meta_text = json.dumps(meta, indent=1, sort_keys=True) + "\\n"
header = {"meta_text": meta_text, "npz_bytes": len(blob)}
out = sys.stdout.buffer
out.write(b"@@SHADOW_CASTERS_STREAM@@\\n")
out.write((json.dumps(header) + "\\n").encode())
out.write(blob)
out.flush()
"""


class GateError(RuntimeError):
    """A validation gate failed; nothing is written."""


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_bytes(blob):
    return hashlib.sha256(blob).hexdigest()


def refuse_inside(path, what, repo_root=REPO):
    """Return the resolved path, or raise if it lies inside the repository or inside any artifacts/ directory."""
    resolved = Path(path).resolve()
    repo = Path(repo_root).resolve()
    if resolved == repo or repo in resolved.parents:
        raise ValueError(f"Refusing {what} inside the repository: {resolved}")
    if "artifacts" in resolved.parts:
        raise ValueError(f"Refusing {what} inside an artifacts/ directory: {resolved}")
    return resolved


def git(*args):
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True, check=True).stdout.strip()


def relative(path):
    path = Path(path).resolve()
    return str(path.relative_to(REPO)) if REPO in path.parents else str(path)


# ---------------------------------------------------------------------------------------------------- geometry
def npz_member_sha256(blob):
    """sha256 of every member's uncompressed bytes: content equality that does not depend on the zlib build."""
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        return {info.filename: sha256_bytes(archive.read(info)) for info in archive.infolist()}


class _MemText:
    def __init__(self, text):
        self.text = text

    def read_text(self):
        return self.text


class MemCache:
    """Stands in for a cache path so the committed dal.Geometry constructor runs unchanged on streamed bytes."""

    def __init__(self, blob, meta_text):
        self.blob, self.meta_text = blob, meta_text

    def with_suffix(self, suffix):
        return io.BytesIO(self.blob) if suffix == ".npz" else _MemText(self.meta_text)


class GeometrySource:
    """A matching extract_shadow_casters cache (read in place), else the committed extract() in memory."""

    def __init__(self, isaac_python=None, cache_root=None):
        self.isaac_python = isaac_python
        self.cache_root = None if cache_root is None else refuse_inside(cache_root, "a geometry cache root")

    def load(self, scene_path):
        scene_sha = sha256(scene_path)
        if self.cache_root is not None:
            for meta_path in sorted(self.cache_root.glob(f"*/{dal.CACHE_STEM}.json")):
                if json.loads(meta_path.read_text())["scene_sha256"] == scene_sha:
                    blob = meta_path.with_suffix(".npz").read_bytes()
                    return meta_path.with_suffix(""), self._provenance("cache", blob, meta_path.read_text())
        if not self.isaac_python:
            raise GateError(f"no geometry cache matches {scene_path} and no --isaac-python to extract it")
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        command = [self.isaac_python, "-c", _EXTRACT_HELPER, str(TOOLS / "extract_shadow_casters.py"), str(scene_path)]
        done = subprocess.run(command, capture_output=True, env=env, timeout=1800)
        if done.returncode != 0 or MARKER not in done.stdout:
            raise GateError(f"extraction failed ({done.returncode}): {done.stderr.decode(errors='replace')[-800:]}")
        stream = done.stdout[done.stdout.index(MARKER) + len(MARKER) :]
        line, _, blob = stream.partition(b"\n")
        header = json.loads(line)
        if len(blob) != header["npz_bytes"]:
            raise GateError("streamed geometry is truncated")
        meta = json.loads(header["meta_text"])
        if meta["scene_sha256"] != scene_sha:
            raise GateError("extracted geometry is not from this scene.usda")
        return MemCache(blob, header["meta_text"]), self._provenance("extract_in_memory", blob, header["meta_text"])

    @staticmethod
    def _provenance(source, blob, meta_text):
        return {
            "source": source,
            "npz_sha256": sha256_bytes(blob),
            "json_sha256": sha256_bytes(meta_text.encode()),
            "npz_member_sha256": npz_member_sha256(blob),
        }


def _owners(lines):
    """For each line, the names of the enclosing `def` prims (by indentation, as the usda writer formats them)."""
    chain, out = [], []
    for line in lines:
        indent = len(line) - len(line.lstrip(" "))
        match = re.match(r"\s*def\s+(?:\w+\s+)?\"([^\"]+)\"", line)
        if match:
            chain = [entry for entry in chain if entry[0] < indent]
            out.append([name for _, name in chain])
            chain.append((indent, match.group(1)))
        else:
            out.append([name for ind, name in chain if ind < indent])
    return out


def _allowed(owners, text):
    text = text.strip()
    if "PruningJawProxy" in owners:
        return "proxy"
    if owners and owners[-1] == "WristCamera" and text.startswith("matrix4d xformOp:transform"):
        return "wrist_camera_transform"
    if "SelectedSpur" in owners and "physics:kinematicEnabled" in text:
        return "spur_kinematic_flag"
    return None


def scene_line_diff(reference_lines, run_lines):
    """Kinds of differing usda lines; anything outside the jaw proxy, wrist camera transform and spur flag is listed."""
    owners_a, owners_b = _owners(reference_lines), _owners(run_lines)
    kinds, disallowed = Counter(), []
    matcher = difflib.SequenceMatcher(None, reference_lines, run_lines, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        for lines, owners, lo, hi, side in (
            (reference_lines, owners_a, i1, i2, "reference"),
            (run_lines, owners_b, j1, j2, "run"),
        ):
            for k in range(lo, hi):
                kind = _allowed(owners[k], lines[k])
                if kind is None:
                    disallowed.append({"side": side, "line": k + 1, "owners": owners[k][-3:]})
                else:
                    kinds[kind] += 1
    return {
        "differing_line_kinds": dict(kinds),
        "disallowed_differences": disallowed,
        "equivalent_geometry": not disallowed,
    }


# ---------------------------------------------------------------------------------------------------- runs
class ImageRun(dal.Run):
    """dal.Run's frame, image, depth and camera helpers without geometry (dal.Run.__init__ is not called)."""

    def __init__(self, path):  # noqa: D107
        self.path = Path(path)
        self.report = json.loads((self.path / "report.json").read_text())
        self.frames = json.loads((self.path / "frames.json").read_text())["frames"]
        if [frame["index"] for frame in self.frames] != list(range(len(self.frames))):
            raise GateError("frame indices are not 0..N-1")
        self.K = np.asarray(self.report["camera"]["wrist_intrinsics"], float)
        self.config = dict(self.frames[0]["live_vision"]["tracker_config"])
        self.config["roi_half_size_px"] = tuple(self.config["roi_half_size_px"])
        scene = self.report["blender_scene"]
        self.roll, self.radius = float(scene["visual_proxy_roll_rad"]), float(scene["target"]["radius_m"])

    def jaw_boxes(self, tool_pose, progress):
        return jsm.jaw_boxes(np.asarray(tool_pose, float), self.roll, float(progress), self.radius)

    def frame_jaw_boxes(self, k):
        frame = self.frames[k]
        return self.jaw_boxes(frame["tool_pose_wxyz"], frame.get("visual_jaw_closure_progress") or 0.0)

    def pose(self, k):
        frame = self.frames[k]
        return dal.rigid(frame["wrist_position_w_m"], frame["wrist_rotation_w_ros"])


class ProjectedRun(ImageRun):
    """ImageRun plus casters from the run's own scene.usda, split into jaw boxes and everything else."""

    def __init__(self, path, source):  # noqa: D107
        super().__init__(path)
        self.scene_sha256 = sha256(self.path / "scene.usda")
        self.cache, self.geometry_provenance = source.load(self.path / "scene.usda")
        self.geometry = dal.Geometry(self.cache, self.report)
        self._nojaw, self._jaw = {}, {}

    def nojaw_scene(self, frame):
        key = frame["index"]
        if key not in self._nojaw:
            links = self.geometry.links(frame["joint_position_rad"])
            residual = float(np.abs(links["ur5e__tool0"][:3, :3] - dal.qmat(frame["tool_pose_wxyz"][3:])).max())
            if residual > dal.FK_TOLERANCE:
                raise GateError(f"FK tool rotation differs from tool_pose_wxyz by {residual:.2e} at frame {key}")
            self.geometry.fk_residual_max = max(self.geometry.fk_residual_max, residual)
            meshes = dict(self.geometry.static)
            for name, (link, vertices, faces) in self.geometry.robot.items():
                meshes[name] = (vertices @ links[link][:3, :3].T + links[link][:3, 3], faces)
            if len(self._nojaw) >= 3:
                self._nojaw.pop(next(iter(self._nojaw)))
            self._nojaw[key] = dal.build_scene(meshes)
        return self._nojaw[key]

    def jaw_scene(self, tool_pose, progress):
        key = (tuple(np.round(np.asarray(tool_pose, float), 12)), round(float(progress), 9))
        if key not in self._jaw:
            if len(self._jaw) >= 256:
                self._jaw.pop(next(iter(self._jaw)))
            self._jaw[key] = dal.build_scene(jaw_meshes(self.jaw_boxes(tool_pose, progress)))
        return self._jaw[key]

    def jaw_model_check(self, k):
        """Max |corner difference| between jaw_self_mask.jaw_boxes and the committed dal.jaw_cubes at frame k."""
        boxes, cubes = self.frame_jaw_boxes(k), dal.jaw_cubes(self.frames[k], self.report)
        return max(
            float(np.abs(jsm.box_corners(boxes[0]) - cubes["jaw_left"][0]).max()),
            float(np.abs(jsm.box_corners(boxes[1]) - cubes["jaw_right"][0]).max()),
        )


def jaw_meshes(boxes):
    return {
        "jaw_left": (jsm.box_corners(boxes[0]), dal.CUBE_FACES),
        "jaw_right": (jsm.box_corners(boxes[1]), dal.CUBE_FACES),
    }


# ---------------------------------------------------------------------------------------------------- shadow helpers
def blocked(scene, origins, sun):
    """Whether the ray from each origin toward the sun hits the scene, and the first-hit group name."""
    t, groups, _ = dal.raycast(scene, origins, np.repeat(np.asarray(sun, float)[None], len(origins), 0))
    hit = np.isfinite(t)
    return hit, np.where(hit, groups, "none")


def classify(facing, blocked_other, blocked_jaw):
    """Sun state: lit, in the jaw boxes' shadow only, or shadowed by anything else (facing away is never lit)."""
    return {
        "facing": facing,
        "lit": facing & ~blocked_other & ~blocked_jaw,
        "jaw_shadow": facing & ~blocked_other & blocked_jaw,
        "other_shadow": facing & blocked_other,
    }


def shadow_origins(surface, back=dal.SHADOW_BACK_M, along=dal.SHADOW_NORMAL_M):
    return surface["P"] - surface["view"] * back + surface["n"] * along


def shadow_split(run, surface, frame, tool_pose=None, progress=None, back=dal.SHADOW_BACK_M, along=dal.SHADOW_NORMAL_M):
    """Sun state of surface points with the casters of `frame`; the jaw at tool_pose/progress when given."""
    sun = run.geometry.sun
    tool_pose = frame["tool_pose_wxyz"] if tool_pose is None else tool_pose
    progress = float(frame.get("visual_jaw_closure_progress") or 0.0) if progress is None else progress
    origins = shadow_origins(surface, back, along)
    b_other, c_other = blocked(run.nojaw_scene(frame), origins, sun)
    b_jaw, c_jaw = blocked(run.jaw_scene(tool_pose, progress), origins, sun)
    state = classify(surface["n"] @ sun > 0, b_other, b_jaw)
    state["caster"] = np.where(b_other, c_other, np.where(b_jaw, c_jaw, "none"))
    state["jaw_caster"] = c_jaw
    return state


def count_flips(on, previous, current):
    """On-target elements whose sun state changes; the caster is the blocker on the shadowed side."""
    to_lit = on & ~previous["lit"] & current["lit"]
    to_shadow = on & previous["lit"] & ~current["lit"]
    flip = to_lit | to_shadow
    casters = np.where(to_lit, previous["caster"], current["caster"])[flip]
    return {
        "to_lit": to_lit,
        "to_shadow": to_shadow,
        "flip": flip,
        "casters": dict(sorted(Counter(casters.tolist()).items())),
        "n_jaw": int(sum(str(c).startswith("jaw") for c in casters)),
    }


def grid(p, half=HALF):
    cols, rows = np.meshgrid(np.arange(-half, half + 1.0), np.arange(-half, half + 1.0))
    return p[0] + cols.ravel(), p[1] + rows.ravel()


def window(run, k, centre, half=WINDOW_HALF):
    """Camera rays of frame k through integer pixel indices within +-half of rint(centre), clipped to the image."""
    height, width = run.depth(k).shape
    cx, cy = np.rint(centre).astype(int)
    us, vs = np.meshgrid(
        np.arange(max(0, cx - half), min(width, cx + half + 1)),
        np.arange(max(0, cy - half), min(height, cy + half + 1)),
    )
    us, vs = us.ravel().astype(float), vs.ravel().astype(float)
    return us, vs, run.surface(k, us, vs)


def patch_distance(us, vs, p):
    """Euclidean pixel distance from pixel (u, v) to the patch square [p - 6, p + 6]^2 (0 inside)."""
    dx = np.maximum(0.0, np.abs(us - p[0]) - HALF)
    dy = np.maximum(0.0, np.abs(vs - p[1]) - HALF)
    return np.hypot(dx, dy)


def window_summary(us, vs, on, jaw_shadow, p):
    """Distance from the patch to the nearest jaw-shadowed on-target pixel and shifted-footprint fractions."""
    shadow = on & jaw_shadow
    distance = patch_distance(us, vs, p)
    out = {
        "n_window_on_tree": int(on.sum()),
        "n_window_jaw_shadow": int(shadow.sum()),
        "min_distance_px_patch_to_jaw_shadow": float(distance[shadow].min()) if shadow.any() else None,
        "n_jaw_shadow_within_3px_of_patch": int((shadow & (distance <= 3.0)).sum()),
    }
    if shadow.any():
        nearest = np.argmin(np.where(shadow, distance, np.inf))
        out["nearest_jaw_shadow_offset_px"] = [float(us[nearest] - p[0]), float(vs[nearest] - p[1])]
    cx, cy = np.rint(p)
    fractions = {}
    for dx in range(-SHIFT_MAX, SHIFT_MAX + 1):
        for dy in range(-SHIFT_MAX, SHIFT_MAX + 1):
            inside = (np.abs(us - (cx + dx)) <= HALF) & (np.abs(vs - (cy + dy)) <= HALF) & on
            fractions[(dx, dy)] = float((shadow & inside).sum() / inside.sum()) if inside.any() else None
    values = [v for v in fractions.values() if v is not None]
    out["shift_jaw_shadow_fraction_at_rint"] = fractions[(0, 0)]
    out["shift_jaw_shadow_fraction_max"] = max(values) if values else None
    out["shift_n_with_jaw_shadow"] = int(sum(v > 0 for v in values))
    out["shift_n"] = len(values)
    return out


def footprint_state_summary(on, state):
    n_on = int(on.sum())
    lit_wo_jaw = on & state["facing"] & ~state["other_shadow"]
    jaw = on & state["jaw_shadow"]
    return {
        "n_on_target": n_on,
        "n_sun_facing": int((on & state["facing"]).sum()),
        "n_sunlit_without_jaw": int(lit_wo_jaw.sum()),
        "n_jaw_shadow": int(jaw.sum()),
        "n_lit": int((on & state["lit"]).sum()),
        "jaw_shadow_fraction": float(jaw.sum() / n_on) if n_on else None,
        "jaw_shadow_fraction_of_sunlit_without_jaw": float(jaw.sum() / lit_wo_jaw.sum()) if lit_wo_jaw.any() else None,
        "lit_fraction": float((on & state["lit"]).sum() / n_on) if n_on else None,
        "other_shadow_fraction": float((on & state["other_shadow"]).sum() / n_on) if n_on else None,
        "jaw_casters": dict(sorted(Counter(state["jaw_caster"][jaw].tolist()).items()))
        if "jaw_caster" in state
        else {},
    }


def jaw_clearance(run, boxes, pose, pixel):
    """Min signed distance (px) of the patch element centres to the undilated jaw silhouette (negative inside)."""
    us, vs = grid(np.asarray(pixel, float))
    try:
        distance = jsm.jaw_signed_distance_px(boxes, run.K, pose, np.stack([us, vs], 1))
    except ValueError:
        return None
    return float(distance.min())


def measurement(frame):
    return (frame.get("live_vision") or {}).get("measurement") or {}


def propagated_from_stats(frame):
    """The propagated pixel a depth-appearance event recorded (its stats are a dict or their repr)."""
    event = measurement(frame).get("depth_appearance")
    if not isinstance(event, dict):
        return None
    stats = event.get("stats")
    if isinstance(stats, str):
        match = re.search(r"'propagated_pixel': \[([-\d.e]+), ([-\d.e]+)\]", stats)
        return [float(match.group(1)), float(match.group(2))] if match else None
    if isinstance(stats, dict):
        return stats.get("propagated_pixel")
    return None


def plain_tracker(run):
    return all("depth_appearance" not in measurement(frame) for frame in run.frames[:80])


# ---------------------------------------------------------------------------------------------------- per update
def _pixels(run, i, replayed):
    previous_m, current_m = measurement(run.frames[i - 1]), measurement(run.frames[i])
    update = replayed.get(i, {})
    p = np.asarray(update.get("prev_pixel", previous_m["pixel_xy"]), float)
    if "next_pixel" in update:
        return p, np.asarray(update["next_pixel"], float), "replay"
    stats = propagated_from_stats(run.frames[i])
    if current_m.get("state") == "tracking" or stats is not None:
        return p, np.asarray(stats or current_m["pixel_xy"], float), "recorded"
    return p, None, None


def _offset_sensitivity(run, surface, fp, fc):
    on, out = surface["on"], []
    for back, along in dal.SHADOW_OFFSET_SENSITIVITY:
        a = shadow_split(run, surface, fp, back=back, along=along)
        b = shadow_split(run, surface, fc, back=back, along=along)
        out.append(
            {
                "back_m": back,
                "along_normal_m": along,
                "n_shadow_to_lit": int((on & ~a["lit"] & b["lit"]).sum()),
                "n_lit_to_shadow": int((on & a["lit"] & ~b["lit"]).sum()),
                "jaw_shadow_fraction_prev": float((on & a["jaw_shadow"]).sum() / max(1, on.sum())),
                "jaw_shadow_fraction_cur": float((on & b["jaw_shadow"]).sum() / max(1, on.sum())),
            }
        )
    return out


def analyse_updates(run, first, last, replayed):
    rows = []
    for i in range(first, last + 1):
        if measurement(run.frames[i - 1]).get("state") != "tracking":
            break
        current_m = measurement(run.frames[i])
        p, q, q_source = _pixels(run, i, replayed)
        fp, fc = run.frames[i - 1], run.frames[i]
        surface = run.surface(i - 1, *grid(p))
        on = surface["on"]
        prev_state, cur_state = shadow_split(run, surface, fp), shadow_split(run, surface, fc)
        args = (surface["P"], surface["n"], surface["view"], run.geometry.sun)
        full_prev, _ = dal.sun_visibility(run.geometry.scene(fp), *args)
        full_cur, _ = dal.sun_visibility(run.geometry.scene(fc), *args)
        flips = count_flips(on, prev_state, cur_state)
        previous_patch = dal.patch(run.gray(i - 1), p)
        row = {
            "update": i,
            "phase": fc["phase"],
            "closure_progress_prev_cur": [fp.get("visual_jaw_closure_progress"), fc.get("visual_jaw_closure_progress")],
            "recorded_state": current_m.get("state"),
            "recorded_reason": current_m.get("reason"),
            "recorded_corr": current_m.get("patch_correlation"),
            "recorded_inliers": current_m.get("roundtrip_inlier_count"),
            "prev_pixel": p.tolist(),
            "propagated_pixel": None if q is None else q.tolist(),
            "propagated_pixel_source": q_source,
            "patch_prev_grey_std": float(previous_patch.std()),
            "depth_at_prev_pixel_m": float(run.depth(i - 1)[int(np.rint(p[1])), int(np.rint(p[0]))]),
            "split_equals_committed_full_scene": bool(
                np.array_equal(on & prev_state["lit"], on & full_prev)
                and np.array_equal(on & cur_state["lit"], on & full_cur)
            ),
            "prev": footprint_state_summary(on, prev_state),
            "cur": footprint_state_summary(on, cur_state),
            "n_shadow_to_lit": int(flips["to_lit"].sum()),
            "n_lit_to_shadow": int(flips["to_shadow"].sum()),
            "flip_casters": flips["casters"],
            "n_jaw_flips": flips["n_jaw"],
        }
        if flips["flip"].any() and q is not None:
            current_patch = dal.patch(run.gray(i), q)
            mask = flips["flip"].reshape(2 * HALF + 1, 2 * HALF + 1)
            masked = dal.masked_ncc(previous_patch, current_patch, mask, draws=MASKED_NCC_DRAWS)
            row["ncc_unmasked_recomputed"] = dal.ncc(previous_patch, current_patch)
            row["masked_ncc"] = {
                name: {key: masked[name][key] for key in ("n_kept", "ncc", "random_p95", "random_fraction_ge_true")}
                for name in ("undilated", "dilated")
            }
            ring = masked["dilated"]["shift_null"]["ring_ge2_wrap"]["max"]
            row["masked_ncc"]["dilated"]["shift_ring_ge2_max_wrap"] = ring
        wu, wv, wsurface = window(run, i - 1, p)
        row["window_prev"] = window_summary(wu, wv, wsurface["on"], shadow_split(run, wsurface, fp)["jaw_shadow"], p)
        row["window_cur"] = window_summary(wu, wv, wsurface["on"], shadow_split(run, wsurface, fc)["jaw_shadow"], p)
        boxes = run.frame_jaw_boxes(i)
        row["jaw_silhouette_min_signed_distance_px"] = jaw_clearance(run, boxes, run.pose(i), q if q is not None else p)
        closing = fc["phase"] == "simulated_closure" or (fc.get("visual_jaw_closure_progress") or 0) > 0
        if closing or current_m.get("state") != "tracking" or row["n_jaw_flips"] > 0:
            row["shadow_offset_sensitivity"] = _offset_sensitivity(run, surface, fp, fc)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------------------------- closure
def _rotation_deg(a, b):
    cosine = (np.trace(np.asarray(a).T @ np.asarray(b)) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def _grid_image(us, vs, values):
    r0, c0 = int(vs.min()), int(us.min())
    img = np.zeros((int(vs.max()) - r0 + 1, int(us.max()) - c0 + 1), bool)
    img[vs.astype(int) - r0, us.astype(int) - c0] = values
    return img


def mask_agreement(us, vs, observed, predicted, usable):
    """IoU and 1-px-tolerant precision and recall of two pixel masks on the window grid."""
    kernel = np.ones((3, 3), np.uint8)
    o_img, p_img, u_img = (_grid_image(us, vs, m) for m in (observed, predicted, usable))
    o_dil = cv2.dilate(o_img.astype(np.uint8), kernel).astype(bool) & u_img
    p_dil = cv2.dilate(p_img.astype(np.uint8), kernel).astype(bool) & u_img
    union = (o_img | p_img).sum()
    return {
        "n_observed": int(o_img.sum()),
        "n_predicted": int(p_img.sum()),
        "iou": float((o_img & p_img).sum() / union) if union else None,
        "precision_1px": float((p_img & o_dil).sum() / p_img.sum()) if p_img.any() else None,
        "recall_1px": float((o_img & p_dil).sum() / o_img.sum()) if o_img.any() else None,
    }


def _body_mask(run, k, rows_idx, cols_idx, shape):
    try:
        mask = jsm.jaw_mask(run.frame_jaw_boxes(k), run.K, run.pose(k), shape, margin_px=JAW_BODY_EXCLUSION_MARGIN_PX)
    except ValueError:
        return np.ones(len(rows_idx), bool)
    return mask[rows_idx, cols_idx]


def closure_pixel_check(run, reference=CLOSURE_REFERENCE_FRAME, frames=CLOSURE_UPDATES):
    """Recorded grey change from the reference frame against the modelled jaw-shadow change (static camera)."""
    ref = run.frames[reference]
    p = np.asarray(measurement(ref)["pixel_xy"], float)
    us, vs, surface = window(run, reference, p)
    ref_state = shadow_split(run, surface, ref)
    ref_gray = run.gray(reference)
    shape = ref_gray.shape
    rows_idx, cols_idx = vs.astype(int), us.astype(int)
    rows = []
    for k in frames:
        frame = run.frames[k]
        state = shadow_split(run, surface, frame)
        delta = run.gray(k)[rows_idx, cols_idx] - ref_gray[rows_idx, cols_idx]
        body = _body_mask(run, reference, rows_idx, cols_idx, shape) | _body_mask(run, k, rows_idx, cols_idx, shape)
        usable = surface["on"] & ~body
        p2s = usable & ref_state["lit"] & ~state["lit"]
        s2l = usable & ~ref_state["lit"] & state["lit"]
        same = usable & ~p2s & ~s2l
        motion = np.asarray(frame["wrist_position_w_m"]) - np.asarray(ref["wrist_position_w_m"])
        row = {
            "frame": k,
            "closure_progress": frame.get("visual_jaw_closure_progress"),
            "camera_motion_from_reference_m": float(np.linalg.norm(motion)),
            "camera_rotation_from_reference_deg": _rotation_deg(
                ref["wrist_rotation_w_ros"], frame["wrist_rotation_w_ros"]
            ),
            "n_usable": int(usable.sum()),
            "n_excluded_jaw_body": int((surface["on"] & body).sum()),
            "n_pred_lit_to_shadow": int(p2s.sum()),
            "n_pred_shadow_to_lit": int(s2l.sum()),
            "median_delta_pred_lit_to_shadow": float(np.median(delta[p2s])) if p2s.any() else None,
            "median_delta_pred_shadow_to_lit": float(np.median(delta[s2l])) if s2l.any() else None,
            "median_abs_delta_pred_unchanged": float(np.median(np.abs(delta[same]))) if same.any() else None,
            "p95_abs_delta_pred_unchanged": float(np.percentile(np.abs(delta[same]), 95)) if same.any() else None,
            "patch_pred_jaw_shadow_change": int(((p2s | s2l) & (patch_distance(us, vs, p) == 0)).sum()),
        }
        for threshold in GREY_THRESHOLDS:
            row[f"darkening_T{int(threshold)}"] = mask_agreement(us, vs, usable & (delta < -threshold), p2s, usable)
            row[f"brightening_T{int(threshold)}"] = mask_agreement(us, vs, usable & (delta > threshold), s2l, usable)
        rows.append(row)
    return {"reference_frame": reference, "reference_pixel": p.tolist(), "frames": rows}


def closure_sweep(run, reference, pixel, tool_pose=None, schedule=None):
    """Jaw-shadow fraction of the footprint at `pixel` (camera of `reference`) for closure progress 0..1."""
    frame = run.frames[reference]
    tool_pose = frame["tool_pose_wxyz"] if tool_pose is None else tool_pose
    pixel = np.asarray(pixel, float)
    surface = run.surface(reference, *grid(pixel))
    wu, wv, wsurface = window(run, reference, pixel)
    pose = run.pose(reference)
    rows = []
    for progress in sorted(set(SWEEP.tolist()) | set(schedule.values() if schedule else [])):
        state = shadow_split(run, surface, frame, tool_pose=tool_pose, progress=progress)
        wstate = shadow_split(run, wsurface, frame, tool_pose=tool_pose, progress=progress)
        row = {"progress": progress} | footprint_state_summary(surface["on"], state)
        row["window"] = window_summary(wu, wv, wsurface["on"], wstate["jaw_shadow"], pixel)
        boxes = run.jaw_boxes(tool_pose, progress)
        row["jaw_silhouette_min_signed_distance_px"] = jaw_clearance(run, boxes, pose, pixel)
        rows.append(row)
    return {
        "reference_frame": reference,
        "pixel": pixel.tolist(),
        "tool_pose_wxyz": list(map(float, tool_pose)),
        "depth_at_pixel_m": float(run.depth(reference)[int(np.rint(pixel[1])), int(np.rint(pixel[0]))]),
        "rows": rows,
    }


def schedule_from(frames):
    """The recorded closure progress of frames 70-78."""
    return {k: float(frames[k].get("visual_jaw_closure_progress") or 0.0) for k in range(70, 79)}


def tracker_pixel_of_world_point(run, k, world):
    """Pixel of a tracker world point in frame k: the tracker back-projects through K without the +0.5 centre
    (visual_servo.project_depth_to_world), so dal.project's -0.5 is undone."""
    pixel, z = run.project(k, np.asarray(world, float)[None])
    return pixel[0] + 0.5, float(z[0])


def settled_continuation(run, schedule):
    """Shadowed run frozen after its stop: closure applied at its settled pose; footprint at its last tracked point."""
    last = max(k for k in range(70) if measurement(run.frames[k]).get("state") == "tracking")
    world = np.asarray(measurement(run.frames[last])["target_position_world_m"], float)
    pixel, z = tracker_pixel_of_world_point(run, CLOSURE_REFERENCE_FRAME, world)
    depth = run.depth(CLOSURE_REFERENCE_FRAME)
    return {
        "last_tracking_frame": last,
        "last_tracked_world_point_m": world.tolist(),
        "projected_pixel_at_reference": pixel.tolist(),
        "projected_optical_z_m": z,
        "recorded_depth_at_projection_m": float(depth[int(np.rint(pixel[1])), int(np.rint(pixel[0]))]),
        "tool_pose_reference_wxyz": run.frames[CLOSURE_REFERENCE_FRAME]["tool_pose_wxyz"],
        "sweep": closure_sweep(run, CLOSURE_REFERENCE_FRAME, pixel, schedule=schedule),
    }


# ---------------------------------------------------------------------------------------------------- render pairs
def track_centre(run, k):
    """The tracked pixel at frame k, or (after a stop) the last tracked world point projected into frame k."""
    m = measurement(run.frames[k])
    if m.get("state") == "tracking":
        return np.asarray(m["pixel_xy"], float), "tracked"
    last = max(j for j in range(k + 1) if measurement(run.frames[j]).get("state") == "tracking")
    world = measurement(run.frames[last])["target_position_world_m"]
    return tracker_pixel_of_world_point(run, k, world)[0], f"projected_from_{last}"


def pair_check(name, runs_root, source, inputs):
    """Observed jaw shadow = run A (shadow kept) darker than run B (shadow removed) warped into A's camera through
    the exact ray-cast bark point and B's recorded pose; compared with A's modelled jaw shadow."""
    path_a, path_b, first, last = PAIRS[name]
    a, b = ProjectedRun(runs_root / path_a, source), ImageRun(runs_root / path_b)
    record_inputs(inputs, runs_root / path_a)
    record_inputs(inputs, runs_root / path_b, scene=False)
    kernel = np.ones((3, 3), np.uint8)
    rows = []
    for k in range(first, last + 1):
        fa, fb = a.frames[k], b.frames[k]
        centre, centre_source = track_centre(a, k)
        us, vs, surface = window(a, k, centre)
        state = shadow_split(a, surface, fa)
        depth_b = b.depth(k)
        height, width = depth_b.shape
        uv_b, z_b = b.project(k, surface["P"])
        inside = (uv_b[:, 0] >= 0) & (uv_b[:, 0] <= width - 1) & (uv_b[:, 1] >= 0) & (uv_b[:, 1] <= height - 1)
        rb = np.clip(np.rint(uv_b[:, 1]).astype(int), 0, height - 1)
        cb = np.clip(np.rint(uv_b[:, 0]).astype(int), 0, width - 1)
        with np.errstate(invalid="ignore"):
            visible_b = inside & (np.abs(depth_b[rb, cb] - z_b) < PAIR_VISIBLE_M)
        maps = np.nan_to_num(uv_b.astype(np.float32), nan=-1.0).reshape(1, -1, 2)
        grey_b = cv2.remap(b.gray(k), maps[..., 0].copy(), maps[..., 1].copy(), cv2.INTER_LINEAR).ravel()
        delta = a.gray(k)[vs.astype(int), us.astype(int)] - grey_b
        body = _body_mask(a, k, vs.astype(int), us.astype(int), (height, width)) | _body_mask(
            b, k, rb, cb, (height, width)
        )
        on_img = _grid_image(us, vs, surface["on"])
        interior = cv2.erode(on_img.astype(np.uint8), kernel).astype(bool)[
            vs.astype(int) - int(vs.min()), us.astype(int) - int(us.min())
        ]
        usable = surface["on"] & interior & visible_b & ~body
        predicted = usable & state["jaw_shadow"]
        lit_both = usable & state["lit"]
        in_patch = patch_distance(us, vs, centre) == 0
        n_patch = max(1, int((usable & in_patch).sum()))
        row = {
            "frame": k,
            "centre_source": centre_source,
            "centre": centre.tolist(),
            "closure_progress_a_b": [fa.get("visual_jaw_closure_progress"), fb.get("visual_jaw_closure_progress")],
            "tool_offset_a_b_m": float(
                np.linalg.norm(np.asarray(fa["tool_pose_wxyz"][:3]) - np.asarray(fb["tool_pose_wxyz"][:3]))
            ),
            "n_usable": int(usable.sum()),
            "n_predicted_jaw_shadow": int(predicted.sum()),
            "median_delta_predicted_jaw_shadow": float(np.median(delta[predicted])) if predicted.any() else None,
            "median_delta_predicted_lit": float(np.median(delta[lit_both])) if lit_both.any() else None,
            "p05_delta_predicted_lit": float(np.percentile(delta[lit_both], 5)) if lit_both.any() else None,
            "patch_n_usable": int((usable & in_patch).sum()),
            "patch_predicted_jaw_shadow_fraction": float((predicted & in_patch).sum() / n_patch),
            "predicted_min_distance_px_to_patch": (
                float(patch_distance(us, vs, centre)[predicted].min()) if predicted.any() else None
            ),
        }
        for threshold in GREY_THRESHOLDS:
            observed = usable & (delta < -threshold)
            agreement = mask_agreement(us, vs, observed, predicted, usable)
            row[f"T{int(threshold)}"] = {
                "n_observed_darkening": agreement["n_observed"],
                "iou": agreement["iou"],
                "precision_1px": agreement["precision_1px"],
                "recall_1px": agreement["recall_1px"],
                "patch_observed_darkening_fraction": float((observed & in_patch).sum() / n_patch),
                "observed_min_distance_px_to_patch": (
                    float(patch_distance(us, vs, centre)[observed].min()) if observed.any() else None
                ),
            }
        rows.append(row)
    return {
        "case": name,
        "run_shadow_kept": path_a,
        "run_shadow_removed": path_b,
        "geometry": a.geometry_provenance,
        "frames": rows,
    }


# ---------------------------------------------------------------------------------------------------- driver
def record_inputs(inputs, run_dir, scene=True):
    for key in ("report.json", "frames.json", *(("scene.usda",) if scene else ())):
        path = Path(run_dir) / key
        inputs.setdefault(relative(path), sha256(path))


def run_case(name, runs_root, source, inputs):
    path, reference, first, last, roles = CASES[name]
    run_dir = runs_root / path
    record_inputs(inputs, run_dir)
    record_inputs(inputs, runs_root / REFERENCE_RUNS[reference], scene=True)
    reference_scene = (runs_root / REFERENCE_RUNS[reference] / "scene.usda").read_text().splitlines()
    equivalence = scene_line_diff(reference_scene, (run_dir / "scene.usda").read_text().splitlines())
    run = ProjectedRun(run_dir, source)
    out = {
        "case": name,
        "run": path,
        "reference_run": REFERENCE_RUNS[reference],
        "window": [first, last],
        "roles": list(roles),
        "scene_difference_from_reference_run": equivalence,
        "geometry": run.geometry_provenance,
        "task_outcome": run.report.get("task_outcome"),
        "sun_direction": run.geometry.sun.tolist(),
        "daylight_report": (run.report["blender_scene"].get("daylight") or {}).get("world_direction_to_sun"),
        "jaw_model_max_corner_difference_m": max(run.jaw_model_check(k) for k in range(first - 1, last + 1)),
        "plain_tracker": plain_tracker(run),
    }
    replayed = {}
    if out["plain_tracker"]:
        try:
            replayed = dal.replay(run, last)
        except dal.GateError as exc:
            out["replay"] = {"ok": False, "error": str(exc)}
        else:
            corr = [abs(u["corr"] - u["recorded_corr"]) for u in replayed.values() if u["corr"] is not None]
            out["replay"] = {"ok": True, "updates": len(replayed), "max_abs_corr_difference": max(corr, default=None)}
    out["updates"] = analyse_updates(run, first, last, replayed)
    if "closure" in roles:
        out["closure_pixel_check"] = closure_pixel_check(run)
        ref_m = measurement(run.frames[CLOSURE_REFERENCE_FRAME])
        if ref_m.get("target_position_world_m") is not None:
            back = tracker_pixel_of_world_point(run, CLOSURE_REFERENCE_FRAME, ref_m["target_position_world_m"])[0]
            out["tracker_world_point_reprojection_px"] = float(np.linalg.norm(back - np.asarray(ref_m["pixel_xy"])))
        schedule = schedule_from(run.frames)
        out["closure_sweep"] = closure_sweep(run, CLOSURE_REFERENCE_FRAME, ref_m["pixel_xy"], schedule=schedule)
    if "settled" in roles:
        record_inputs(inputs, runs_root / SCHEDULE_SOURCE, scene=False)
        schedule = schedule_from(json.loads((runs_root / SCHEDULE_SOURCE / "frames.json").read_text())["frames"])
        out["settled_continuation"] = settled_continuation(run, schedule)
    out["fk_rotation_residual_max"] = run.geometry.fk_residual_max
    return out


# ---------------------------------------------------------------------------------------------------- summaries
def _round(value, digits):
    return round(value, digits) if isinstance(value, float) else value


def value_range(values, digits=2):
    values = [v for v in values if v is not None]
    return [_round(min(values), digits), _round(max(values), digits)] if values else None


def predicts(row, min_flips=MIN_JAW_FLIPS):
    return row["n_jaw_flips"] >= min_flips


def dips(row, dip):
    return row["recorded_corr"] is not None and row["recorded_corr"] < dip


def confusion(rows, dip, min_flips=MIN_JAW_FLIPS, updates=None):
    """Updates by predicted jaw-shadow change against observed dip; updates without a correlation are skipped."""
    table = {"tp": [], "fp": [], "fn": [], "tn": []}
    for row in rows:
        if row["recorded_corr"] is None or (updates is not None and row["update"] not in updates):
            continue
        observed = dips(row, dip)
        key = ("tp" if observed else "fp") if predicts(row, min_flips) else ("fn" if observed else "tn")
        table[key].append(row["update"])
    return table | {f"{key}_n": len(value) for key, value in list(table.items())}


def pooled(cases, names, dip, min_flips=MIN_JAW_FLIPS, updates=None):
    total = {"tp": 0, "fp": 0, "fn": 0, "tn": 0}
    for name in names:
        table = confusion(cases[name]["updates"], dip, min_flips, updates)
        for key in total:
            total[key] += table[f"{key}_n"]
    return total


def update_table(case):
    rows = []
    for row in case["updates"]:
        masked = (row.get("masked_ncc") or {}).get("dilated") or {}
        sens = row.get("shadow_offset_sensitivity")
        rows.append(
            {
                "update": row["update"],
                "phase": row["phase"],
                "progress_prev_cur": row["closure_progress_prev_cur"],
                "recorded_state": row["recorded_state"],
                "recorded_corr": row["recorded_corr"],
                "recorded_inliers": row["recorded_inliers"],
                "patch_prev_grey_std": row["patch_prev_grey_std"],
                "n_on_target": row["prev"]["n_on_target"],
                "jaw_shadow_fraction_prev": row["prev"]["jaw_shadow_fraction"],
                "jaw_shadow_fraction_cur": row["cur"]["jaw_shadow_fraction"],
                "lit_fraction_prev": row["prev"]["lit_fraction"],
                "lit_fraction_cur": row["cur"]["lit_fraction"],
                "n_lit_to_shadow": row["n_lit_to_shadow"],
                "n_shadow_to_lit": row["n_shadow_to_lit"],
                "n_jaw_flips": row["n_jaw_flips"],
                "flip_casters": row["flip_casters"],
                "window_min_distance_px_prev": row["window_prev"]["min_distance_px_patch_to_jaw_shadow"],
                "window_min_distance_px_cur": row["window_cur"]["min_distance_px_patch_to_jaw_shadow"],
                "shift_max_jaw_shadow_fraction_cur": row["window_cur"]["shift_jaw_shadow_fraction_max"],
                "masked_ncc_dilated": masked.get("ncc"),
                "masked_ncc_dilated_kept": masked.get("n_kept"),
                "masked_ncc_random_p95": masked.get("random_p95"),
                "jaw_silhouette_min_signed_distance_px": row["jaw_silhouette_min_signed_distance_px"],
                "offset_sensitivity_jaw_flips": (
                    [s["n_lit_to_shadow"] + s["n_shadow_to_lit"] for s in sens] if sens else None
                ),
            }
        )
    return rows


def reproduction(cases):
    evidence = {run["run"]: run for run in json.loads(EVIDENCE.read_text())["runs"]}
    out = {}
    for name in ("repro_eve14944", "repro_mor15004"):
        case = cases[name]
        committed_run = evidence[case["run"]]
        rows = {row["update"]: row for row in committed_run["updates"] if "n_on_target" in row}
        differences, compared = [], 0
        for row in case["updates"]:
            committed = rows.get(row["update"])
            if committed is None:
                continue
            compared += 1
            mine = {
                "n_on_target": row["prev"]["n_on_target"],
                "n_shadow_to_lit": row["n_shadow_to_lit"],
                "n_lit_to_shadow": row["n_lit_to_shadow"],
                "flip_casters": row["flip_casters"],
                "lit_fraction_prev": round(row["prev"]["lit_fraction"], 9),
                "lit_fraction_cur": round(row["cur"]["lit_fraction"], 9),
            }
            theirs = {k: round(committed[k], 9) if isinstance(committed[k], float) else committed[k] for k in mine}
            if mine != theirs:
                differences.append({"update": row["update"], "this_tool": mine, "committed": theirs})
        stop = next((r for r in case["updates"] if r["update"] == committed_run["stop_update"]), {})
        committed_stop = rows.get(committed_run["stop_update"], {})
        out[name] = {
            "run": case["run"],
            "updates_compared": compared,
            "identical": not differences and compared > 0,
            "differences": differences,
            "masked_ncc_dilated_at_stop": [
                ((stop.get("masked_ncc") or {}).get("dilated") or {}).get("ncc"),
                ((committed_stop.get("masked_ncc") or {}).get("dilated") or {}).get("ncc"),
            ],
            "offset_counts_at_stop": [
                [s["n_shadow_to_lit"] + s["n_lit_to_shadow"] for s in stop.get("shadow_offset_sensitivity") or []],
                [
                    s["n_shadow_to_lit"] + s["n_lit_to_shadow"]
                    for s in committed_run.get("shadow_offset_sensitivity_at_stop") or []
                ],
            ],
            "split_equals_committed_full_scene_all": all(
                r["split_equals_committed_full_scene"] for r in case["updates"]
            ),
            "replay": case.get("replay"),
            "geometry_json_equals_committed_cache": (
                case["geometry"]["json_sha256"] == committed_run["sha256"]["shadow_casters.json"]
            ),
        }
    return out


def geometry_consistency(cases):
    """Per target, whether the static casters (trees, orchard) are byte-identical in content across all runs."""
    by_target = {}
    for case in cases.values():
        target = "14944" if "v14944" in case.get("run", case.get("run_shadow_kept", "")) else "15004"
        members = case["geometry"]["npz_member_sha256"]
        static = tuple(members[k] for k in ("static_vertices.npy", "static_faces.npy", "static_groups.npy"))
        robot = tuple(members[k] for k in sorted(members) if k.startswith("robot_"))
        entry = by_target.setdefault(target, {"static": set(), "robot": set(), "n_runs": 0})
        entry["static"].add(static)
        entry["robot"].add(robot)
        entry["n_runs"] += 1
    return {
        target: {
            "n_runs": entry["n_runs"],
            "static_casters_identical_across_runs": len(entry["static"]) == 1,
            "robot_link_meshes_identical_across_runs": len(entry["robot"]) == 1,
        }
        for target, entry in sorted(by_target.items())
    }


def validation(cases):
    out = {}
    for group, names in GROUPS.items():
        out[group] = {}
        for name in names:
            case = cases[name]
            rows = update_table(case)
            entry = {
                "run": case["run"],
                "reference_run": case["reference_run"],
                "scene_difference_from_reference_run": {
                    k: case["scene_difference_from_reference_run"][k]
                    for k in ("equivalent_geometry", "differing_line_kinds")
                },
                "replay": case.get("replay"),
                "geometry_source": case["geometry"]["source"],
                "jaw_model_max_corner_difference_m": case["jaw_model_max_corner_difference_m"],
                "fk_rotation_residual_max": case["fk_rotation_residual_max"],
                "sun_direction": case["sun_direction"],
                "updates": rows,
            }
            for dip in DIPS:
                entry[f"confusion_minflips{MIN_JAW_FLIPS}_dip{dip}"] = confusion(rows, dip)
                entry[f"confusion_minflips1_dip{dip}"] = confusion(rows, dip, min_flips=1)
                entry[f"closure_confusion_minflips{MIN_JAW_FLIPS}_dip{dip}"] = confusion(
                    rows, dip, updates=CLOSURE_UPDATES
                )
            if case.get("closure_pixel_check"):
                entry["closure_pixel_check"] = case["closure_pixel_check"]
            out[group][name] = entry
    return out


def pooled_confusion(cases):
    names = lambda groups: [n for g in groups for n in GROUPS[g]]  # noqa: E731
    out = {}
    for dip in DIPS:
        out[f"shadow_kept_dip{dip}"] = pooled(cases, names(SHADOW_KEPT), dip)
        out[f"shadow_kept_closure_73_77_dip{dip}"] = pooled(cases, names(SHADOW_KEPT[1:]), dip, updates=CLOSURE_UPDATES)
        out[f"source_controls_dip{dip}"] = pooled(cases, names(["V4_source_light_controls"]), dip)
    for dip in DIPS:
        out[f"shadow_kept_minflips1_dip{dip}"] = pooled(cases, names(SHADOW_KEPT), dip, min_flips=1)
        out[f"shadow_kept_closure_73_77_minflips1_dip{dip}"] = pooled(
            cases, names(SHADOW_KEPT[1:]), dip, min_flips=1, updates=CLOSURE_UPDATES
        )
    return out


def pair_summary(pairs):
    out = {}
    for name, case in pairs.items():
        frames = []
        for row in case["frames"]:
            t20, t40 = row["T20"], row["T40"]
            frames.append(
                {
                    "frame": row["frame"],
                    "centre_source": row["centre_source"],
                    "progress_a_b": row["closure_progress_a_b"],
                    "tool_offset_a_b_mm": round(1000 * row["tool_offset_a_b_m"], 3),
                    "n_usable": row["n_usable"],
                    "n_predicted": row["n_predicted_jaw_shadow"],
                    "n_observed_T20": t20["n_observed_darkening"],
                    "n_observed_T40": t40["n_observed_darkening"],
                    "iou_T20": t20["iou"],
                    "iou_T40": t40["iou"],
                    "precision_1px_T20": t20["precision_1px"],
                    "recall_1px_T20": t20["recall_1px"],
                    "precision_1px_T40": t40["precision_1px"],
                    "recall_1px_T40": t40["recall_1px"],
                    "median_delta_predicted_jaw_shadow": row["median_delta_predicted_jaw_shadow"],
                    "median_delta_predicted_lit": row["median_delta_predicted_lit"],
                    "patch_predicted_jaw_shadow_fraction": row["patch_predicted_jaw_shadow_fraction"],
                    "patch_observed_darkening_fraction_T20": t20["patch_observed_darkening_fraction"],
                    "patch_observed_darkening_fraction_T40": t40["patch_observed_darkening_fraction"],
                    "predicted_min_distance_px_to_patch": row["predicted_min_distance_px_to_patch"],
                    "observed_min_distance_px_to_patch_T20": t20["observed_min_distance_px_to_patch"],
                    "observed_min_distance_px_to_patch_T40": t40["observed_min_distance_px_to_patch"],
                }
            )
        sizable = [f for f in frames if f["n_predicted"] >= 20 or f["n_observed_T40"] >= 20]

        def median(values):
            values = sorted(v for v in values if v is not None)
            return values[len(values) // 2] if values else None

        out[name] = {
            "run_shadow_kept": case["run_shadow_kept"],
            "run_shadow_removed": case["run_shadow_removed"],
            "geometry_source": case["geometry"]["source"],
            "n_frames_with_sizable_shadow": len(sizable),
            "median_iou_T40_sizable": median(f["iou_T40"] for f in sizable),
            "median_precision_1px_T40_sizable": median(f["precision_1px_T40"] for f in sizable),
            "median_recall_1px_T40_sizable": median(f["recall_1px_T40"] for f in sizable),
            "frames": frames,
        }
    return out


def sweep_summary(sweep):
    rows = sweep["rows"]
    distances = [r["window"]["min_distance_px_patch_to_jaw_shadow"] for r in rows]
    return {
        "reference_frame": sweep["reference_frame"],
        "pixel": sweep["pixel"],
        "depth_at_pixel_m": sweep["depth_at_pixel_m"],
        "n_progress_values": len(rows),
        "max_jaw_shadow_fraction": max((r["jaw_shadow_fraction"] or 0) for r in rows),
        "progress_with_jaw_shadow_on_footprint": [r["progress"] for r in rows if (r["jaw_shadow_fraction"] or 0) > 0],
        "progress_with_jaw_shadow_on_any_3px_shift": [
            r["progress"] for r in rows if (r["window"]["shift_jaw_shadow_fraction_max"] or 0) > 0
        ],
        "min_distance_px_over_sweep": min((d for d in distances if d is not None), default=None),
        "rows": [
            {
                "progress": r["progress"],
                "n_on_target": r["n_on_target"],
                "n_sunlit_without_jaw": r["n_sunlit_without_jaw"],
                "jaw_shadow_fraction": r["jaw_shadow_fraction"],
                "lit_fraction": r["lit_fraction"],
                "min_distance_px": r["window"]["min_distance_px_patch_to_jaw_shadow"],
                "nearest_offset_px": r["window"].get("nearest_jaw_shadow_offset_px"),
                "shift_max": r["window"]["shift_jaw_shadow_fraction_max"],
                "jaw_silhouette_min_signed_distance_px": r["jaw_silhouette_min_signed_distance_px"],
            }
            for r in rows
        ],
    }


def prediction_table(cases):
    """Frames 71-77 in six geometries: both no-shadow runs (their own closure) and the four shadowed runs frozen after
    their stop (last tracked world point, settled frame-72 pose, the no-shadow run's recorded closure schedule)."""
    geometries, schedule = {}, None
    for name in PREDICTION_CASES:
        rows = {r["update"]: r for r in cases[name]["updates"]}
        schedule = {k: (rows[k]["closure_progress_prev_cur"][1] or 0.0) for k in PREDICTION_FRAMES}
        per = {}
        for k in PREDICTION_FRAMES:
            r = rows[k]
            sens = r.get("shadow_offset_sensitivity") or []
            per[k] = {
                "closure_progress": schedule[k],
                "tracked_pixel": r["propagated_pixel"],
                "jaw_shadow_fraction": r["cur"]["jaw_shadow_fraction"],
                "lit_fraction": r["cur"]["lit_fraction"],
                "n_on_target": r["cur"]["n_on_target"],
                "new_jaw_shadowed_elements": r["n_lit_to_shadow"] if r["flip_casters"] else 0,
                "update_jaw_flips": r["n_jaw_flips"],
                "min_distance_px_patch_to_jaw_shadow": r["window_cur"]["min_distance_px_patch_to_jaw_shadow"],
                "shift_max_jaw_shadow_fraction_3px": r["window_cur"]["shift_jaw_shadow_fraction_max"],
                "offset_range_jaw_shadow_fraction": value_range([x["jaw_shadow_fraction_cur"] for x in sens], 3),
                "offset_range_update_flips": value_range([x["n_lit_to_shadow"] + x["n_shadow_to_lit"] for x in sens]),
                "jaw_silhouette_min_distance_px": r["jaw_silhouette_min_signed_distance_px"],
                "recorded_corr_in_no_shadow_render": r["recorded_corr"],
            }
        geometries[name] = {"run": cases[name]["run"], "kind": "no-shadow run, own closure", "frames": per}
    for name in SETTLED_CASES:
        settled = cases[name]["settled_continuation"]
        rows = {round(r["progress"], 6): r for r in settled["sweep"]["rows"]}
        per, previous = {}, None
        for k in PREDICTION_FRAMES:
            r = rows[round(schedule[k], 6)]
            per[k] = {
                "closure_progress": schedule[k],
                "jaw_shadow_fraction": r["jaw_shadow_fraction"],
                "lit_fraction": r["lit_fraction"],
                "n_on_target": r["n_on_target"],
                "new_jaw_shadowed_elements": (
                    None if previous is None else round((r["jaw_shadow_fraction"] - previous) * r["n_on_target"])
                ),
                "min_distance_px_patch_to_jaw_shadow": r["window"]["min_distance_px_patch_to_jaw_shadow"],
                "shift_max_jaw_shadow_fraction_3px": r["window"]["shift_jaw_shadow_fraction_max"],
                "jaw_silhouette_min_distance_px": r["jaw_silhouette_min_signed_distance_px"],
            }
            previous = r["jaw_shadow_fraction"]
        geometries[name] = {
            "run": cases[name]["run"],
            "kind": f"shadowed run frozen after its stop; closure schedule of {SCHEDULE_SOURCE}",
            "footprint_pixel_at_72": settled["projected_pixel_at_reference"],
            "sweep_summary": sweep_summary(settled["sweep"]),
            "frames": per,
        }
    for name in PREDICTION_CASES:
        geometries[name]["sweep_summary"] = sweep_summary(cases[name]["closure_sweep"])
    by_frame = {}
    for k in PREDICTION_FRAMES:
        rows = [g["frames"][k] for g in geometries.values()]
        flips = [v for r in rows if r.get("offset_range_update_flips") for v in r["offset_range_update_flips"]]
        by_frame[k] = {
            "closure_progress": schedule[k],
            "jaw_shadow_fraction_range": value_range([r["jaw_shadow_fraction"] for r in rows], 3),
            "lit_fraction_range": value_range([r["lit_fraction"] for r in rows], 3),
            "new_jaw_shadowed_elements_range": value_range(
                [r["new_jaw_shadowed_elements"] for r in rows] if k > PREDICTION_FRAMES[0] else []
            ),
            "shift_max_range": value_range([r["shift_max_jaw_shadow_fraction_3px"] for r in rows], 3),
            "jaw_silhouette_min_distance_px_range": value_range([r["jaw_silhouette_min_distance_px"] for r in rows], 1),
            "offset_range_update_flips_no_shadow_geometries": value_range(flips),
        }
    return {"schedule": schedule, "by_frame": by_frame, "by_geometry": geometries}


def _closure_rows(case):
    return {r["update"]: r for r in case["updates"] if r["update"] in CLOSURE_UPDATES}


def _masked(row):
    value = ((row.get("masked_ncc") or {}).get("dilated") or {}).get("ncc")
    return None if value is None else round(value, 3)


def _within(cases, frames, key, sub):
    return value_range(
        [f[key][sub] for c in cases for f in c["closure_pixel_check"]["frames"] if f["frame"] in frames], 2
    )


def _disagreements(cases):
    """Missed dips and false alarms of the shadow-kept recordings under the main rule."""
    out = {"missed_dips": [], "false_alarms": []}
    for group in SHADOW_KEPT:
        for name in GROUPS[group]:
            rows = {r["update"]: r for r in cases[name]["updates"]}
            table = confusion(cases[name]["updates"], DIPS[0])
            for key, label in (("fn", "missed_dips"), ("fp", "false_alarms")):
                for update in table[key]:
                    r = rows[update]
                    out[label].append(
                        {
                            "case": name,
                            "update": update,
                            "n_jaw_flips": r["n_jaw_flips"],
                            "recorded_corr": round(r["recorded_corr"], 3),
                            "modelled_lit_fraction_before": round(r["prev"]["lit_fraction"], 3),
                            "patch_prev_grey_std": round(r["patch_prev_grey_std"], 1),
                        }
                    )
    return out


def summary(cases, pooled_out, pairs):
    v2 = [cases["v2_eve15004_dl_r1"], cases["v2_eve15004_dl_r2"]]
    v3 = [cases["v3_mor15004_dl_r1"], cases["v3_mor15004_dl_r2"]]
    v2_flips = [[_closure_rows(c)[k]["n_jaw_flips"] for k in CLOSURE_UPDATES] for c in v2]
    v2_corr = [[round(_closure_rows(c)[k]["recorded_corr"], 3) for k in CLOSURE_UPDATES] for c in v2]
    v2_masked = [[_masked(_closure_rows(c)[k]) for k in CLOSURE_UPDATES] for c in v2]
    v3_flips = [[_closure_rows(c)[k]["n_jaw_flips"] for k in CLOSURE_UPDATES] for c in v3]
    v3_corr = [[round(_closure_rows(c)[k]["recorded_corr"], 3) for k in CLOSURE_UPDATES] for c in v3]

    def pair_rows(names, frames):
        return [f for n in names for f in pairs[n]["frames"] if f["frame"] in frames]

    p15 = pair_rows(["pair_eve15004_r1", "pair_eve15004_r2"], CLOSURE_UPDATES)
    p14 = pair_rows(["pair_eve14944_r1", "pair_eve14944_r2"], range(68, 73))
    p14a = pair_rows(["pair_eve14944_r1", "pair_eve14944_r2"], range(57, 67))
    src = [cases["v4_src14944_dl"], cases["v4_src15004_dl"]]
    noshadow = [cases[n] for n in GROUPS["render_controls_jaw_shadow_removed"]]
    v2c = cases["v2c_eve15004_jsb_r1"]
    table = prediction_table(cases)
    t = table["by_frame"]
    nos = [cases[n] for n in PREDICTION_CASES]
    offsets = [
        x["n_lit_to_shadow"] + x["n_shadow_to_lit"]
        for c in nos
        for r in c["updates"]
        if 73 <= r["update"] <= 75
        for x in r["shadow_offset_sensitivity"]
    ]
    flips68 = [next(r["n_shadow_to_lit"] for r in c["updates"] if r["update"] == 68) for c in nos]
    v1_tp_inliers = [
        r["recorded_inliers"]
        for n in GROUPS["V1_evening_14944_approach_shadow_kept"]
        for r in cases[n]["updates"]
        if predicts(r) and dips(r, DIPS[0])
    ]
    disagreements = _disagreements(cases)
    noshadow_dark = [
        f["darkening_T20"]["n_observed"]
        for c in noshadow
        for f in c["closure_pixel_check"]["frames"]
        if f["frame"] <= 76
    ]
    noshadow_dark77 = [
        f["darkening_T20"]["n_observed"]
        for c in noshadow
        for f in c["closure_pixel_check"]["frames"]
        if f["frame"] == 77
    ]
    facts = {
        "evening_15004_closure_73_77": {
            "predicted_jaw_flips_r1_r2": v2_flips,
            "recorded_corr_r1_r2": v2_corr,
            "masked_ncc_dilated_r1_r2": v2_masked,
            "within_run_precision_1px_T20": _within(v2, CLOSURE_UPDATES, "darkening_T20", "precision_1px"),
            "within_run_recall_1px_T20": _within(v2, CLOSURE_UPDATES, "darkening_T20", "recall_1px"),
            "cross_render_iou_T40": value_range([f["iou_T40"] for f in p15]),
            "cross_render_precision_1px_T40": value_range([f["precision_1px_T40"] for f in p15]),
            "cross_render_recall_1px_T40": value_range([f["recall_1px_T40"] for f in p15]),
            "patch_predicted_vs_observed_T20": [
                [
                    round(f["patch_predicted_jaw_shadow_fraction"], 2),
                    round(f["patch_observed_darkening_fraction_T20"], 2),
                ]
                for f in p15
            ],
            "no_shadow_render_corr_73_77": [
                round(r["recorded_corr"], 3) for r in v2c["updates"] if r["update"] in CLOSURE_UPDATES
            ],
        },
        "morning_15004_closure_73_77": {
            "predicted_jaw_flips_r1_r2": v3_flips,
            "recorded_corr_r1_r2": v3_corr,
            "within_run_precision_1px_T20_73_75": _within(v3, range(73, 76), "darkening_T20", "precision_1px"),
            "within_run_recall_1px_T20_73_75": _within(v3, range(73, 76), "darkening_T20", "recall_1px"),
            "within_run_recall_1px_T20_76_77": _within(v3, range(76, 78), "darkening_T20", "recall_1px"),
        },
        "evening_14944_approach_pairs_57_66": {
            "cross_render_iou_T40": value_range([f["iou_T40"] for f in p14a]),
            "cross_render_precision_1px_T40": value_range([f["precision_1px_T40"] for f in p14a]),
            "cross_render_recall_1px_T40": value_range([f["recall_1px_T40"] for f in p14a]),
        },
        "evening_14944_pre_closure_68_72_observed": {
            "patch_predicted_jaw_shadow_fraction": value_range([f["patch_predicted_jaw_shadow_fraction"] for f in p14]),
            "patch_observed_darkening_fraction_T20": value_range(
                [f["patch_observed_darkening_fraction_T20"] for f in p14]
            ),
            "patch_observed_darkening_fraction_T40": value_range(
                [f["patch_observed_darkening_fraction_T40"] for f in p14]
            ),
            "cross_render_iou_T40": value_range([f["iou_T40"] for f in p14]),
        },
        "unexplained_darkening_75_77_px_T20": {
            "source_light_shadow_casting": [
                f["darkening_T20"]["n_observed"]
                for c in src
                for f in c["closure_pixel_check"]["frames"]
                if f["frame"] >= 75
            ],
            "source_light_predicted_lit_to_shadow": [
                f["n_pred_lit_to_shadow"] for c in src for f in c["closure_pixel_check"]["frames"] if f["frame"] >= 75
            ],
            "no_shadow_renders_observed_73_77": {
                c["case"]: [f["darkening_T20"]["n_observed"] for f in c["closure_pixel_check"]["frames"]]
                for c in noshadow
            },
        },
        "evening_14944_no_shadow_geometry_shadow_to_lit_at_update_68": flips68,
        "evening_14944_offset_flips_min_73_75": min(offsets),
        "evening_14944_approach_dip_inliers": value_range(v1_tp_inliers),
        "disagreements": disagreements,
    }
    misses, alarms = disagreements["missed_dips"], disagreements["false_alarms"]
    fe, fpre = facts["evening_15004_closure_73_77"], facts["evening_14944_pre_closure_68_72_observed"]
    dark = facts["unexplained_darkening_75_77_px_T20"]

    def jsf(k):
        return t[k]["jaw_shadow_fraction_range"]

    def nse(k):
        return t[k]["new_jaw_shadowed_elements_range"]

    j_min = min(t[k]["jaw_silhouette_min_distance_px_range"][0] for k in PREDICTION_FRAMES)
    statement = [
        "Prediction: in evening 14944 the closing jaw's cast shadow reaches the tracked 13 x 13 patch during closure. "
        f"In all six geometries the jaw-shadow fraction of the footprint rises from {jsf(72)} at frame 72 to "
        f"{jsf(73)} (73), {jsf(74)} (74) and {jsf(75)} (75), then stays at {jsf(77)}; the lit fraction falls from "
        f"{t[72]['lit_fraction_range']} to {t[75]['lit_fraction_range']} by 75. New jaw-shadowed elements per update: "
        f"{nse(73)} (73), {nse(74)} (74), {nse(75)} (75), {nse(76)} (76), {nse(77)} (77).",
        "Closest validated analogue, evening 15004 closure (same light, same schedule): predicted jaw flips "
        f"{v2_flips} at 73-77 against recorded correlation {v2_corr}; removing the modelled flip pixels restores "
        f"{v2_masked}; within-run pixel precision {fe['within_run_precision_1px_T20']} and recall "
        f"{fe['within_run_recall_1px_T20']} (1 px), cross-render IoU {fe['cross_render_iou_T40']}, recall "
        f"{fe['cross_render_recall_1px_T40']}; the same closure with the jaw shadow removed from the render holds "
        f"{fe['no_shadow_render_corr_73_77']}.",
        "Robustness: the six geometries span about 1 mm of tool pose and agree; at the four committed shadow-ray "
        f"offsets every update 73-75 keeps at least {facts['evening_14944_offset_flips_min_73_75']} flips; the "
        f"most-shadowed footprint within +-3 px reaches {t[75]['shift_max_range']} at 75. The pre-closure state is "
        "observed in the shadowed renders: at frames 68-72 the cross-render check finds "
        f"{fpre['patch_observed_darkening_fraction_T20']} of the patch darkened by more than 20 grey levels against "
        f"{fpre['patch_predicted_jaw_shadow_fraction']} predicted (window IoU {fpre['cross_render_iou_T40']}).",
        f"Validation pooled over the eight shadow-kept recordings ({DECISION_RULES['predicted_jaw_shadow_change']}; "
        f"{DECISION_RULES['observed_dip']}): {pooled_out[f'shadow_kept_dip{DIPS[0]}']}; closure 73-77 only: "
        f"{pooled_out[f'shadow_kept_closure_73_77_dip{DIPS[0]}']}; source-light controls: "
        f"{pooled_out[f'source_controls_dip{DIPS[0]}']}. The {len(misses)} missed dips have at most "
        f"{max((m['n_jaw_flips'] for m in misses), default=0)} predicted jaw flips on a patch whose modelled lit "
        f"fraction before the update is at most {max((m['modelled_lit_fraction_before'] for m in misses), default=0)} "
        f"(grey std at most {max((m['patch_prev_grey_std'] for m in misses), default=0)}): the flip count understates "
        "how sensitive the correlation is once the patch is flat. False alarms: "
        f"{[(a['case'], a['update'], a['n_jaw_flips'], a['recorded_corr']) for a in alarms]}.",
        "Not supported: frames 76-77. The model predicts no further direct-sun change there for 14944 (the patch is "
        f"already fully shadowed), but the morning 15004 dip at 76 ({[c[3] for c in v3_corr]}) was missed with "
        f"{[f[3] for f in v3_flips]} predicted flips, and renders with a shadow-casting jaw darken near contact where "
        f"the model predicts nothing (source light, 75-77: {dark['source_light_shadow_casting']} px observed against "
        f"{dark['source_light_predicted_lit_to_shadow']} predicted; morning 76-77 within-run recall "
        f"{facts['morning_15004_closure_73_77']['within_run_recall_1px_T20_76_77']}), while the renders with the jaw "
        f"shadow removed show {value_range(noshadow_dark)} px at 73-76 and {value_range(noshadow_dark77)} px at 77. "
        "A plausible, untested cause is the jaw occluding sky light near contact. Whether any closure update falls "
        "below the 0.35 gate is not predicted.",
        f"Jaw silhouette (J) clearance of the patch stays at least {j_min} px through frame 77 in all six geometries, "
        "above the 2 px margin. Optical-flow inlier loss during the sweep is not modelled; at the evening 14944 "
        f"approach dips the recorded round-trip inliers were {facts['evening_14944_approach_dip_inliers']}.",
    ]
    limitations = [
        "Direct-sun hard shadow of the two-box visual surrogate only (point sun, pixel-centre rays); sky occlusion, "
        "interreflection, soft edges and denoiser blur are not modelled.",
        "Predicts where and when the patch's sun state changes, not the NCC value; pass/fail at the 0.35 gate and the "
        "task grade are not predicted (tools/validate_vision_sequence.py alone grades runs).",
        "No shadowed 14944 recording reaches closure: the closure geometry is that of the no-shadow runs (about 1 mm "
        "and 0.03 deg from the stopped runs) or the stopped runs' settled pose with the no-shadow closure schedule; a "
        "closed-loop continuation could settle elsewhere within that range.",
        "The footprint in a continuation is assumed to stay on the same bark point (no tracker drift); optical-flow "
        "inlier loss and the depth test are not modelled.",
        "Geometry is re-extracted from each run's own scene.usda; robot link vertices are expressed in link frames "
        "through a matrix inverse and may differ in the last bits between extraction hosts (their content hash is "
        "reported); the reproduction section checks the committed evidence's counts.",
        "Thresholds of the decision rule were fixed after the recorded correlations had been seen; variants are "
        "reported. One render per run (simulator, path traced).",
    ]
    not_done = [
        "Magnitude: synthesize the jaw shadow onto the no-shadow 14944 closure frames and compute the tracker NCC, "
        "validated first on evening and morning 15004 closure where the shadowed NCC is recorded.",
        "A jaw sky-occlusion term, tested on the source-light closure frames where it is the only jaw lighting effect.",
    ]
    return {
        "facts": facts,
        "prediction_table": table,
        "reliability_statement": statement,
        "limitations": limitations,
        "not_done": not_done,
    }


def finite(value):
    return dal.finite(value)


def build_result(cases, pairs, inputs):
    pooled_out = pooled_confusion(cases)
    pair_out = pair_summary(pairs)
    tool_paths = {
        "tools/project_closure_shadow.py": Path(__file__),
        "tools/diagnose_appearance_loss.py": TOOLS / "diagnose_appearance_loss.py",
        "tools/extract_shadow_casters.py": TOOLS / "extract_shadow_casters.py",
        "jaw_self_mask.py": Path(jsm.__file__),
        "visual_servo.py": Path(sys.modules[dal.VisualServoTracker.__module__].__file__),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "code_revision": git("rev-parse", "HEAD"),
        "code_tree_dirty": bool(git("status", "--porcelain", "--untracked-files=no")),
        "code_sha256": {name: sha256(path) for name, path in tool_paths.items()},
        "inputs_sha256": dict(sorted(inputs.items())),
        "decision_rules": DECISION_RULES,
        "summary": summary(cases, pooled_out, pair_out),
        "reproduction_of_committed_evidence": reproduction(cases),
        "geometry_consistency": geometry_consistency({**cases, **pairs}),
        "validation": validation(cases),
        "pooled_confusion": pooled_out,
        "render_pair_pixel_check": pair_out,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path, help="New JSON file outside the repository")
    parser.add_argument("--runs-root", type=Path, default=RUNS_DEFAULT, help="Directory of the recorded batches")
    parser.add_argument(
        "--isaac-python",
        default=os.environ.get("PRUNING_ISAAC_PYTHON"),
        help="Python with pxr that runs tools/extract_shadow_casters.extract() in memory (env PRUNING_ISAAC_PYTHON)",
    )
    parser.add_argument("--cache-root", type=Path, help="extract_shadow_casters caches, outside the repository")
    args = parser.parse_args(argv)
    try:
        output = refuse_inside(args.output, "an output")
        source = GeometrySource(args.isaac_python, args.cache_root)
    except ValueError as exc:
        parser.error(str(exc))
    if output.exists():
        parser.error(f"Refusing to overwrite {output}")
    if not args.isaac_python and args.cache_root is None:
        parser.error("Give --isaac-python (in-memory extraction) or --cache-root")
    inputs = {relative(EVIDENCE): sha256(EVIDENCE)}
    cases, pairs = {}, {}
    try:
        for name in CASES:
            cases[name] = run_case(name, args.runs_root, source, inputs)
            print(f"{name}: {len(cases[name]['updates'])} updates", flush=True)
        for name in PAIRS:
            pairs[name] = pair_check(name, args.runs_root, source, inputs)
            print(f"{name}: {len(pairs[name]['frames'])} frames", flush=True)
    except (GateError, dal.GateError) as exc:
        parser.exit(1, f"gate failed, nothing written: {exc}\n")
    result = build_result(cases, pairs, inputs)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        json.dump(finite(result), stream, indent=1, allow_nan=False)
        stream.write("\n")
    for line in result["summary"]["reliability_statement"][:1]:
        print(line)
    print(f"pooled: {result['pooled_confusion'][f'shadow_kept_dip{DIPS[0]}']}; wrote {output}")


if __name__ == "__main__":
    main()
