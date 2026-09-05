#!/usr/bin/env python3
"""Render component demos from measured contracts. No Isaac Sim required.

Evidence-backed animations validate their input before opening an output file.
In particular, the live-ToF animation cannot be produced from a failed or
partial smoke report.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any

# Headless login nodes do not necessarily have a writable Matplotlib cache.
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / f"pruning-mpl-{os.getuid()}"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import animation
from mpl_toolkits.mplot3d.art3d import Line3DCollection

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "source" / "isaaclab_pruning"))
OUT = ROOT / "docs" / "demo"
EVIDENCE = ROOT / "docs" / "evidence"
IMPORT_FAILURE_EVIDENCE = EVIDENCE / "urdf_import_21125352.json"
IMPORT_SUCCESS_EVIDENCE = EVIDENCE / "urdf_import_21136450.json"
IMPORT_SUCCESS_ASSET_ID = "ur5e_mock_pruner_bdsdfede4c0_ur18e6f603_calib_3941312424972580002_urdf6b02ce9330be"
IMPORT_SUCCESS_URDF_SHA256 = "a5c04da197c7de2588f1716bb7b25fee47b500ee98856bdc4d640fea90218f44"
IMPORT_SUCCESS_ROOT_SHA256 = "6ffa65568f8585f7ea5938c4f49263a8b12fd7dc03e18e6872d75761efa2da82"

_EXPECTED_UR_JOINTS = (
    "ur5e__shoulder_pan_joint",
    "ur5e__shoulder_lift_joint",
    "ur5e__elbow_joint",
    "ur5e__wrist_1_joint",
    "ur5e__wrist_2_joint",
    "ur5e__wrist_3_joint",
)
_EXPECTED_PROVENANCE_FRAMES = {
    "mock_pruner__camera0": (-0.0017977, -0.0715747, 0.0711646),
    "mock_pruner__tof0": (0.04685226669, 0.0, 0.14444246761),
    "mock_pruner__tof1": (-0.04685226669, 0.0, 0.14444246761),
    "mock_pruner__tool0": (0.0, 0.0, 0.1601525),
}

from isaaclab_pruning.geometry import Cylinder, cylinder_endpoints  # noqa: E402
from isaaclab_pruning.policies.observations import (  # noqa: E402
    ObservationVariant,
    fuse_tof_and_metric,
    observation_width,
)
from isaaclab_pruning.robot import load_ur5e_pruner_spec  # noqa: E402


class EvidenceContractError(ValueError):
    """Raised before rendering when an evidence report is not a proven result."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceContractError(message)


def _read_evidence(path: Path) -> dict[str, Any]:
    _require(path.is_file(), f"evidence file does not exist: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise EvidenceContractError(f"could not read evidence JSON {path}: {error}") from error
    _require(isinstance(value, dict), f"evidence root must be an object: {path}")
    return value


def _save_gif_atomic(anim: animation.Animation, output_path: Path, *, dpi: int = 100) -> Path:
    """Save an animation without exposing a partial or failed output."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.stem}.", suffix=".gif", dir=output_path.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        anim.save(temporary_path, writer=animation.PillowWriter(fps=2), dpi=dpi)
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def _symbolic_asset_path(path: str, asset_id: str) -> str:
    """Keep the meaningful path shape visible without a 100-character hash ID."""
    return path.replace(asset_id, "{asset}")


def _validate_import_gate_failure(evidence_path: Path) -> dict[str, Any]:
    report = _read_evidence(evidence_path)
    match = re.fullmatch(r"urdf_import_(\d+)\.json", evidence_path.name)
    _require(match is not None, "import evidence filename must be urdf_import_<jobid>.json")
    job_id = match.group(1)
    _require(job_id == "21125352", f"expected the reviewed failure job 21125352, got {job_id}")
    _require(report.get("schema_version") == 1, "import evidence schema_version must be 1")
    _require(report.get("status") == "failed", "import evidence status must be failed")
    _require(report.get("ok") is False, "failed import evidence must have ok=false")
    _require(report.get("imported") is False, "failed import evidence must have imported=false")
    _require(report.get("stage_validation") is None, "failed path-contract import must not claim stage validation")

    asset_id = report.get("asset_id")
    _require(isinstance(asset_id, str) and bool(asset_id), "import evidence requires asset_id")
    node = report.get("node", {}).get("hostname")
    _require(isinstance(node, str) and bool(node), "import evidence requires node.hostname")
    expected_path = report.get("output", {}).get("root_layer")
    _require(isinstance(expected_path, str) and bool(expected_path), "import evidence requires output.root_layer")

    files = report.get("output", {}).get("files")
    _require(isinstance(files, list), "import evidence requires output.files")
    root_candidates = []
    for entry in files:
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            continue
        candidate = Path(entry["path"])
        if candidate.suffix in {".usd", ".usda"} and candidate.parent.name == candidate.stem:
            root_candidates.append(entry["path"])
    _require(len(root_candidates) == 1, "expected exactly one nested converter root in output inventory")
    actual_path = root_candidates[0]
    _require(actual_path != expected_path, "failure evidence does not contain a root-layer mismatch")

    reason = report.get("reason")
    _require(isinstance(reason, str), "failed import evidence requires a reason")
    reason_match = re.fullmatch(r"RuntimeError: converter returned unexpected USD path: (.+) != (.+)", reason)
    _require(reason_match is not None, "failure reason is not the reviewed converter path mismatch")
    _require(reason_match.group(1).endswith(actual_path), "reason's actual path disagrees with output inventory")
    _require(reason_match.group(2).endswith(expected_path), "reason's expected path disagrees with output.root_layer")

    configured_name = report.get("converter_config", {}).get("usd_file_name")
    _require(
        isinstance(configured_name, str) and Path(expected_path).name == configured_name,
        "configured USD filename disagrees with expected root layer",
    )
    return {
        "job_id": job_id,
        "node": node,
        "asset_id": asset_id,
        "expected_path": expected_path,
        "actual_path": actual_path,
    }


def _validate_import_gate_success(evidence_path: Path) -> dict[str, Any]:  # noqa: C901, PLR0915
    """Validate the exact reviewed import-success evidence before rendering."""
    report = _read_evidence(evidence_path)
    _require(
        evidence_path.name == "urdf_import_21136450.json",
        "expected the reviewed success evidence urdf_import_21136450.json",
    )
    _require(report.get("schema_version") == 1, "import evidence schema_version must be 1")
    _require(report.get("status") == "complete", "success import evidence status must be complete")
    _require(report.get("ok") is True, "success import evidence must have ok=true")
    _require(report.get("imported") is True, "success import evidence must have imported=true")
    _require(
        report.get("asset_id") == IMPORT_SUCCESS_ASSET_ID,
        f"expected the reviewed content-addressed asset {IMPORT_SUCCESS_ASSET_ID}",
    )

    job = report.get("job")
    _require(isinstance(job, dict), "success import evidence requires job metadata")
    _require(job.get("id") == "21136450", "expected the reviewed success job 21136450")
    _require(job.get("name") == "prune-urdf-import", "success evidence has the wrong job name")
    node = report.get("node", {}).get("hostname")
    _require(isinstance(node, str) and bool(node), "success import evidence requires node.hostname")

    asset_id = IMPORT_SUCCESS_ASSET_ID
    root_name = f"{asset_id}_abs"
    expected_root = f"artifacts/usd/{asset_id}/{root_name}/{root_name}.usda"
    expected_urdf = f"artifacts/urdf/{asset_id}/{root_name}.urdf"

    input_fact = report.get("input")
    _require(isinstance(input_fact, dict), "success import evidence requires input metadata")
    _require(input_fact.get("urdf") == expected_urdf, "input URDF path is not the reviewed asset path")
    _require(
        input_fact.get("absolute_urdf_sha256_verified") is True,
        "input absolute URDF hash was not verified",
    )
    _require(
        input_fact.get("provenance_asset_id_verified") is True,
        "generation provenance asset ID was not verified",
    )
    _require(
        input_fact.get("urdf_sha256") == IMPORT_SUCCESS_URDF_SHA256,
        "input URDF SHA-256 is not the reviewed hash",
    )
    provenance = input_fact.get("provenance")
    _require(
        provenance == f"docs/evidence/urdf_generation_{asset_id}.json",
        "input provenance path is not the matching generation evidence",
    )
    _require(
        re.fullmatch(r"[0-9a-f]{64}", str(input_fact.get("provenance_sha256"))) is not None,
        "generation provenance SHA-256 is missing or malformed",
    )

    converter = report.get("converter_config")
    _require(isinstance(converter, dict), "success import evidence requires converter_config")
    _require(
        converter.get("usd_file_name") == f"{root_name}/{root_name}.usda",
        "converter nested root filename disagrees with the reviewed output contract",
    )

    output = report.get("output")
    _require(isinstance(output, dict), "success import evidence requires output metadata")
    _require(output.get("directory") == f"artifacts/usd/{asset_id}", "output directory is not asset-addressed")
    root_layer = output.get("root_layer")
    _require(root_layer == expected_root, "output root layer is not the reviewed nested root")
    files = output.get("files")
    _require(isinstance(files, list) and bool(files), "success import evidence requires an output inventory")
    paths: list[str] = []
    root_entries: list[dict[str, Any]] = []
    for entry in files:
        _require(isinstance(entry, dict), "each output inventory entry must be an object")
        path = entry.get("path")
        sha256 = entry.get("sha256")
        byte_count = entry.get("bytes")
        _require(isinstance(path, str) and bool(path), "each output inventory entry requires a path")
        _require(
            isinstance(byte_count, int) and byte_count >= 0,
            f"output inventory bytes are invalid for {path}",
        )
        _require(
            re.fullmatch(r"[0-9a-f]{64}", str(sha256)) is not None,
            f"output inventory SHA-256 is missing or malformed for {path}",
        )
        paths.append(path)
        if path == root_layer:
            root_entries.append(entry)
    _require(len(paths) == len(set(paths)), "output inventory contains duplicate paths")
    _require(len(root_entries) == 1, "output inventory must contain the root layer exactly once")
    _require(
        root_entries[0].get("sha256") == IMPORT_SUCCESS_ROOT_SHA256,
        "root-layer SHA-256 is not the reviewed hash",
    )

    stage = report.get("stage_validation")
    _require(isinstance(stage, dict), "success import evidence requires stage_validation")
    _require(stage.get("ok") is True, "USD stage validation must have ok=true")
    _require(stage.get("meters_per_unit") == 1.0, "USD stage must use meters")
    _require(stage.get("up_axis") == "Z", "USD stage must use the Z up-axis")
    _require(stage.get("linear_slider_prim_paths") == [], "hardware slider must be absent")
    expected_joints = [
        {
            "name": name,
            "path": f"/pruning_robot/Physics/{name}",
            "type": "PhysicsRevoluteJoint",
        }
        for name in _EXPECTED_UR_JOINTS
    ]
    _require(stage.get("active_ur_joint_count") == 6, "USD stage must contain six active UR joints")
    _require(
        stage.get("active_ur_joints") == expected_joints,
        "USD stage does not contain the six exact reviewed UR revolute joints",
    )

    frames = stage.get("frames")
    _require(isinstance(frames, dict), "USD stage validation requires frame metadata")
    for name, expected_translation in _EXPECTED_PROVENANCE_FRAMES.items():
        frame = frames.get(name)
        _require(isinstance(frame, dict), f"USD stage validation is missing {name}")
        _require(frame.get("type") == "Xform", f"{name} must be an Xform")
        _require(frame.get("path", "").endswith(f"/{name}"), f"{name} has the wrong stage path")
        _require(
            frame.get("provenance_transform_verified") is True,
            f"{name} transform was not provenance-verified",
        )
        relative = frame.get("relative_to_mock_pruner_base")
        _require(isinstance(relative, dict), f"{name} is missing its relative transform")
        translation = np.asarray(relative.get("translation_m"), dtype=float)
        rotation = np.asarray(relative.get("rotation_matrix"), dtype=float)
        matrix = np.asarray(relative.get("matrix"), dtype=float)
        _require(
            translation.shape == (3,)
            and bool(np.isfinite(translation).all())
            and bool(np.allclose(translation, expected_translation, rtol=0.0, atol=1.0e-12)),
            f"{name} translation is not the reviewed hardware transform",
        )
        _require(
            rotation.shape == (3, 3)
            and bool(np.isfinite(rotation).all())
            and bool(np.allclose(rotation, np.eye(3), rtol=0.0, atol=1.0e-12)),
            f"{name} rotation is not the reviewed hardware transform",
        )
        _require(matrix.shape == (4, 4) and bool(np.isfinite(matrix).all()), f"{name} matrix is invalid")

    stack = report.get("stack")
    _require(isinstance(stack, dict), "success import evidence requires stack metadata")
    _require(stack.get("isaacsim_version") == "6.0.0.1", "success evidence is not from Isaac Sim 6.0.0.1")
    _require(stack.get("isaaclab_version") == "3.0.0b2", "success evidence is not from Isaac Lab 3.0.0b2")
    return {
        "job_id": job["id"],
        "node": node,
        "asset_id": asset_id,
        "root_layer": root_layer,
        "root_sha256": root_entries[0]["sha256"],
        "urdf_sha256": input_fact["urdf_sha256"],
        "file_count": len(files),
        "joint_names": list(_EXPECTED_UR_JOINTS),
        "frame_names": list(_EXPECTED_PROVENANCE_FRAMES),
        "isaacsim_version": stack["isaacsim_version"],
        "isaaclab_version": stack["isaaclab_version"],
    }


def _numeric_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    _require(array.ndim == 1 and array.size > 0, f"{name} must be a non-empty numeric vector")
    _require(bool(np.isfinite(array).all()), f"{name} must contain only finite values")
    return array


def _validate_live_tof_success(evidence_path: Path) -> dict[str, Any]:  # noqa: C901
    report = _read_evidence(evidence_path)
    _require(report.get("ok") is True, "live-ToF GIF requires a passing smoke report (ok=true)")
    job_id = report.get("job_id")
    _require(str(job_id).isdigit(), "passing smoke evidence requires a numeric job_id")
    _require(
        evidence_path.name == f"smoke_{job_id}.json",
        "smoke evidence filename and embedded job_id must match",
    )
    for key in ("node", "asset_id", "usd", "usd_evidence"):
        _require(isinstance(report.get(key), str) and bool(report[key]), f"passing smoke evidence requires {key}")

    before_state = report.get("tof_before")
    after_state = report.get("tof_after")
    _require(isinstance(before_state, dict) and isinstance(after_state, dict), "missing before/after ToF state")
    for label, state in (("before", before_state), ("after", after_state)):
        _require(state.get("source") == "live_multi_mesh_ray_caster", f"{label} ToF source is not live")
        _require(state.get("noise_enabled") is False, f"{label} ToF state must be deterministic")
        limits = np.asarray(state.get("range_limits_m"), dtype=float)
        _require(
            limits.shape == (2,) and bool(np.isfinite(limits).all()) and 0.0 < limits[0] < limits[1],
            f"{label} ToF range limits are invalid",
        )

    before_raw = report.get("tof_raw_frames", {}).get("before")
    after_raw = report.get("tof_raw_frames", {}).get("after")
    changes = report.get("tof_range_change")
    _require(isinstance(before_raw, dict) and isinstance(after_raw, dict), "missing raw before/after ToF frames")
    _require(isinstance(changes, dict), "missing ToF range-change evidence")

    normalized_before: dict[str, np.ndarray] = {}
    normalized_after: dict[str, np.ndarray] = {}
    normalized_delta: dict[str, np.ndarray] = {}
    for name in ("tof0", "tof1"):
        for label, state in (("before", before_state), ("after", after_state)):
            sensor = state.get("sensors", {}).get(name)
            _require(isinstance(sensor, dict), f"missing {label} state for {name}")
            _require(sensor.get("class") == "MultiMeshRayCasterCamera", f"{name} is not the live ray-caster class")
            observation = sensor.get("observation", {})
            _require(observation.get("shape") == [1, 8, 8], f"{label} {name} observation shape is not [1,8,8]")
            valid_fraction = observation.get("valid_fraction")
            _require(
                isinstance(valid_fraction, (int, float)) and math.isfinite(valid_fraction) and valid_fraction > 0.0,
                f"{label} {name} has no valid observation pixels",
            )

        before_frame = _numeric_vector(before_state["sensors"][name].get("frame"), f"before {name} frame")
        after_frame = _numeric_vector(after_state["sensors"][name].get("frame"), f"after {name} frame")
        _require(
            before_frame.shape == after_frame.shape and float(after_frame.min()) > float(before_frame.min()),
            f"{name} frame counter did not advance",
        )

        before = np.asarray(before_raw.get(name), dtype=float)
        after = np.asarray(after_raw.get(name), dtype=float)
        _require(before.shape == (1, 8, 8), f"before {name} raw frame shape is not [1,8,8]")
        _require(after.shape == (1, 8, 8), f"after {name} raw frame shape is not [1,8,8]")
        before, after = before[0], after[0]
        shared = np.isfinite(before) & np.isfinite(after)
        _require(bool(shared.any()), f"{name} has no shared finite pixels")
        delta = np.full_like(before, np.nan)
        delta[shared] = np.abs(after[shared] - before[shared])
        measured_max = float(np.nanmax(delta))
        recorded = changes.get(name)
        _require(isinstance(recorded, dict), f"missing range-change summary for {name}")
        recorded_max = recorded.get("max_abs_delta_m")
        _require(
            isinstance(recorded_max, (int, float))
            and math.isfinite(recorded_max)
            and math.isclose(measured_max, float(recorded_max), rel_tol=1.0e-5, abs_tol=1.0e-8),
            f"{name} raw frames disagree with recorded max range change",
        )
        _require(measured_max > 1.0e-4, f"{name} does not prove geometry-responsive range data")
        normalized_before[name] = before
        normalized_after[name] = after
        normalized_delta[name] = delta

    spaces = report.get("observation_space")
    widths = report.get("obs_last_dim")
    _require(isinstance(spaces, dict) and isinstance(widths, dict), "missing observation-width contracts")
    _require(spaces == widths, "configured and emitted observation widths differ")
    _require(spaces.get("A_flow") != spaces.get("B_tof") != spaces.get("C_metric"), "A/B/C widths are not distinct")
    _require(spaces.get("A_flow") != spaces.get("C_metric"), "A/C widths are not distinct")
    _require(spaces.get("C_metric") == spaces.get("D_fused"), "C/D widths must match")

    tool = report.get("tool_frame")
    _require(isinstance(tool, dict), "missing tool-frame evidence")
    separation = _numeric_vector(tool.get("body_to_tool_distance_m"), "body-to-tool distance")
    _require(bool(np.allclose(separation, 0.1601525, rtol=0.0, atol=1.0e-5)), "body-to-tool transform drifted")
    hold_drift = _numeric_vector(tool.get("hold_translation_drift_m"), "hold translation drift")
    hold_rotation = _numeric_vector(tool.get("hold_rotation_drift_rad"), "hold rotation drift")
    move_initial = _numeric_vector(tool.get("move_initial_error_m"), "move initial error")
    move_final = _numeric_vector(tool.get("move_final_error_m"), "move final error")
    _require(bool((hold_drift < 5.0e-3).all()), "tool did not hold within 5 mm")
    _require(bool((hold_rotation < 2.0e-2).all()), "tool rotation did not hold within 0.02 rad")
    _require(bool(((move_initial > 4.0e-3) & (move_initial < 6.0e-3)).all()), "motion command was not 5 mm")
    _require(bool((move_final < move_initial).all()), "tool did not move toward its command")
    _require(report.get("contact", {}).get("finite") is True, "contact evidence is not finite")
    _require(report.get("n_arm_joints") == 6, "passing smoke must contain six arm joints")
    _require(report.get("slider_in_joint_names") is False, "hardware slider must be absent")

    return {
        "job_id": str(job_id),
        "node": report["node"],
        "asset_id": report["asset_id"],
        "before": normalized_before,
        "after": normalized_after,
        "delta": normalized_delta,
        "before_frames": {name: int(min(before_state["sensors"][name]["frame"])) for name in ("tof0", "tof1")},
        "after_frames": {name: int(min(after_state["sensors"][name]["frame"])) for name in ("tof0", "tof1")},
        "move_command_mm": float(move_initial.mean() * 1_000.0),
        "move_final_error_mm": float(move_final.mean() * 1_000.0),
        "hold_drift_mm": float(hold_drift.max() * 1_000.0),
    }


def _sample_tree() -> list[Cylinder]:
    return [
        Cylinder("t", "trunk_1", np.array([0.0, 0.0, 0.6]), np.array([0.0, 0.0, 1.0]), 0.04, 1.2),
        Cylinder("b", "branch_1", np.array([0.15, 0.0, 1.0]), np.array([1.0, 0.0, 0.2]), 0.012, 0.4),
        Cylinder("s", "spur_1", np.array([0.32, 0.02, 1.05]), np.array([0.2, 1.0, 0.0]), 0.005, 0.12),
    ]


def _cylinder_lines(cylinders: list[Cylinder]) -> Line3DCollection:
    segs, colors, widths = [], [], []
    palette = {"trunk": "#8d6e63", "branch": "#43a047", "spur": "#fb8c00"}
    for cyl in cylinders:
        a, b = cylinder_endpoints(cyl)
        segs.append([a, b])
        colors.append(palette.get(cyl.organ_class, "#90a4ae"))
        widths.append(max(1.5, cyl.radius * 80))
    return Line3DCollection(segs, colors=colors, linewidths=widths)


def write_tree() -> Path:
    fig = plt.figure(figsize=(6, 6), facecolor="#0e1116")
    ax = fig.add_subplot(111, projection="3d", facecolor="#0e1116")
    ax.add_collection3d(_cylinder_lines(_sample_tree()))
    ax.set_xlim(-0.2, 0.5)
    ax.set_ylim(-0.2, 0.4)
    ax.set_zlim(0.0, 1.3)
    ax.set_title("UsdGeom.Cylinder tree  ·  not capsules", color="white", fontsize=11)
    ax.tick_params(colors="#90a4ae")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    path = OUT / "tree_cylinders.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_widths() -> Path:
    labels = ["A flow", "B ToF×2", "C metric", "D fused"]
    values = [
        observation_width(ObservationVariant.FLOW),
        observation_width(ObservationVariant.TOF),
        observation_width(ObservationVariant.METRIC),
        observation_width(ObservationVariant.FUSED),
    ]
    fig, ax = plt.subplots(figsize=(7, 3.4), facecolor="#0e1116")
    ax.set_facecolor("#0e1116")
    bars = ax.bar(labels, values, color=["#7e57c2", "#29b6f6", "#66bb6a", "#ffa726"])
    ax.axhline(128, color="#ef5350", ls="--", lw=1, label="BHL trap (128)")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 6, str(value), ha="center", color="white", fontsize=10)
    ax.set_ylabel("observation last-dim", color="white")
    ax.set_title("A≠B≠C  ·  C=D width  ·  never 128", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0e1116", labelcolor="white", frameon=False)
    for spine in ax.spines.values():
        spine.set_color("#455a64")
    path = OUT / "obs_widths.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_fusion_gif() -> Path:
    rng = np.random.default_rng(0)
    tof0 = torch.tensor(0.40 + 0.04 * rng.standard_normal((8, 8)), dtype=torch.float32)
    tof1 = torch.tensor(0.42 + 0.04 * rng.standard_normal((8, 8)), dtype=torch.float32)
    metric = torch.tensor(1.20 + 0.08 * rng.standard_normal((16, 16)), dtype=torch.float32)
    frames = []
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.8), facecolor="#0e1116")
    titles = ["ToF0 8×8", "metric 16×16 → 8×8", "D = fuse(ToF0,ToF1,metric)"]
    for ax, title in zip(axes, titles, strict=True):
        ax.set_facecolor("#0e1116")
        ax.set_title(title, color="white", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    ims = [
        axes[0].imshow(tof0.numpy(), vmin=0.2, vmax=1.4, cmap="magma"),
        axes[1].imshow(metric.numpy(), vmin=0.2, vmax=1.4, cmap="magma"),
        axes[2].imshow(np.zeros((8, 8)), vmin=0.2, vmax=1.4, cmap="magma"),
    ]

    def update(frame: int):
        noise = 0.02 * np.sin(frame / 3.0)
        t0 = tof0 + noise
        t1 = tof1 - 0.5 * noise
        fused = fuse_tof_and_metric(
            t0.unsqueeze(0),
            t1.unsqueeze(0),
            metric.unsqueeze(0),
            torch.full_like(t0, 1e-4).unsqueeze(0),
            torch.full_like(t1, 1e-4).unsqueeze(0),
            torch.full_like(metric, 1e-2).unsqueeze(0),
            torch.ones_like(t0, dtype=torch.bool).unsqueeze(0),
            torch.ones_like(t1, dtype=torch.bool).unsqueeze(0),
        )[0]
        ims[0].set_data(t0.numpy())
        ims[2].set_data(fused.numpy())
        return ims

    anim = animation.FuncAnimation(fig, update, frames=24, interval=80, blit=True)
    path = OUT / "fusion_d.gif"
    anim.save(path, writer="pillow", dpi=110)
    plt.close(fig)
    frames.append(path)
    return path


def write_gate0() -> Path:
    evidence = json.loads((ROOT / "docs" / "evidence" / "isaac_smoke_21077170.json").read_text())
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor="#0e1116")
    ax.set_facecolor("#0e1116")
    ax.axis("off")
    ax.set_title("Gate 0  ·  Isaac Sim 6.0 RTX  ·  job 21077170", color="white", pad=12)
    lines = [
        f"node  {evidence['node']}   A40",
        f"cube planar z   {evidence['cube_depth_m']:.4f} m   expect 1.5000",
        f"plane planar z  {evidence['plane_depth_m']:.4f} m   expect 2.0000",
        f"RGB std         {evidence['cube_rgb_std']:.1f}     not a black frame",
        f"trees           {evidence['trees']['count']} Envy debug USDA",
    ]
    for index, line in enumerate(lines):
        ax.text(
            0.04,
            0.78 - 0.16 * index,
            line,
            color="#eceff1",
            fontsize=11,
            family="monospace",
            transform=ax.transAxes,
        )
    path = OUT / "gate0_rtx.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_cutter() -> Path:
    spec = load_ur5e_pruner_spec()
    fig, ax = plt.subplots(figsize=(5.5, 3.4), facecolor="#0e1116")
    ax.set_facecolor("#0e1116")

    def rect(half, offset, color, label):
        x, y = offset[1] - half[1], offset[2] - half[2]
        ax.add_patch(plt.Rectangle((x, y), 2 * half[1], 2 * half[2], fill=False, ec=color, lw=2, label=label))

    rect(spec.mouth_half_extents_m, spec.mouth_offset_m, "#ff7043", "mouth AABB")
    rect(spec.failure_half_extents_m, spec.failure_offset_m, "#ef5350", "failure AABB")
    ax.set_aspect("equal")
    ax.set_xlabel("EEF Y (m)", color="white")
    ax.set_ylabel("EEF Z (m)", color="white")
    ax.set_title("Cutter boxes from pybullet-tree-sim STL", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#0e1116", labelcolor="white", frameon=False)
    path = OUT / "cutter_boxes.png"
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return path


def write_import_gate_failure(
    evidence_path: Path = IMPORT_FAILURE_EVIDENCE,
    output_path: Path = OUT / "import_gate_failure.gif",
) -> Path:
    """Render the reviewed 21125352 application-gate rejection."""
    evidence_path = Path(evidence_path)
    output_path = Path(output_path)
    fact = _validate_import_gate_failure(evidence_path)
    expected = _symbolic_asset_path(fact["expected_path"], fact["asset_id"])
    actual = _symbolic_asset_path(fact["actual_path"], fact["asset_id"])

    fig, ax = plt.subplots(figsize=(8.2, 3.4), facecolor="#0e1116")
    ax.set_facecolor("#0e1116")
    ax.axis("off")
    title = ax.text(
        0.04,
        0.90,
        f"URDF import gate  ·  job {fact['job_id']}",
        color="white",
        fontsize=14,
        weight="bold",
        transform=ax.transAxes,
    )
    stage = ax.text(0.04, 0.71, "", fontsize=12, weight="bold", transform=ax.transAxes)
    detail = ax.text(
        0.04,
        0.52,
        "",
        color="#eceff1",
        fontsize=10,
        family="monospace",
        va="top",
        linespacing=1.5,
        transform=ax.transAxes,
    )
    footer = ax.text(
        0.04,
        0.08,
        f"Evidence: {evidence_path.name}  ·  {fact['node']}",
        color="#90a4ae",
        fontsize=8.5,
        transform=ax.transAxes,
    )
    dots = [ax.scatter(0.73 + index * 0.075, 0.91, s=75, color="#455a64", transform=ax.transAxes) for index in range(3)]
    states = [
        (
            "1  CONVERTER WROTE A NESTED ROOT",
            "#ffb74d",
            f"actual\n  {actual}\n\n11 inventoried files; structural validation not run",
        ),
        (
            "2  OUTPUT CONTRACT DID NOT MATCH",
            "#ef5350",
            f"configured / expected\n  {expected}\n\nactual\n  {actual}",
        ),
        (
            "3  APPLICATION GATE REJECTED IT",
            "#66bb6a",
            "status = failed    ok = false    imported = false\n\n"
            "The failed output was not accepted as runtime evidence.",
        ),
    ]

    def update(frame: int):
        index = frame // 2
        label, color, lines = states[index]
        stage.set_text(label)
        stage.set_color(color)
        detail.set_text(lines)
        for dot_index, dot in enumerate(dots):
            dot.set_color(color if dot_index <= index else "#455a64")
        return [title, stage, detail, footer, *dots]

    anim = animation.FuncAnimation(fig, update, frames=6, interval=500, blit=True)
    try:
        return _save_gif_atomic(anim, output_path)
    finally:
        plt.close(fig)


def write_import_gate_success(
    evidence_path: Path = IMPORT_SUCCESS_EVIDENCE,
    output_path: Path = OUT / "import_gate_success.gif",
) -> Path:
    """Render the reviewed 21136450 application-gate acceptance."""
    evidence_path = Path(evidence_path)
    output_path = Path(output_path)
    fact = _validate_import_gate_success(evidence_path)
    root_layer = _symbolic_asset_path(fact["root_layer"], fact["asset_id"])
    joint_lines = "\n".join(
        (
            "  shoulder_pan · shoulder_lift · elbow",
            "  wrist_1 · wrist_2 · wrist_3",
        )
    )

    fig, ax = plt.subplots(figsize=(8.2, 3.4), facecolor="#0e1116")
    ax.set_facecolor("#0e1116")
    ax.axis("off")
    title = ax.text(
        0.04,
        0.90,
        f"URDF import gate  ·  job {fact['job_id']}  ·  PASS",
        color="white",
        fontsize=14,
        weight="bold",
        transform=ax.transAxes,
    )
    stage = ax.text(0.04, 0.71, "", fontsize=12, weight="bold", transform=ax.transAxes)
    detail = ax.text(
        0.04,
        0.52,
        "",
        color="#eceff1",
        fontsize=10,
        family="monospace",
        va="top",
        linespacing=1.5,
        transform=ax.transAxes,
    )
    footer = ax.text(
        0.04,
        0.08,
        (
            f"Evidence: {evidence_path.name}  ·  {fact['node']}  ·  "
            f"Isaac Sim {fact['isaacsim_version']} / Isaac Lab {fact['isaaclab_version']}"
        ),
        color="#90a4ae",
        fontsize=8.2,
        transform=ax.transAxes,
    )
    dots = [ax.scatter(0.74 + index * 0.055, 0.91, s=70, color="#455a64", transform=ax.transAxes) for index in range(4)]
    states = [
        (
            "1  NESTED ROOT CONTRACT RESOLVED",
            "#42a5f5",
            f"root\n  {root_layer}\n\n{fact['file_count']} files recorded in the output inventory",
        ),
        (
            "2  ROBOT STRUCTURE VALIDATED",
            "#66bb6a",
            f"6 active PhysicsRevoluteJoint prims\n{joint_lines}\n\nlinear slider prims = 0",
        ),
        (
            "3  HARDWARE FRAME PROVENANCE VERIFIED",
            "#66bb6a",
            "camera0  ·  tof0  ·  tof1  ·  tool0\n\nAll four relative transforms match the reviewed source assembly.",
        ),
        (
            "4  APPLICATION GATE ACCEPTED IT",
            "#66bb6a",
            "status = complete    ok = true    imported = true\n\n"
            f"URDF sha256  {fact['urdf_sha256'][:16]}…\n"
            f"root sha256  {fact['root_sha256'][:16]}…",
        ),
    ]

    def update(frame: int):
        index = frame // 2
        label, color, lines = states[index]
        stage.set_text(label)
        stage.set_color(color)
        detail.set_text(lines)
        for dot_index, dot in enumerate(dots):
            dot.set_color(color if dot_index <= index else "#455a64")
        return [title, stage, detail, footer, *dots]

    anim = animation.FuncAnimation(fig, update, frames=8, interval=500, blit=True)
    try:
        return _save_gif_atomic(anim, output_path)
    finally:
        plt.close(fig)


def write_live_tof_success(
    evidence_path: Path,
    output_path: Path = OUT / "live_dual_tof_success.gif",
) -> Path:
    """Render measured dual-ToF frames from a passing geometry smoke only."""
    evidence_path = Path(evidence_path)
    output_path = Path(output_path)
    fact = _validate_live_tof_success(evidence_path)

    finite_ranges = np.concatenate(
        [values[np.isfinite(values)] for state in (fact["before"], fact["after"]) for values in state.values()]
    )
    range_min = float(finite_ranges.min())
    range_max = float(finite_ranges.max())
    if math.isclose(range_min, range_max):
        range_min -= 1.0e-3
        range_max += 1.0e-3
    delta_max_mm = max(float(np.nanmax(values)) for values in fact["delta"].values()) * 1_000.0

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.65), facecolor="#0e1116")
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.18, top=0.76, wspace=0.16)
    images = []
    for ax, name in zip(axes, ("tof0", "tof1"), strict=True):
        ax.set_facecolor("#263238")
        ax.set_title(name.upper(), color="white", fontsize=11, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        image = ax.imshow(
            np.ma.masked_invalid(fact["before"][name]),
            vmin=range_min,
            vmax=range_max,
            cmap="magma",
            interpolation="nearest",
        )
        image.cmap.set_bad("#455a64")
        images.append(image)

    headline = fig.text(
        0.04,
        0.94,
        f"LIVE DUAL ToF  ·  PASS  ·  job {fact['job_id']}",
        color="#66bb6a",
        fontsize=14,
        weight="bold",
    )
    stage = fig.text(0.04, 0.82, "", color="white", fontsize=10.5, family="monospace")
    footer = fig.text(
        0.04,
        0.055,
        (
            f"{fact['node']}  ·  5 mm command={fact['move_command_mm']:.3f} mm  ·  "
            f"final error={fact['move_final_error_mm']:.3f} mm  ·  "
            f"max hold drift={fact['hold_drift_mm']:.3f} mm"
        ),
        color="#b0bec5",
        fontsize=8,
    )
    states = ("before", "after", "delta")

    def update(frame: int):
        state = states[frame // 2]
        if state == "delta":
            for image, name in zip(images, ("tof0", "tof1"), strict=True):
                image.set_data(np.ma.masked_invalid(fact["delta"][name] * 1_000.0))
                image.set_cmap("viridis")
                image.set_clim(0.0, delta_max_mm)
            stage.set_text(f"|AFTER − BEFORE|  ·  shared finite rays  ·  scale 0–{delta_max_mm:.3f} mm")
        else:
            for image, name in zip(images, ("tof0", "tof1"), strict=True):
                image.set_data(np.ma.masked_invalid(fact[state][name]))
                image.set_cmap("magma")
                image.set_clim(range_min, range_max)
            if state == "before":
                frames = fact["before_frames"]
                stage.set_text(
                    f"MEASURED BEFORE  ·  sensor frames {frames['tof0']} / {frames['tof1']}  ·  "
                    f"shared scale {range_min:.3f}–{range_max:.3f} m"
                )
            else:
                frames = fact["after_frames"]
                stage.set_text(
                    f"MEASURED AFTER   ·  sensor frames {frames['tof0']} / {frames['tof1']}  ·  "
                    f"shared scale {range_min:.3f}–{range_max:.3f} m"
                )
        return [headline, stage, footer, *images]

    anim = animation.FuncAnimation(fig, update, frames=6, interval=500, blit=False)
    try:
        return _save_gif_atomic(anim, output_path)
    finally:
        plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--smoke-evidence",
        type=Path,
        help="Passing docs/evidence/smoke_<jobid>.json used to render the live dual-ToF GIF.",
    )
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    written = [
        write_tree(),
        write_widths(),
        write_fusion_gif(),
        write_gate0(),
        write_cutter(),
        write_import_gate_failure(),
        write_import_gate_success(),
    ]
    if args.smoke_evidence is not None:
        written.append(write_live_tof_success(args.smoke_evidence))
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
