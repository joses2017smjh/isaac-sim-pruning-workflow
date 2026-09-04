from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools import render_demos

ROOT = Path(__file__).resolve().parents[1]
IMPORT_FAILURE = ROOT / "docs" / "evidence" / "urdf_import_21125352.json"
IMPORT_SUCCESS = ROOT / "docs" / "evidence" / "urdf_import_21136450.json"


def _passing_smoke_report(job_id: str = "12345") -> dict:
    before = {}
    after = {}
    changes = {}
    for index, name in enumerate(("tof0", "tof1")):
        first = np.full((1, 8, 8), 0.40 + index * 0.02)
        second = first.copy()
        second[0, 3, 4] += 0.002 + index * 0.001
        before[name] = first.astype(object).tolist()
        after[name] = second.astype(object).tolist()
        before[name][0][0][0] = None
        after[name][0][0][0] = None
        changes[name] = {
            "shared_finite_pixels": 63,
            "median_abs_delta_m": 0.0,
            "max_abs_delta_m": 0.002 + index * 0.001,
        }

    def tof_state(frame: int) -> dict:
        return {
            "source": "live_multi_mesh_ray_caster",
            "noise_enabled": False,
            "range_limits_m": [0.03, 3.4],
            "sensors": {
                name: {
                    "class": "MultiMeshRayCasterCamera",
                    "frame": [frame],
                    "observation": {"shape": [1, 8, 8], "valid_fraction": 63 / 64},
                }
                for name in ("tof0", "tof1")
            },
        }

    widths = {"A_flow": 150, "B_tof": 278, "C_metric": 86, "D_fused": 86}
    return {
        "ok": True,
        "job_id": job_id,
        "node": "test-gpu",
        "asset_id": "robot_hash",
        "usd": "/tmp/robot.usda",
        "usd_evidence": "/tmp/import.json",
        "observation_space": widths,
        "obs_last_dim": widths,
        "tool_frame": {
            "body_to_tool_distance_m": [0.1601525],
            "hold_translation_drift_m": [0.001],
            "hold_rotation_drift_rad": [0.001],
            "move_initial_error_m": [0.005],
            "move_final_error_m": [0.002],
        },
        "tof_before": tof_state(2),
        "tof_after": tof_state(4),
        "tof_range_change": changes,
        "tof_raw_frames": {"before": before, "after": after},
        "contact": {"finite": True},
        "n_arm_joints": 6,
        "slider_in_joint_names": False,
    }


def test_reviewed_import_failure_contract_and_gif(tmp_path: Path) -> None:
    contract = render_demos._validate_import_gate_failure(IMPORT_FAILURE)
    assert contract["job_id"] == "21125352"
    assert contract["actual_path"] != contract["expected_path"]

    output = tmp_path / "failure.gif"
    assert render_demos.write_import_gate_failure(IMPORT_FAILURE, output) == output
    with Image.open(output) as image:
        assert image.format == "GIF"
        assert image.n_frames == 3


def test_import_failure_renderer_rejects_tampering_without_overwrite(tmp_path: Path) -> None:
    report = json.loads(IMPORT_FAILURE.read_text(encoding="utf-8"))
    report["status"] = "complete"
    evidence = tmp_path / "urdf_import_21125352.json"
    evidence.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "failure.gif"
    output.write_bytes(b"keep")

    with pytest.raises(render_demos.EvidenceContractError, match="status must be failed"):
        render_demos.write_import_gate_failure(evidence, output)
    assert output.read_bytes() == b"keep"


def test_reviewed_import_success_contract_and_gif(tmp_path: Path) -> None:
    contract = render_demos._validate_import_gate_success(IMPORT_SUCCESS)
    assert contract["job_id"] == "21136450"
    assert contract["asset_id"] == render_demos.IMPORT_SUCCESS_ASSET_ID
    assert contract["joint_names"] == list(render_demos._EXPECTED_UR_JOINTS)
    assert contract["frame_names"] == list(render_demos._EXPECTED_PROVENANCE_FRAMES)
    assert contract["root_sha256"] == render_demos.IMPORT_SUCCESS_ROOT_SHA256

    output = tmp_path / "success.gif"
    assert render_demos.write_import_gate_success(IMPORT_SUCCESS, output) == output
    with Image.open(output) as image:
        assert image.format == "GIF"
        assert image.n_frames == 4


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("asset", "expected the reviewed content-addressed asset"),
        ("stage", "stage validation must have ok=true"),
        ("joint", "six exact reviewed UR revolute joints"),
        ("slider", "hardware slider must be absent"),
        ("frame", "tof1 transform was not provenance-verified"),
        ("hash", "root-layer SHA-256 is not the reviewed hash"),
    ],
)
def test_import_success_renderer_rejects_tampering_without_overwrite(
    tmp_path: Path, tamper: str, message: str
) -> None:
    report = json.loads(IMPORT_SUCCESS.read_text(encoding="utf-8"))
    if tamper == "asset":
        report["asset_id"] = "different-asset"
    elif tamper == "stage":
        report["stage_validation"]["ok"] = False
    elif tamper == "joint":
        report["stage_validation"]["active_ur_joints"][0]["type"] = "PhysicsPrismaticJoint"
    elif tamper == "slider":
        report["stage_validation"]["linear_slider_prim_paths"] = ["/pruning_robot/slider"]
    elif tamper == "frame":
        report["stage_validation"]["frames"]["mock_pruner__tof1"]["provenance_transform_verified"] = False
    elif tamper == "hash":
        root_layer = report["output"]["root_layer"]
        root_entry = next(entry for entry in report["output"]["files"] if entry["path"] == root_layer)
        root_entry["sha256"] = "0" * 64

    evidence = tmp_path / "urdf_import_21136450.json"
    evidence.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "success.gif"
    output.write_bytes(b"keep")

    with pytest.raises(render_demos.EvidenceContractError, match=message):
        render_demos.write_import_gate_success(evidence, output)
    assert output.read_bytes() == b"keep"


def test_live_tof_success_contract_accepts_measured_frame_pair(tmp_path: Path) -> None:
    evidence = tmp_path / "smoke_12345.json"
    payload = _passing_smoke_report()
    evidence.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
    contract = render_demos._validate_live_tof_success(evidence)

    assert contract["job_id"] == "12345"
    assert contract["before"]["tof0"].shape == (8, 8)
    assert np.isclose(np.nanmax(contract["delta"]["tof1"]), 0.003)
    assert contract["after_frames"]["tof0"] > contract["before_frames"]["tof0"]


def test_live_tof_renderer_fails_closed_without_overwrite(tmp_path: Path) -> None:
    report = _passing_smoke_report()
    report["tof_range_change"]["tof1"]["max_abs_delta_m"] = 0.0
    evidence = tmp_path / "smoke_12345.json"
    evidence.write_text(json.dumps(report, allow_nan=False), encoding="utf-8")
    output = tmp_path / "success.gif"
    output.write_bytes(b"keep")

    with pytest.raises(render_demos.EvidenceContractError, match="raw frames disagree"):
        render_demos.write_live_tof_success(evidence, output)
    assert output.read_bytes() == b"keep"
