from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest

from isaaclab_pruning.robot import (
    ALLOW_STALE_USD_ENV,
    RUNTIME_ASSET_ID_ENV,
    RUNTIME_USD_EVIDENCE_ENV,
    assert_runtime_usd_ready,
    imported_usd_path,
    load_ur5e_pruner_config,
    load_ur5e_pruner_spec,
)
from isaaclab_pruning.usd import imported_usd_payload_paths, load_imported_usd

_USD = imported_usd_path()
_ASSET_ID = "ur5e_mock_pruner_bdsdfede4c0_ur18e6f603_calib_3941312424972580002_urdf6b02ce9330be"


def test_yaml_points_at_the_imported_usd() -> None:
    path = imported_usd_path()
    config = load_ur5e_pruner_config()
    assert path.name == f"{_ASSET_ID}_abs.usda"
    assert "artifacts/usd" in str(path)
    assert config["usd"]["asset_id"] == _ASSET_ID
    assert config["usd"]["import_job"] == "21136450"
    assert config["usd"]["import_evidence"] == "docs/evidence/urdf_import_21136450.json"
    assert config["usd"]["status"] == "validated_content_addressed"
    assert config["usd"]["hardware_transform_match"] is True
    assert config["usd"]["reimport_required"] is False


def test_promoted_config_matches_tracked_import_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_ur5e_pruner_config()
    evidence_path = root / config["usd"]["import_evidence"]
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    assert evidence["status"] == "complete"
    assert evidence["ok"] is True
    assert evidence["imported"] is True
    assert evidence["asset_id"] == config["usd"]["asset_id"]
    assert evidence["job"]["id"] == config["usd"]["import_job"]
    assert evidence["output"]["root_layer"] == config["usd"]["relative_path"]
    assert evidence["stage_validation"]["ok"] is True
    assert evidence["stage_validation"]["active_ur_joint_count"] == 6
    assert evidence["stage_validation"]["linear_slider_prim_paths"] == []


def test_stale_default_usd_is_diagnostics_only(monkeypatch) -> None:
    config = deepcopy(load_ur5e_pruner_config())
    config["usd"]["status"] = "stale_generated_snapshot"
    config["usd"]["reimport_required"] = True
    monkeypatch.delenv("PRUNING_USD", raising=False)
    monkeypatch.delenv(RUNTIME_ASSET_ID_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_USD_EVIDENCE_ENV, raising=False)
    monkeypatch.delenv(ALLOW_STALE_USD_ENV, raising=False)
    with pytest.raises(RuntimeError, match="stale generated snapshot"):
        assert_runtime_usd_ready(config)
    untrusted = deepcopy(config)
    untrusted["usd"].pop("asset_id", None)
    untrusted["usd"].pop("import_evidence", None)
    with pytest.raises(RuntimeError, match="Runtime USD selection requires"):
        assert_runtime_usd_ready(untrusted, explicit_asset=True, usd_path="/tmp/not-trusted.usda")
    monkeypatch.setenv(ALLOW_STALE_USD_ENV, "1")
    assert_runtime_usd_ready(config)


@pytest.mark.skipif(not _USD.is_file(), reason="USD import artifacts are not on this machine")
def test_promoted_default_runtime_usd_matches_completed_evidence(monkeypatch) -> None:
    monkeypatch.delenv("PRUNING_USD", raising=False)
    monkeypatch.delenv(RUNTIME_ASSET_ID_ENV, raising=False)
    monkeypatch.delenv(RUNTIME_USD_EVIDENCE_ENV, raising=False)
    monkeypatch.delenv(ALLOW_STALE_USD_ENV, raising=False)
    assert_runtime_usd_ready()


def _write_runtime_asset_evidence(tmp_path, asset_id="robot_abc123"):
    usd = (tmp_path / f"{asset_id}.usda").resolve()
    usd.write_text("#usda 1.0\n", encoding="utf-8")
    digest = hashlib.sha256(usd.read_bytes()).hexdigest()
    frames = {
        name: {"path": f"/Robot/{name}"}
        for name in (
            "ur5e__base_link",
            "mock_pruner__base",
            "mock_pruner__camera0",
            "mock_pruner__tof0",
            "mock_pruner__tof1",
            "mock_pruner__tool0",
        )
    }
    evidence = (tmp_path / "import.json").resolve()
    evidence.write_text(
        json.dumps(
            {
                "status": "complete",
                "ok": True,
                "imported": True,
                "asset_id": asset_id,
                "output": {
                    "root_layer": str(usd),
                    "files": [{"path": str(usd), "sha256": digest}],
                },
                "stage_validation": {
                    "ok": True,
                    "active_ur_joint_count": 6,
                    "linear_slider_prim_paths": [],
                    "frames": frames,
                },
            }
        ),
        encoding="utf-8",
    )
    return usd, evidence, asset_id


def test_explicit_runtime_usd_requires_matching_completed_evidence(tmp_path, monkeypatch) -> None:
    config = load_ur5e_pruner_config()
    usd, evidence, asset_id = _write_runtime_asset_evidence(tmp_path)
    monkeypatch.setenv(RUNTIME_ASSET_ID_ENV, asset_id)
    monkeypatch.setenv(RUNTIME_USD_EVIDENCE_ENV, str(evidence))

    assert_runtime_usd_ready(config, explicit_asset=True, usd_path=usd)

    usd.write_text("#usda 1.0\n# changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        assert_runtime_usd_ready(config, explicit_asset=True, usd_path=usd)


@pytest.mark.skipif(not _USD.is_file(), reason="USD import artifacts are not on this machine")
def test_imported_usd_composes_robot_and_physx_payloads() -> None:
    names = {path.name for path in imported_usd_payload_paths()}
    assert f"{_ASSET_ID}_abs.usda" in names
    assert "robot.usda" in names
    assert "physx.usda" in names


@pytest.mark.skipif(not _USD.is_file(), reason="USD import artifacts are not on this machine")
def test_imported_usd_has_ur5e_and_mock_pruner_not_slider() -> None:
    text = load_imported_usd()
    spec = load_ur5e_pruner_spec()
    for joint in spec.arm_joints:
        assert joint.name in text
    assert "mock_pruner__tool0" in text
    assert "mock_pruner__tof0" in text
    assert "linear_slider__joint1" not in text
    for expr in spec.joint_names_expr:
        assert "linear_slider" not in expr
