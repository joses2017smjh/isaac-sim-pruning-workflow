from __future__ import annotations

import json
from pathlib import Path

import pytest

from hpc.inner import import_urdf


def _valid_input(tmp_path: Path, *, asset_id: str = "robot_urdf0123456789") -> tuple[dict[str, str], Path, Path]:
    urdf = tmp_path / "artifacts" / "urdf" / asset_id / f"{asset_id}_abs.urdf"
    urdf.parent.mkdir(parents=True)
    urdf.write_text('<robot name="pruning_robot"/>\n', encoding="utf-8")

    provenance = tmp_path / "docs" / "evidence" / f"urdf_generation_{asset_id}.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(
        json.dumps(
            {
                "ok": True,
                "asset_id": asset_id,
                "outputs": {
                    "isaac_absolute_urdf": {
                        "path": str(urdf.relative_to(tmp_path)),
                        "bytes": urdf.stat().st_size,
                        "sha256": import_urdf.sha256_file(urdf),
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    environment = {
        "PRUNING_ROOT": str(tmp_path),
        "PRUNING_ASSET_ID": asset_id,
        "PRUNING_URDF": str(urdf),
        "PRUNING_URDF_PROVENANCE": str(provenance),
        "SLURM_JOB_ID": "12345",
        "BHL_STACK": "v60",
    }
    return environment, urdf, provenance


def test_cpu_preflight_binds_asset_hash_and_immutable_output(tmp_path: Path) -> None:
    environment, urdf, provenance = _valid_input(tmp_path)

    plan = import_urdf.build_import_plan(environment)
    report = import_urdf.initial_report(plan, environment)

    assert plan.urdf_path == urdf
    assert plan.provenance_path == provenance
    assert plan.output_dir == tmp_path / "artifacts" / "usd" / environment["PRUNING_ASSET_ID"]
    assert plan.usd_path == plan.output_dir / f"{environment['PRUNING_ASSET_ID']}_abs" / (
        f"{environment['PRUNING_ASSET_ID']}_abs.usda"
    )
    assert plan.report_path == tmp_path / "docs" / "evidence" / "urdf_import_12345.json"
    assert report["status"] == "pending"
    assert report["input"]["provenance_asset_id_verified"] is True
    assert report["input"]["absolute_urdf_sha256_verified"] is True
    assert report["converter_config"]["usd_file_name"] == (
        f"{environment['PRUNING_ASSET_ID']}_abs/{environment['PRUNING_ASSET_ID']}_abs.usda"
    )
    assert report["converter_config"]["merge_fixed_joints"] is False


@pytest.mark.parametrize("missing", import_urdf.REQUIRED_ENVIRONMENT)
def test_cpu_preflight_requires_explicit_inputs(tmp_path: Path, missing: str) -> None:
    environment, _, _ = _valid_input(tmp_path)
    del environment[missing]
    with pytest.raises(import_urdf.ImportPreflightError, match=missing):
        import_urdf.build_import_plan(environment)


def test_cpu_preflight_rejects_provenance_asset_mismatch(tmp_path: Path) -> None:
    environment, _, provenance = _valid_input(tmp_path)
    document = json.loads(provenance.read_text(encoding="utf-8"))
    document["asset_id"] = "different_asset"
    provenance.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(import_urdf.ImportPreflightError, match="asset ID mismatch"):
        import_urdf.build_import_plan(environment)


def test_cpu_preflight_rejects_changed_absolute_urdf(tmp_path: Path) -> None:
    environment, urdf, _ = _valid_input(tmp_path)
    urdf.write_text('<robot name="tampered"/>\n', encoding="utf-8")

    with pytest.raises(import_urdf.ImportPreflightError, match="SHA-256 mismatch"):
        import_urdf.build_import_plan(environment)


def test_cpu_preflight_refuses_existing_destination(tmp_path: Path) -> None:
    environment, _, _ = _valid_input(tmp_path)
    destination = tmp_path / "artifacts" / "usd" / environment["PRUNING_ASSET_ID"]
    destination.mkdir(parents=True)

    with pytest.raises(import_urdf.ImportPreflightError, match="destination already exists"):
        import_urdf.build_import_plan(environment)


def test_cpu_preflight_refuses_existing_evidence(tmp_path: Path) -> None:
    environment, _, _ = _valid_input(tmp_path)
    report = tmp_path / "docs" / "evidence" / "urdf_import_12345.json"
    report.write_text("{}\n", encoding="utf-8")

    with pytest.raises(import_urdf.ImportPreflightError, match="evidence report already exists"):
        import_urdf.build_import_plan(environment)


def test_output_inventory_records_every_hash(tmp_path: Path) -> None:
    output_dir = tmp_path / "artifacts" / "usd" / "asset"
    payload = output_dir / "payloads" / "robot.usda"
    payload.parent.mkdir(parents=True)
    root_layer = output_dir / "asset.usda"
    root_layer.write_bytes(b"root")
    payload.write_bytes(b"payload")

    inventory = import_urdf.inventory_output(output_dir, tmp_path)

    assert [entry["path"] for entry in inventory] == [
        "artifacts/usd/asset/asset.usda",
        "artifacts/usd/asset/payloads/robot.usda",
    ]
    assert all(len(entry["sha256"]) == 64 for entry in inventory)


def test_slurm_wrapper_has_no_stale_asset_default() -> None:
    wrapper = (Path(__file__).resolve().parents[1] / "hpc" / "slurm" / "import_urdf.sbatch").read_text(encoding="utf-8")
    assert "ur5e_pruner_abs.urdf" not in wrapper
    assert "${PRUNING_ASSET_ID:?" in wrapper
    assert "${PRUNING_URDF:?" in wrapper
    assert "${PRUNING_URDF_PROVENANCE:?" in wrapper
    assert "urdf_import_${SLURM_JOB_ID}.json" in wrapper
    assert 'mkdir -p "$PRUNING_ROOT/logs" "$PRUNING_USD_DIR"' not in wrapper
    assert 'report.get("status") == "complete"' in wrapper
    assert 'report.get("ok") is True' in wrapper
    assert 'report.get("imported") is True' in wrapper
    assert 'report["stage_validation"].get("ok") is True' in wrapper
    assert 'raise SystemExit(f"import postflight: root layer is missing:' in wrapper
