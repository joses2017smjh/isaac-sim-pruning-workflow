from __future__ import annotations

import json
from pathlib import Path

from tools.generate_robot_urdf import _canonicalize_package_paths

ROOT = Path(__file__).resolve().parents[1]
ASSET_ID = (
    "ur5e_mock_pruner_bdsdfede4c0_ur18e6f603_"
    "calib_3941312424972580002_urdf6b02ce9330be"
)
EVIDENCE = ROOT / "docs" / "evidence" / f"urdf_generation_{ASSET_ID}.json"


def test_package_paths_are_canonicalized_without_touching_other_paths(tmp_path: Path) -> None:
    package = tmp_path / "share" / "demo_description"
    source = (
        f'<mesh filename="file://{package}/meshes/body.stl"/>'
        '<mesh filename="/opt/reviewed/other.stl"/>'
    )

    canonical = _canonicalize_package_paths(source, {"demo_description": package})

    assert 'filename="package://demo_description/meshes/body.stl"' in canonical
    assert 'filename="/opt/reviewed/other.stl"' in canonical


def test_reviewed_urdf_generation_evidence_is_source_and_transform_complete() -> None:
    report = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert report["ok"] is True
    assert report["asset_id"] == ASSET_ID
    assert report["xacro_version"] == "2.1.1"
    assert report["sources"]["branch_detection_system"]["actual_revision"] == (
        "dfede4c0f251358ebed7a1f90ff887847c2fbeb0"
    )
    assert report["sources"]["universal_robots_ros2_description"]["actual_revision"] == (
        "18e6f603b3ebc2ec479fecb62d6be544b15755e9"
    )
    assert all(source["clean"] for source in report["sources"].values())
    assert report["inputs"]["calibration"]["hash"] == "calib_3941312424972580002"
    assert report["outputs"]["canonical_urdf"]["sha256"] == (
        "6b02ce9330beff2ceded6a870d68c4fde1601c1c5f35a0ddb1cef98d1fa25c32"
    )
    assert report["outputs"]["isaac_absolute_urdf"]["sha256"] == (
        "a5c04da197c7de2588f1716bb7b25fee47b500ee98856bdc4d640fea90218f44"
    )

    validation = report["validation"]
    assert len(validation["active_joint_names"]) == 6
    assert validation["forbidden_names"] == []
    assert validation["required_frames"] == [
        "mock_pruner__base",
        "mock_pruner__camera0",
        "mock_pruner__tof0",
        "mock_pruner__tof1",
        "mock_pruner__tool0",
    ]
    assert validation["fixed_frames"]["mock_pruner__base_to_tool0"]["translation_m"] == [
        0.0,
        0.0,
        0.1601525,
    ]
    assert validation["unresolved_uri_count"] == 0
    assert len(validation["fixed_joint_table_sha256"]) == 64
    assert len(validation["meshes"]) == 18
    assert all(len(mesh["sha256"]) == 64 for mesh in validation["meshes"])
