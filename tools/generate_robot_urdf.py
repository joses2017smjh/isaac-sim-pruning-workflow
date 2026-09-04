#!/usr/bin/env python3
"""Generate and validate the pinned UR5e + mock-pruner URDF.

This is the CPU-only source stage. It deliberately expands the BDS Xacro
without requiring a sourced ROS installation: package lookup is restricted to
the reviewed package map, and both source repositories must match the manifest.
The canonical URDF keeps ``package://`` mesh URIs for a path-independent hash;
the companion ``*_abs.urdf`` is rewritten for Isaac's importer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import yaml

if __package__:
    from .rewrite_urdf_paths import load_package_map, rewrite_urdf
else:
    from rewrite_urdf_paths import load_package_map, rewrite_urdf

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = REPOSITORY_ROOT / "third_party" / "sources.yaml"
DEFAULT_PACKAGE_MAP = REPOSITORY_ROOT / "artifacts" / "urdf" / "package_map.json"
DEFAULT_RIG = (
    REPOSITORY_ROOT
    / "source"
    / "isaaclab_pruning"
    / "isaaclab_pruning"
    / "config"
    / "rigs"
    / "mock_pruner_vl53l8cx.yaml"
)
XACRO_VERSION = "2.1.1"
CALIBRATION_RELATIVE_PATH = (
    "branch_detection_system_description/config/robot_calibration.yaml"
)
ROOT_XACRO_RELATIVE_PATH = (
    "branch_detection_system_description/urdf/robot/robot.urdf.xacro"
)

JOINT_TO_CALIBRATION_SECTION = {
    "ur5e__shoulder_pan_joint": "shoulder",
    "ur5e__shoulder_lift_joint": "upper_arm",
    "ur5e__elbow_joint": "forearm",
    "ur5e__wrist_1_joint": "wrist_1",
    "ur5e__wrist_2_joint": "wrist_2",
    "ur5e__wrist_3_joint": "wrist_3",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_is_dirty(path: Path) -> bool:
    return bool(
        subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )


def _repo_relative(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(REPOSITORY_ROOT.resolve()))
    except ValueError:
        return str(resolved)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a mapping in {path}.")
    return payload


def _source_inputs(manifest_path: Path) -> dict[str, dict[str, str]]:
    manifest = _load_yaml(manifest_path)
    sources = manifest["sources"]
    selected: dict[str, dict[str, str]] = {}
    for name in ("branch_detection_system", "universal_robots_ros2_description"):
        expected = str(sources[name]["revision"])
        checkout = REPOSITORY_ROOT / "third_party" / "src" / name
        actual = _git_revision(checkout)
        if actual != expected:
            raise RuntimeError(
                f"{name} is at {actual}; manifest requires {expected}. "
                f"Run tools/fetch_sources.py --source {name}."
            )
        if _git_is_dirty(checkout):
            raise RuntimeError(f"{name} has local changes; refusing an unreviewed source expansion.")
        selected[name] = {
            "branch": str(sources[name]["branch"]),
            "expected_revision": expected,
            "actual_revision": actual,
            "clean": True,
        }
    return selected


def _patch_xacro_package_lookup(package_map: dict[str, Path]):
    try:
        import xacro
        from xacro import substitution_args
    except ImportError as error:
        raise RuntimeError(
            f"xacro {XACRO_VERSION} is required. Install it in an isolated environment "
            "or add its target directory to PYTHONPATH."
        ) from error

    version = importlib.metadata.version("xacro")
    if version != XACRO_VERSION:
        raise RuntimeError(f"xacro {XACRO_VERSION} is required; found {version}.")

    def _local_find(package: str) -> str:
        try:
            return str(package_map[package])
        except KeyError as error:
            raise RuntimeError(f"Xacro requested unreviewed package {package!r}.") from error

    substitution_args._eval_find = _local_find
    substitution_args._eval_dict["find"] = _local_find
    return xacro, version


def _canonicalize_package_paths(text: str, package_map: dict[str, Path]) -> str:
    """Replace resolved local package roots with portable package URIs."""
    canonical = text
    for package, root in sorted(
        package_map.items(), key=lambda item: len(str(item[1])), reverse=True
    ):
        root_text = str(root.resolve())
        canonical = canonical.replace(f"file://{root_text}/", f"package://{package}/")
        canonical = canonical.replace(f"{root_text}/", f"package://{package}/")
    return canonical


def _origin(joint: ET.Element) -> tuple[list[float], list[float]]:
    node = joint.find("origin")
    if node is None:
        return [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    xyz = [float(value) for value in node.attrib.get("xyz", "0 0 0").split()]
    rpy = [float(value) for value in node.attrib.get("rpy", "0 0 0").split()]
    return xyz, rpy


def _assert_vector(actual: list[float], expected: list[float], label: str) -> None:
    if len(actual) != len(expected) or any(
        abs(left - right) > 1.0e-12 for left, right in zip(actual, expected, strict=True)
    ):
        raise RuntimeError(f"{label} mismatch: generated {actual}, expected {expected}.")


def _validate_urdf(
    canonical: str,
    absolute: str,
    *,
    rig: dict[str, Any],
    calibration: dict[str, Any],
) -> dict[str, Any]:
    root = ET.fromstring(canonical)
    links = root.findall("link")
    joints = root.findall("joint")
    link_names = [node.attrib["name"] for node in links]
    joint_names = [node.attrib["name"] for node in joints]
    if len(link_names) != len(set(link_names)):
        raise RuntimeError("Generated URDF contains duplicate link names.")
    if len(joint_names) != len(set(joint_names)):
        raise RuntimeError("Generated URDF contains duplicate joint names.")

    forbidden = ("amiga__", "cart__", "linear_slider__")
    forbidden_names = sorted(
        name for name in (*link_names, *joint_names) if name.startswith(forbidden)
    )
    if forbidden_names:
        raise RuntimeError(f"Arm-only URDF unexpectedly contains: {forbidden_names}")

    active = sorted(
        node.attrib["name"] for node in joints if node.attrib.get("type") != "fixed"
    )
    if active != sorted(JOINT_TO_CALIBRATION_SECTION):
        raise RuntimeError(f"Expected exactly the six UR5e joints; generated {active}.")

    joints_by_name = {node.attrib["name"]: node for node in joints}
    kinematics = calibration["kinematics"]
    calibration_evidence: dict[str, Any] = {}
    for joint_name, section_name in JOINT_TO_CALIBRATION_SECTION.items():
        values = kinematics[section_name]
        expected_xyz = [float(values[key]) for key in ("x", "y", "z")]
        expected_rpy = [float(values[key]) for key in ("roll", "pitch", "yaw")]
        actual_xyz, actual_rpy = _origin(joints_by_name[joint_name])
        _assert_vector(actual_xyz, expected_xyz, f"{joint_name} xyz")
        _assert_vector(actual_rpy, expected_rpy, f"{joint_name} rpy")
        calibration_evidence[joint_name] = {"xyz": actual_xyz, "rpy": actual_rpy}

    hash_nodes = root.findall(".//param[@name='kinematics/hash']")
    generated_hashes = sorted({(node.text or "").strip() for node in hash_nodes})
    expected_hash = str(kinematics["hash"])
    if generated_hashes != [expected_hash]:
        raise RuntimeError(
            f"Generated kinematics hash {generated_hashes} does not match {expected_hash}."
        )

    expected_frames = {
        "camera0": rig["wrist_camera"]["physical_source_frame"]["offset_m"],
        "tof0": next(sensor for sensor in rig["sensors"] if sensor["name"] == "tof0")[
            "mount_offset_m"
        ],
        "tof1": next(sensor for sensor in rig["sensors"] if sensor["name"] == "tof1")[
            "mount_offset_m"
        ],
        "tool0": rig["control_eef_translation_in_source_frame_m"],
    }
    frame_evidence: dict[str, Any] = {}
    for frame, expected_xyz_raw in expected_frames.items():
        joint = joints_by_name[f"mock_pruner__base--{frame}"]
        actual_xyz, actual_rpy = _origin(joint)
        expected_xyz = [float(value) for value in expected_xyz_raw]
        _assert_vector(actual_xyz, expected_xyz, f"mock_pruner__base--{frame} xyz")
        _assert_vector(actual_rpy, [0.0, 0.0, 0.0], f"mock_pruner__base--{frame} rpy")
        frame_evidence[f"mock_pruner__base_to_{frame}"] = {
            "translation_m": actual_xyz,
            "rpy_rad": actual_rpy,
        }

    absolute_root = ET.fromstring(absolute)
    mesh_paths = sorted(
        {Path(node.attrib["filename"]).resolve() for node in absolute_root.findall(".//mesh")}
    )
    missing_meshes = [str(path) for path in mesh_paths if not path.is_file()]
    if missing_meshes:
        raise RuntimeError(f"Generated URDF has missing meshes: {missing_meshes}")
    meshes = [
        {"path": _repo_relative(path), "sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in mesh_paths
    ]

    fixed_joint_table = []
    for joint in joints:
        if joint.attrib.get("type") != "fixed":
            continue
        xyz, rpy = _origin(joint)
        parent = joint.find("parent")
        child = joint.find("child")
        fixed_joint_table.append(
            {
                "name": joint.attrib["name"],
                "parent": None if parent is None else parent.attrib["link"],
                "child": None if child is None else child.attrib["link"],
                "xyz": xyz,
                "rpy": rpy,
            }
        )
    fixed_joint_table.sort(key=lambda item: item["name"])
    fixed_joint_bytes = json.dumps(
        fixed_joint_table, sort_keys=True, separators=(",", ":")
    ).encode()

    return {
        "robot_name": root.attrib.get("name"),
        "link_count": len(links),
        "joint_count": len(joints),
        "active_joint_names": active,
        "forbidden_names": forbidden_names,
        "required_frames": sorted(
            name
            for name in (
                "mock_pruner__base",
                "mock_pruner__camera0",
                "mock_pruner__tof0",
                "mock_pruner__tof1",
                "mock_pruner__tool0",
            )
            if name in link_names
        ),
        "calibration_hash": expected_hash,
        "calibrated_joint_origins": calibration_evidence,
        "fixed_frames": frame_evidence,
        "fixed_joint_table": fixed_joint_table,
        "fixed_joint_table_sha256": hashlib.sha256(fixed_joint_bytes).hexdigest(),
        "meshes": meshes,
        "unresolved_uri_count": 0,
    }


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--package-map", type=Path, default=DEFAULT_PACKAGE_MAP)
    parser.add_argument("--rig", type=Path, default=DEFAULT_RIG)
    parser.add_argument("--canonical-output", type=Path)
    parser.add_argument("--absolute-output", type=Path)
    parser.add_argument("--evidence-output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    sources = _source_inputs(args.manifest)
    package_map = load_package_map(args.package_map)
    bds_root = package_map["branch_detection_system_description"]
    calibration_path = bds_root / Path(CALIBRATION_RELATIVE_PATH).relative_to(
        "branch_detection_system_description"
    )
    root_xacro = bds_root / Path(ROOT_XACRO_RELATIVE_PATH).relative_to(
        "branch_detection_system_description"
    )
    calibration = _load_yaml(calibration_path)
    rig = _load_yaml(args.rig)
    expected_calibration_hash = str(calibration["kinematics"]["hash"])

    xacro, xacro_version = _patch_xacro_package_lookup(package_map)
    sensor_offsets = {
        sensor["name"]: " ".join(str(value) for value in sensor["mount_offset_m"])
        for sensor in rig["sensors"]
        if sensor["name"] in {"tof0", "tof1"}
    }
    mappings = {
        "name": "pruning_robot",
        "robot_stack_qty": "2",
        "parent0": "world",
        "robot_part0": "ur5e",
        "parent1": "ur5e",
        "robot_part1": "mock_pruner",
        "ur_prefix": "ur5e__",
        "kinematics_params": str(calibration_path.resolve()),
        "tof0_offset": sensor_offsets["tof0"],
        "tof1_offset": sensor_offsets["tof1"],
    }
    document = xacro.process_file(str(root_xacro.resolve()), mappings=dict(mappings))
    expanded = document.toprettyxml(indent="  ")
    canonical = _canonicalize_package_paths(expanded, package_map)
    missing: list[str] = []
    absolute = rewrite_urdf(canonical, package_map, missing)
    if missing:
        raise RuntimeError(f"URDF path rewrite failed; missing files: {missing}")
    validation = _validate_urdf(canonical, absolute, rig=rig, calibration=calibration)

    canonical_sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    asset_id = (
        "ur5e_mock_pruner_"
        f"bds{sources['branch_detection_system']['actual_revision'][:8]}_"
        f"ur{sources['universal_robots_ros2_description']['actual_revision'][:8]}_"
        f"{expected_calibration_hash}_"
        f"urdf{canonical_sha256[:12]}"
    )
    asset_dir = REPOSITORY_ROOT / "artifacts" / "urdf" / asset_id
    canonical_output = args.canonical_output or asset_dir / f"{asset_id}.urdf"
    absolute_output = args.absolute_output or asset_dir / f"{asset_id}_abs.urdf"
    evidence_output = (
        args.evidence_output
        or REPOSITORY_ROOT / "docs" / "evidence" / f"urdf_generation_{asset_id}.json"
    )

    _write_text(canonical_output, canonical)
    _write_text(absolute_output, absolute)
    evidence = {
        "ok": True,
        "asset_id": asset_id,
        "generator": _repo_relative(Path(__file__)),
        "xacro_version": xacro_version,
        "sources": sources,
        "inputs": {
            "manifest": _repo_relative(args.manifest),
            "package_map": {
                "path": _repo_relative(args.package_map),
                "sha256": _sha256(args.package_map),
            },
            "rig": _repo_relative(args.rig),
            "root_xacro": {
                "path": _repo_relative(root_xacro),
                "sha256": _sha256(root_xacro),
            },
            "calibration": {
                "path": _repo_relative(calibration_path),
                "sha256": _sha256(calibration_path),
                "hash": expected_calibration_hash,
            },
            "mappings": mappings,
        },
        "outputs": {
            "canonical_urdf": {
                "path": _repo_relative(canonical_output),
                "sha256": canonical_sha256,
                "bytes": len(canonical.encode()),
            },
            "isaac_absolute_urdf": {
                "path": _repo_relative(absolute_output),
                "sha256": hashlib.sha256(absolute.encode()).hexdigest(),
                "bytes": len(absolute.encode()),
            },
        },
        "validation": validation,
    }
    _write_text(evidence_output, json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:  # noqa: BLE001 - CLI must emit one actionable failure
        print(f"error: {error}", file=sys.stderr)
        raise
