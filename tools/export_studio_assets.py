#!/usr/bin/env python3
"""Prepare a LOCAL browser robot bundle; this does not grant redistribution rights.

The input must be the generated absolute-path URDF. No source file is modified.
The output contains copied CAD meshes and must stay outside Git/public builds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "source/isaaclab_pruning/isaaclab_pruning/config/robot/ur5e_pruner.yaml"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def local_mesh(uri, urdf_dir):
    parsed = urlparse(uri)
    if parsed.scheme not in ("", "file") or parsed.netloc not in ("", "localhost"):
        raise ValueError(f"Only resolved local mesh files are supported: {uri}")
    path = Path(unquote(parsed.path))
    path = (urdf_dir / path).resolve() if not path.is_absolute() else path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.suffix.lower() not in (".stl", ".dae"):
        raise ValueError(f"Unsupported browser mesh format: {path.suffix}")
    if path.suffix.lower() == ".dae":
        document = ET.parse(path)
        images = document.findall(".//{*}library_images/{*}image/{*}init_from")
        if any((image.text or "").strip() for image in images):
            raise ValueError(
                "DAE external images need explicit bundling; refusing a broken or remote texture reference"
            )
    return path


def glb_document(path):
    """Read only a self-contained glTF 2 binary, with no external resource URIs."""
    data = Path(path).read_bytes()
    if len(data) < 20:
        raise ValueError("Truncated GLB")
    magic, version, length, chunk_length, chunk_type = struct.unpack_from("<4sIIII", data)
    if magic != b"glTF" or version != 2 or length != len(data) or chunk_type != 0x4E4F534A:
        raise ValueError("Expected glTF 2 GLB with an initial JSON chunk")
    if 20 + chunk_length > len(data):
        raise ValueError("Truncated GLB JSON chunk")
    document = json.loads(data[20 : 20 + chunk_length])
    for resource in document.get("buffers", []) + document.get("images", []):
        if "uri" in resource:
            raise ValueError("GLB must embed all buffers and images")
    names = {node.get("name") for node in document.get("nodes", [])}
    expected = {f"tree{index}_{part}" for index in (0, 1) for part in ("TRUNK", "BRANCH", "SPUR")}
    if not expected.issubset(names):
        raise ValueError("GLB is missing one or more of the six original tree nodes")
    return document


def export_bundle(urdf, output, config, *, orchard_glb=None):
    urdf, output = Path(urdf).resolve(), Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"Refusing to replace existing bundle: {output}")
    document = ET.parse(urdf)
    robot = document.getroot()
    if robot.tag != "robot":
        raise ValueError("Expected URDF robot root")
    joint_order = [joint["name"] for joint in config["joints"]["arm"]]
    movable = [joint for joint in robot.findall("joint") if joint.get("type") != "fixed"]
    if len(joint_order) != 6 or {joint.get("name") for joint in movable} != set(joint_order):
        raise ValueError("URDF must contain exactly the six configured arm joints, with no slider")
    if len(movable) != 6 or any(joint.get("type") != "revolute" for joint in movable):
        raise ValueError("Expected six distinct revolute joints")
    if robot.findall(".//texture"):
        raise ValueError("URDF texture elements need explicit bundling")
    sources = {}
    for mesh in robot.findall(".//mesh"):
        source = local_mesh(mesh.attrib["filename"], urdf.parent)
        digest = sha256(source)
        relative = f"meshes/{digest}{source.suffix.lower()}"
        sources[relative] = source
        mesh.set("filename", relative)
    if not sources:
        raise ValueError("URDF has no mesh assets")
    if orchard_glb is not None:
        glb_document(orchard_glb)
    # Validate everything before creating the output directory. Never edit source
    # XML, source CAD, or an earlier generated bundle.
    output.mkdir(parents=True)
    (output / "meshes").mkdir()
    inventory = []
    for relative, source in sorted(sources.items()):
        target = output / relative
        shutil.copyfile(source, target)
        inventory.append({"file": relative, "sha256": sha256(target), "bytes": target.stat().st_size})
    document.write(output / "robot.urdf", encoding="utf-8", xml_declaration=True)
    manifest = {
        "schema_version": 1,
        "robot_id": config["name"],
        "joint_order": joint_order,
        "units": "m",
        "world_up": "Z",
        "quaternion_order": "wxyz",
        "robot": {
            "urdf": "robot.urdf",
            "base_position_m": [0.0, 0.0, 0.7],
            "base_quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            "sha256": sha256(output / "robot.urdf"),
        },
        "provenance": {
            "source_urdf_sha256": sha256(urdf),
            "exporter_sha256": sha256(__file__),
            "mesh_inventory": inventory,
            "local_use_only": True,
            "redistribution_rights_cleared": False,
            "scope": "Kinematic browser preview, not simulator physics or policy inference",
        },
    }
    if orchard_glb is not None:
        shutil.copyfile(orchard_glb, output / "orchard.glb")
        manifest["scene"] = {
            "gltf": "orchard.glb",
            "tree_count": 2,
            "gltf_up": "Y",
            "position_m": [0.31827075, 1.09701897, -0.08344020],
            "quaternion_wxyz": [math.cos(math.radians(75)), 0.0, 0.0, math.sin(math.radians(75))],
            "sha256": sha256(output / "orchard.glb"),
            "scope": "Original uncut scene, rebased to tree0 trunk; viewer converts glTF Y-up to Z-up before placement",
        }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--urdf", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--orchard-glb", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    asset_id = config["usd"]["asset_id"]
    urdf = args.urdf or ROOT / "artifacts/urdf" / asset_id / f"{asset_id}_abs.urdf"
    result = export_bundle(urdf, args.output_dir, config, orchard_glb=args.orchard_glb)
    print(json.dumps({"robot_id": result["robot_id"], "meshes": len(result["provenance"]["mesh_inventory"])}))


if __name__ == "__main__":
    main()
