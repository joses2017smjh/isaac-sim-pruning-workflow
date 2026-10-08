"""Portable local browser bundles must not leak original paths or fetch remote assets."""

import copy
import json
import struct
import xml.etree.ElementTree as ET

import pytest

from tools.export_studio_assets import export_bundle, glb_document, local_mesh, sha256


def inputs(tmp_path):
    mesh = tmp_path / "source.stl"
    mesh.write_bytes(b"solid fixture\nendsolid fixture\n")
    joint_order = [f"joint{index}" for index in range(6)]
    joints = "".join(f'<joint name="{name}" type="revolute"/>' for name in joint_order)
    urdf = tmp_path / "source.urdf"
    urdf.write_text(
        f'<robot name="fixture"><link name="base"><visual><mesh filename="{mesh}"/></visual></link>{joints}</robot>'
    )
    config = {"name": "fixture", "joints": {"arm": [{"name": name} for name in joint_order]}}
    return urdf, config


def test_bundle_relative_references_hashes_and_source_immutable(tmp_path):
    urdf, config = inputs(tmp_path)
    before = sha256(urdf)
    output = tmp_path / "bundle"
    result = export_bundle(urdf, output, config)
    assert sha256(urdf) == before
    assert result["provenance"]["source_urdf_sha256"] == before
    assert result["provenance"]["redistribution_rights_cleared"] is False
    assert str(tmp_path) not in (output / "manifest.json").read_text()
    assert str(tmp_path) not in (output / "robot.urdf").read_text()
    reference = ET.parse(output / "robot.urdf").find(".//mesh").get("filename")
    assert reference.startswith("meshes/") and (output / reference).is_file()
    assert result["provenance"]["mesh_inventory"][0]["sha256"] == sha256(output / reference)


def test_existing_bundle_is_never_overwritten(tmp_path):
    urdf, config = inputs(tmp_path)
    output = tmp_path / "bundle"
    output.mkdir()
    with pytest.raises(FileExistsError):
        export_bundle(urdf, output, config)


def test_invalid_joint_contract_rejected_before_output(tmp_path):
    urdf, config = inputs(tmp_path)
    config = copy.deepcopy(config)
    config["joints"]["arm"][0]["name"] = "slider"
    with pytest.raises(ValueError, match="six configured"):
        export_bundle(urdf, tmp_path / "bundle", config)
    assert not (tmp_path / "bundle").exists()


@pytest.mark.parametrize("uri", ["https://example.com/mesh.stl", "package://robot/mesh.stl", "file://host/mesh.stl"])
def test_remote_unresolved_meshes_rejected(tmp_path, uri):
    with pytest.raises(ValueError, match="local mesh"):
        local_mesh(uri, tmp_path)


def test_dae_external_images_rejected(tmp_path):
    path = tmp_path / "source.dae"
    path.write_text(
        '<COLLADA xmlns="urn:collada"><library_images><image><init_from>remote.png</init_from>'
        "</image></library_images></COLLADA>"
    )
    with pytest.raises(ValueError, match="external images"):
        local_mesh(str(path), tmp_path)


def glb(path, *, images=None, missing_tree=False):
    names = [f"tree{index}_{part}" for index in (0, 1) for part in ("TRUNK", "BRANCH", "SPUR")]
    document = {"asset": {"version": "2.0"}, "nodes": [{"name": name} for name in names], "images": images or []}
    if missing_tree:
        document["nodes"].pop()
    data = json.dumps(document).encode()
    data += b" " * (-len(data) % 4)
    path.write_bytes(struct.pack("<4sIIII", b"glTF", 2, 20 + len(data), len(data), 0x4E4F534A) + data)


def test_glb_contract_and_scene_placement(tmp_path):
    urdf, config = inputs(tmp_path)
    scene = tmp_path / "scene.glb"
    glb(scene)
    result = export_bundle(urdf, tmp_path / "bundle", config, orchard_glb=scene)
    assert result["scene"]["tree_count"] == 2
    assert result["scene"]["sha256"] == sha256(scene)
    assert len(glb_document(scene)["nodes"]) == 6


@pytest.mark.parametrize("kwargs", [{"images": [{"uri": "outside.png"}]}, {"missing_tree": True}])
def test_incomplete_or_nonlocal_glb_rejected(tmp_path, kwargs):
    scene = tmp_path / "scene.glb"
    glb(scene, **kwargs)
    with pytest.raises(ValueError):
        glb_document(scene)


def test_truncated_glb_rejected(tmp_path):
    scene = tmp_path / "scene.glb"
    scene.write_bytes(b"glTF")
    with pytest.raises(ValueError, match="Truncated"):
        glb_document(scene)
