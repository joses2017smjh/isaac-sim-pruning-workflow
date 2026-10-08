"""Local GLB export contract; generated proprietary assets are optional in CI."""

import importlib.util
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def exporter():
    path = Path(__file__).resolve().parents[1] / "tools/export_studio_orchard.py"
    spec = importlib.util.spec_from_file_location("studio_orchard_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scene_objects():
    return [
        SimpleNamespace(name=name, type="MESH")
        for name in sorted(exporter().TREE_NAMES | {"ground", "post0", "wire0_0"})
    ]


def gltf_fixture():
    return {
        "asset": {"version": "2.0"},
        "buffers": [{"byteLength": 4}],
        "images": [{"bufferView": 0, "mimeType": "image/png", "name": "bark"}],
        "nodes": [{"name": name, "mesh": index} for index, name in enumerate(sorted(exporter().TREE_NAMES))],
        "meshes": [{} for _ in range(6)],
    }


def write_glb(path, data):
    chunk = json.dumps(data).encode()
    chunk += b" " * (-len(chunk) % 4)
    payload = struct.pack("<I4s", len(chunk), b"JSON") + chunk + struct.pack("<I4s", 4, b"BIN\x00") + b"1234"
    path.write_bytes(struct.pack("<4sII", b"glTF", 2, 12 + len(payload)) + payload)
    return path


def test_selects_both_original_trees_and_scenery_without_robot_lights_or_camera():
    objects = scene_objects() + [
        SimpleNamespace(name="Sun", type="LIGHT"),
        SimpleNamespace(name="Camera", type="CAMERA"),
        SimpleNamespace(name="Robot", type="MESH"),
    ]
    selected = exporter().selected_objects(objects)
    assert {obj.name for obj in selected} == exporter().TREE_NAMES | {"ground", "post0", "wire0_0"}


@pytest.mark.parametrize("missing", ["tree1_TRUNK", "tree1_SPUR", "ground", "post0", "wire0_0"])
def test_missing_original_tree_or_scenery_fails_instead_of_synthesizing(missing):
    with pytest.raises(ValueError):
        exporter().selected_objects([obj for obj in scene_objects() if obj.name != missing])


def test_existing_export_refused_before_loading_blender(tmp_path):
    existing = tmp_path / "orchard.glb"
    existing.write_bytes(b"unchanged")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        exporter().main(["--template", "unused.blend", "--texture-root", "unused", "--output-dir", str(tmp_path)])
    assert existing.read_bytes() == b"unchanged"


def test_inspection_requires_six_distinct_tree_meshes_and_embedded_images(tmp_path):
    result = exporter().inspect_glb(write_glb(tmp_path / "fixture.glb", gltf_fixture()))
    assert result["node_count"] == result["mesh_count"] == 6
    assert result["embedded_image_count"] == 1
    assert not result["has_external_asset_uris"]
    assert not result["has_cameras"]
    assert not result["has_lights"]


@pytest.mark.parametrize(
    "change",
    [
        lambda data: data["buffers"][0].update(uri="external.bin"),
        lambda data: data.update(images=[]),
        lambda data: data["images"][0].update(uri="texture.png"),
        lambda data: data.update(cameras=[{}]),
        lambda data: data.update(extensions={"KHR_lights_punctual": {}}),
        lambda data: data["nodes"][0].update(camera=0),
        lambda data: data["nodes"].pop(),
        lambda data: data["nodes"][1].update(mesh=0),
        lambda data: data["nodes"].append({"name": "Robot", "mesh": 6}),
    ],
)
def test_inspector_rejects_external_or_disallowed_scene_content(tmp_path, change):
    data = gltf_fixture()
    change(data)
    with pytest.raises(ValueError):
        exporter().inspect_glb(write_glb(tmp_path / "fixture.glb", data))


@pytest.mark.parametrize("contents", [b"", b"glTF", struct.pack("<4sII", b"glTF", 1, 12)])
def test_inspector_rejects_truncated_or_old_glb(tmp_path, contents):
    path = tmp_path / "fixture.glb"
    path.write_bytes(contents)
    with pytest.raises(ValueError):
        exporter().inspect_glb(path)


def test_actual_local_glb_matches_manifest_units_original_tree_separation_and_hashes():
    folder = Path(__file__).resolve().parents[1] / "artifacts/studio-orchard"
    if not (folder / "manifest.json").is_file():
        pytest.skip("Optional original Blender browser export is not installed")
    module = exporter()
    manifest = json.loads((folder / "manifest.json").read_text())
    inspection = module.inspect_glb(folder / "orchard.glb")
    assert inspection == manifest["glb"]
    assert manifest["units"] == "meters"
    assert manifest["up_axis"] == "Y"
    assert manifest["source_scene_units"]["scale_length"] == 1.0
    assert manifest["source_unchanged_after_export"] is True
    assert manifest["tree_count"] == 2
    assert "local-only" in manifest["distribution"]
    source = {obj["name"]: obj for obj in json.loads((folder / "inventory.json").read_text())}
    rebased = {obj["name"]: obj for obj in manifest["rebased_objects"]}
    origin = np.asarray(manifest["source_origin_m"])
    for name, obj in rebased.items():
        before = np.asarray(source[name]["matrix_world"])
        after = np.asarray(obj["matrix_world"])
        assert after[:3, :3] == pytest.approx(before[:3, :3], abs=1e-6)
        assert after[:3, 3] == pytest.approx(before[:3, 3] - origin, abs=1e-6)
        assert obj["bounds_min_m"] == pytest.approx(np.asarray(source[name]["bounds_min_m"]) - origin, abs=2e-5)
        assert obj["bounds_max_m"] == pytest.approx(np.asarray(source[name]["bounds_max_m"]) - origin, abs=2e-5)
    for node in inspection["tree_nodes"]:
        position = np.asarray(rebased[node["name"]]["matrix_world"])[:3, 3]
        # Blender glTF exporter: source Z-up (x,y,z) -> standard Y-up (x,z,-y).
        assert node.get("translation", [0, 0, 0]) == pytest.approx(position[[0, 2, 1]] * [1, 1, -1], abs=1e-6)
    separation = np.asarray(rebased["tree1_TRUNK"]["matrix_world"])[:3, 3]
    assert np.linalg.norm(separation) > 0.1
