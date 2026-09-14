"""CPU-only contract tests; Blender's actual export is separately recorded."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


def exporter():
    path = Path(__file__).resolve().parents[1] / "tools/export_blender_orchard.py"
    spec = importlib.util.spec_from_file_location("orchard_export", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "name,kind,expected",
    [
        ("post0", "MESH", "post"),
        ("wire0_1", "CURVE", "wire"),
        ("ground", "MESH", "ground"),
        ("sun", "LIGHT", "light"),
        ("tree0_SPUR", "MESH", None),
        ("Camera", "CAMERA", None),
        ("robot", "MESH", None),
    ],
)
def test_export_role_is_explicit(name, kind, expected):
    assert exporter().export_role(name, kind) == expected


def test_texture_relocation_uses_full_relative_path(tmp_path):
    folder = tmp_path / "bark"
    folder.mkdir()
    image = folder / "color.jpg"
    image.write_bytes(b"fixture")
    assert exporter().texture_path("//old/textures/bark/color.jpg", tmp_path) == image


def test_missing_and_escaping_texture_paths_fail(tmp_path):
    with pytest.raises(FileNotFoundError):
        exporter().texture_path("//old/textures/bark/missing.jpg", tmp_path)
    with pytest.raises(ValueError, match="escapes"):
        exporter().texture_path("//old/textures/../../outside.jpg", tmp_path)
    with pytest.raises(ValueError, match="explicit"):
        exporter().texture_path("/unrelated/image.jpg", tmp_path)


def test_connected_components_keep_original_vertex_ids():
    assert exporter().connected_vertices(6, [(1, 2), (3, 4), (2, 0)]) == [[0, 1, 2], [3, 4], [5]]


def test_two_tree_selection_uses_distinct_original_meshes():
    objects = [
        SimpleNamespace(name=f"tree{i}_{part}", type="MESH") for i in range(2) for part in ("TRUNK", "BRANCH", "SPUR")
    ]
    objects.append(SimpleNamespace(name="tree1_camera", type="CAMERA"))
    groups = exporter().source_tree_groups(objects, 2)
    assert [[obj.name for obj in group] for group in groups] == [
        [f"tree{i}_{part}" for part in ("TRUNK", "BRANCH", "SPUR")] for i in range(2)
    ]
    assert len(exporter().source_tree_groups(objects, 1)) == 1
    with pytest.raises(ValueError, match="tree1_TRUNK"):
        exporter().source_tree_groups(objects[:3], 2)


@pytest.mark.parametrize("count", [0, 3, True, 1.0, "2", None])
def test_invalid_export_tree_count_is_rejected(count):
    with pytest.raises(ValueError, match="tree_count"):
        exporter().source_tree_groups([], count)


def test_bad_mesh_edge_is_rejected():
    with pytest.raises(ValueError):
        exporter().connected_vertices(3, [(0, 3)])


def test_existing_export_is_rejected_before_importing_blender(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("existing evidence")
    with pytest.raises(FileExistsError):
        exporter().main(["--template", "missing.blend", "--texture-root", str(tmp_path), "--output-dir", str(tmp_path)])
    assert manifest.read_text() == "existing evidence"


def test_candidate_geometry_is_measured_in_rebased_world_coordinates():
    angle = np.arange(8) * np.pi / 4
    points = np.array([[x, 0.006 * np.cos(a), 0.006 * np.sin(a)] for x in (-0.05, 0.05) for a in angle])
    points += [2, 3, 4]
    obj = SimpleNamespace(
        name="tree0_SPUR",
        matrix_world=np.eye(3),
        data=SimpleNamespace(
            vertices=[SimpleNamespace(co=point) for point in points],
            edges=[SimpleNamespace(vertices=(i, i + 1)) for i in range(15)],
        ),
    )
    result = exporter().branch_candidates(obj, np.array([1, 2, 3]))
    assert result["component_count"] == result["eligible_count"] == 1
    candidate = result["candidates"][0]
    np.testing.assert_allclose(candidate["center_m"], [1, 1, 1])
    np.testing.assert_allclose(candidate["axis"], [1, 0, 0], atol=1e-12)
    assert candidate["length_m"] == pytest.approx(0.1)
    assert candidate["max_radius_m"] == pytest.approx(0.006)
    assert candidate["source_vertex_indices"] == list(range(16))
