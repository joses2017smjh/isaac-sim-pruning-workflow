from __future__ import annotations

import json
import math

import numpy as np

from isaaclab_pruning.eval.blender_trunk import score_blender_trunk
from isaaclab_pruning.geometry import Cylinder, load_cylinders, transform_cylinders
from isaaclab_pruning.usd.ascii_tree import quat_wxyz_align_z, write_cylinder_tree_usda
from isaaclab_pruning.usd.cylinders import DEFAULT_TREE_TILT_X_DEG, _local_transform
from isaaclab_pruning.usd.orchard import write_orchard_usda


def _sample_cylinders() -> list[Cylinder]:
    return [
        Cylinder(
            record_id="(0, 0, 0)",
            part_name="trunk_1",
            centroid=np.array([0.0, 0.0, 0.5]),
            orientation=np.array([0.0, 0.0, 1.0]),
            radius=0.05,
            length=1.0,
        ),
        Cylinder(
            record_id="(0, 0, 1)",
            part_name="spur_1",
            centroid=np.array([0.2, 0.0, 0.75]),
            orientation=np.array([1.0, 0.0, 0.0]),
            radius=0.006,
            length=0.1,
        ),
    ]


def test_ascii_usda_binds_bark_preview_surface(tmp_path) -> None:
    output = tmp_path / "tree.usda"
    summary = write_cylinder_tree_usda(_sample_cylinders(), output, tree_id="lpy_envy_00000")
    text = output.read_text(encoding="utf-8")
    assert summary["cylinders"] == 2
    assert summary["material"] == "bark_brown_02"
    assert 'upAxis = "Z"' in text
    assert "metersPerUnit = 1" in text
    assert 'uniform token info:id = "UsdPreviewSurface"' in text
    assert "bark_brown_02" in text
    assert "rel material:binding" in text
    assert "PhysicsCollisionAPI" in text
    assert 'custom string pruning:recordId = "(0, 0, 0)"' in text


def test_quat_align_z_identity_and_x_axis() -> None:
    np.testing.assert_allclose(quat_wxyz_align_z(np.array([0.0, 0.0, 1.0])), (1.0, 0.0, 0.0, 0.0), atol=1e-12)
    w, x, y, z = quat_wxyz_align_z(np.array([1.0, 0.0, 0.0]))
    assert abs(w - math.sqrt(0.5)) < 1e-9
    assert abs(y - math.sqrt(0.5)) < 1e-9
    assert abs(x) < 1e-9 and abs(z) < 1e-9


def test_blender_trunk_median_under_two_mm() -> None:
    cylinders = _sample_cylinders()
    tilted = transform_cylinders(cylinders, _local_transform(DEFAULT_TREE_TILT_X_DEG, (0.0, 0.0, 0.0)))
    translation = np.array([-9.96, -9.71, 0.0])
    blender = np.stack([record.centroid + translation for record in tilted])
    report = score_blender_trunk(cylinders, blender)
    assert report["pass"]
    assert report["n_trunk"] == 1
    assert report["median_trunk_error_m"] < 1e-12


def test_orchard_usda_has_preview_surface_not_only_a_string(tmp_path) -> None:
    path = write_orchard_usda(tmp_path / "orchard.usda")
    text = path.read_text(encoding="utf-8")
    assert "bark_brown_02" in text
    assert 'uniform token info:id = "UsdPreviewSurface"' in text
    assert "def Material" in text


def test_tertiarybranch_maps_to_branch_collision(tmp_path) -> None:
    metadata = {
        "cylinder_data": {
            "0": {
                "part_name": "tertiarybranch_1",
                "centroid": [0.0, 0.0, 1.0],
                "orientation": [0.0, 0.0, 1.0],
                "radius": 0.01,
                "length": 0.2,
            }
        }
    }
    path = tmp_path / "ufo.json"
    path.write_text(json.dumps(metadata), encoding="utf-8")
    (record,) = load_cylinders(path)
    assert record.organ_class == "branch"


def test_curobo_bounding_sphere_covers_vertices() -> None:
    from isaaclab_pruning.baselines.curobo import bounding_sphere, ur5e_pruner_oracle_status

    vertices = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.5, 0.0]])
    center, radius = bounding_sphere(vertices)
    assert abs(center[0]) < 1e-9
    assert radius >= 1.0 - 1e-9
    assert not ur5e_pruner_oracle_status(urdf_usd_path="/no/such.usd").configured
