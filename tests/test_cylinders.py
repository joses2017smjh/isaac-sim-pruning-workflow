from __future__ import annotations

import json

import numpy as np
import pytest

from isaaclab_pruning.geometry import (
    collision_enabled,
    cylinder_endpoints,
    load_cylinders,
    transform_cylinders,
)


def _metadata() -> dict:
    return {
        "seed_value": 700116,
        "cylinder_data": {
            "(0, 0, 0)": {
                "part_name": "trunk_1",
                "centroid": [0.0, 0.0, 0.5],
                "orientation": [0.0, 0.0, 2.0],
                "radius": 0.05,
                "length": 1.0,
            },
            "(0, 0, 1)": {
                "part_name": "spur_1",
                "centroid": [0.2, 0.0, 0.75],
                "orientation": [1.0, 0.0, 0.0],
                "radius": 0.006,
                "length": 0.1,
            },
        },
    }


def test_metadata_load_normalizes_axes_and_keeps_metric_geometry(tmp_path) -> None:
    metadata_path = tmp_path / "lpy_envy_00000_metadata.json"
    metadata_path.write_text(json.dumps(_metadata()), encoding="utf-8")

    trunk, spur = load_cylinders(metadata_path)
    np.testing.assert_allclose(trunk.orientation, [0.0, 0.0, 1.0])
    np.testing.assert_allclose(cylinder_endpoints(trunk), ([0.0, 0.0, 0.0], [0.0, 0.0, 1.0]))
    assert trunk.organ_class == "trunk"
    assert spur.organ_class == "spur"
    assert collision_enabled(trunk)
    assert not collision_enabled(spur)
    assert collision_enabled(spur, active_cut_point=[0.2, 0.0, 0.75])


def test_rigid_transform_moves_centroid_and_axis(tmp_path) -> None:
    metadata_path = tmp_path / "tree.json"
    metadata_path.write_text(json.dumps(_metadata()), encoding="utf-8")
    cylinders = load_cylinders(metadata_path)
    transform = np.array(
        [
            [1.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 2.0],
            [0.0, 1.0, 0.0, 3.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )

    transformed = transform_cylinders(cylinders, transform)
    np.testing.assert_allclose(transformed[0].centroid, [1.0, 1.5, 3.0])
    np.testing.assert_allclose(transformed[0].orientation, [0.0, -1.0, 0.0])
    assert transformed[0].radius == cylinders[0].radius
    assert transformed[0].length == cylinders[0].length


def test_centroid_only_annotation_is_rejected_as_world_sidecar(tmp_path) -> None:
    metadata_path = tmp_path / "tree.json"
    metadata_path.write_text(json.dumps(_metadata()), encoding="utf-8")
    sidecar_path = tmp_path / "ann.json"
    sidecar_path.write_text(json.dumps([[0.0, 0.0, 0.0]]), encoding="utf-8")

    with pytest.raises(ValueError, match="full cylinder objects"):
        load_cylinders(metadata_path, world_sidecar_path=sidecar_path)


def test_non_rigid_transform_is_rejected(tmp_path) -> None:
    metadata_path = tmp_path / "tree.json"
    metadata_path.write_text(json.dumps(_metadata()), encoding="utf-8")
    transform = np.diag([2.0, 1.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="rigid"):
        transform_cylinders(load_cylinders(metadata_path), transform)
