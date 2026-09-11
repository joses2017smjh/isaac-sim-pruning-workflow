"""Source-component resolution without a Blender runtime."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from isaaclab_pruning.sim.blender_component import (
    component_from_export,
    component_geometry,
    component_vertex_indices,
)


def test_topology_resolves_only_named_connected_component():
    assert component_vertex_indices(8, [3, 3, 3], [0, 1, 2, 2, 1, 3, 4, 5, 6], 0) == [0, 1, 2, 3]
    assert component_vertex_indices(8, [3, 3, 3], [0, 1, 2, 2, 1, 3, 4, 5, 6], 4) == [4, 5, 6]


@pytest.mark.parametrize(
    "counts,indices,first",
    [
        ([3], [0, 1], 0),
        ([2], [0, 1], 0),
        ([3], [0, 1, 8], 0),
        ([3], [0, 1, 2], 1),
        ([3], [0, 1, 2], 7),
        ([3], [0, 1, 2], -1),
    ],
)
def test_topology_rejects_invalid_or_noncanonical_selection(counts, indices, first):
    with pytest.raises(ValueError):
        component_vertex_indices(8, counts, indices, first)


def test_geometry_preserves_source_ids_and_measures_axis_length_radius():
    points = np.asarray([[x, y, z] for x in (-0.05, 0.05) for y in (-0.003, 0.003) for z in (-0.004, 0.004)])
    points += [1, 2, 3]
    result = component_geometry(points, list(range(8)))
    assert result["center_m"] == pytest.approx([1, 2, 3])
    assert result["axis"] == pytest.approx([1, 0, 0])
    assert result["length_m"] == pytest.approx(0.1)
    assert result["max_radius_m"] == pytest.approx(0.005)
    assert result["source_vertex_indices"] == list(range(8))


@pytest.mark.parametrize(
    "points,indices",
    [
        (np.zeros((6, 3)), list(range(6))),
        (np.full((6, 3), np.nan), list(range(6))),
        (np.zeros((6, 2)), list(range(6))),
        (np.zeros((6, 3)), [0, 1, 2]),
        (np.zeros((6, 3)), [0, 1, 2, 3, 4, 7]),
        (np.zeros((6, 3)), [0, 1, 2, 3, 4, 4]),
    ],
)
def test_geometry_rejects_degenerate_data(points, indices):
    with pytest.raises(ValueError):
        component_geometry(points, indices)


def test_export_hash_checked_before_loading_usd(tmp_path):
    (tmp_path / "tree0.usdc").write_bytes(b"changed source")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "units": "meters",
                "up_axis": "Z",
                "artifacts": [{"path": "tree0.usdc", "sha256": hashlib.sha256(b"original source").hexdigest()}],
            }
        )
    )
    with pytest.raises(ValueError, match="provenance mismatch"):
        component_from_export(tmp_path, 8235)


def test_original_export_component_8235_matches_full_mesh_geometry():
    pytest.importorskip("pxr.Usd")
    export_dir = Path(__file__).resolve().parents[1] / "artifacts/blender_scene/orchard_v1"
    if not (export_dir / "manifest.json").is_file():
        pytest.skip("Optional original Blender orchard export is not installed")
    result = component_from_export(export_dir, 8235)
    assert result["component_first_vertex"] == 8235
    assert result["source_vertex_indices"] == list(range(8235, 8295))
    assert result["center_m"] == pytest.approx([-0.1733916673, 0.4343857377, 0.8986173171], abs=1e-8)
    assert result["axis"] == pytest.approx([-0.0725070438, 0.4663907755, 0.8816021626], abs=1e-8)
    assert result["max_radius_m"] == pytest.approx(0.0067835999828273555)
    assert result["length_m"] == pytest.approx(0.04992177715716928)
