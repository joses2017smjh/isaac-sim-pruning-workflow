"""Resolve a named source component from the original hashed Blender USD mesh.

Selection is known scene topology, not a vision classifier or a cut label. USD
imports are lazy so geometry validation remains usable without Isaac or Blender.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def component_vertex_indices(vertex_count, counts, indices, first_vertex):
    """Find one complete component, requiring its explicit smallest vertex ID."""
    counts = tuple(int(value) for value in counts)
    indices = tuple(int(value) for value in indices)
    first_vertex = int(first_vertex)
    if vertex_count < 1 or not 0 <= first_vertex < vertex_count:
        raise ValueError("Source component vertex is outside the mesh")
    if not counts or any(count < 3 for count in counts) or sum(counts) != len(indices):
        raise ValueError("Expected complete polygon topology")
    if any(index < 0 or index >= vertex_count for index in indices):
        raise ValueError("Mesh topology references an invalid vertex")
    adjacency = [set() for _ in range(vertex_count)]
    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        for left, right in zip(face, face[1:] + face[:1], strict=True):
            adjacency[left].add(right)
            adjacency[right].add(left)
    if not adjacency[first_vertex]:
        raise ValueError("Source component vertex is not used by any polygon")
    found, pending = set(), [first_vertex]
    while pending:
        index = pending.pop()
        if index not in found:
            found.add(index)
            pending.extend(adjacency[index] - found)
    result = sorted(found)
    if result[0] != first_vertex:
        raise ValueError(f"Expected component_first_vertex={result[0]}, not {first_vertex}")
    return result


def component_geometry(points, vertex_indices):
    """Measure PCA centerline/radius while preserving original vertex IDs."""
    points = np.asarray(points, dtype=float)
    indices = list(vertex_indices)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError("Source vertices must be finite Nx3 positions")
    if len(indices) < 6 or indices != sorted(set(indices)) or not 0 <= indices[0] <= indices[-1] < len(points):
        raise ValueError("Expected at least six sorted, unique source vertex IDs")
    sample = points[indices]
    center = sample.mean(axis=0)
    _, _, axes = np.linalg.svd(sample - center, full_matrices=False)
    axis = axes[0]
    if axis[np.argmax(np.abs(axis))] < 0:
        axis *= -1
    projections = (sample - center) @ axis
    length = float(np.ptp(projections))
    radius = float(np.max(np.linalg.norm(sample - center - projections[:, None] * axis, axis=1)))
    if length <= 1e-10 or radius <= 1e-10:
        raise ValueError("Source component has degenerate length or radius")
    return {
        "object_name": "tree0_SPUR",
        "component_first_vertex": indices[0],
        "source_vertex_indices": indices,
        "center_m": center.tolist(),
        "axis": axis.tolist(),
        "length_m": length,
        "max_radius_m": radius,
    }


def component_from_export(export_dir, first_vertex):
    """Resolve a component absent from the export's ten-candidate preview list.

    The full tree stays unchanged. The caller can split only the returned source
    IDs after choosing an orchard placement and validating collision clearance.
    """
    export_dir = Path(export_dir).resolve()
    manifest = json.loads((export_dir / "manifest.json").read_text())
    if manifest.get("units") != "meters" or manifest.get("up_axis") != "Z":
        raise ValueError("Export must explicitly use meters and Z-up")
    expected = [item["sha256"] for item in manifest["artifacts"] if item["path"] == "tree0.usdc"]
    path = export_dir / "tree0.usdc"
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    if len(expected) != 1 or digest.hexdigest() != expected[0]:
        raise ValueError("Export provenance mismatch: tree0.usdc")

    from pxr import Gf, Usd, UsdGeom

    stage = Usd.Stage.Open(str(path))
    matches = [
        prim
        for prim in stage.Traverse()
        if prim.IsA(UsdGeom.Mesh) and str(prim.GetPath()).endswith("/tree0_SPUR/Shape_IndexedFaceSet")
    ]
    if len(matches) != 1:
        raise ValueError("Expected exactly one original tree0_SPUR mesh")
    mesh = UsdGeom.Mesh(matches[0])
    source_points = mesh.GetPointsAttr().Get()
    component = component_vertex_indices(
        len(source_points), mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get(), first_vertex
    )
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(matches[0])
    points = np.asarray([transform.Transform(Gf.Vec3d(point)) for point in source_points])
    return component_geometry(points, component)
