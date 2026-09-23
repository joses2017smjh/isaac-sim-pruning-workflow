#!/usr/bin/env python3
"""Enumerate and pre-register pruning targets from the original orchard meshes.

Every connected spur component on each requested tree is measured with the same
geometry helper the simulator uses, screened by the existing jaw-fit rule, and
then sampled with a fixed seed. No target is chosen by hand, and the rejected
components are written out alongside the accepted ones so the screen is auditable.

Reading the mesh needs USD, so ``pxr`` is imported lazily: the partitioning,
screening and selection logic stays importable and testable without it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "source" / "isaaclab_pruning"))

from isaaclab_pruning.sim.blender_component import component_geometry  # noqa: E402

SCHEMA_VERSION = 1

# The demo jaw geometry, as enforced by blender_demo_scene.select_component.
DEFAULT_MAX_RADIUS_M = 0.012

SCOPE = (
    "Pre-registered target population for a scripted vision-guided pruning sweep. "
    "Components come from the original two-tree Blender export; geometry is measured "
    "from mesh vertices, not learned. Screening is the existing jaw-fit rule only; "
    "visibility is decided later, inside each GPU trial."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def connected_components(vertex_count: int, counts, indices) -> list[list[int]]:
    """Partition a polygon mesh into connected vertex components, by smallest ID."""
    counts, indices = [int(value) for value in counts], [int(value) for value in indices]
    if vertex_count < 1 or not counts or sum(counts) != len(indices):
        raise ValueError("Expected complete polygon topology")
    if any(count < 3 for count in counts) or any(not 0 <= index < vertex_count for index in indices):
        raise ValueError("Mesh topology references an invalid vertex")

    parent = list(range(vertex_count))

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    offset = 0
    for count in counts:
        face = indices[offset : offset + count]
        offset += count
        root = find(face[0])
        for vertex in face[1:]:
            other = find(vertex)
            if other != root:
                parent[other] = root

    groups: dict[int, list[int]] = {}
    for vertex in range(vertex_count):
        if parent[vertex] == vertex and find(vertex) == vertex:
            groups.setdefault(vertex, [])
    for vertex in range(vertex_count):
        groups.setdefault(find(vertex), []).append(vertex)
    return [sorted(members) for members in groups.values() if len(members) >= 6]


def screen_candidates(candidates: list[dict], max_radius_m: float) -> tuple[list[dict], list[dict]]:
    """Apply the jaw-fit rule. Rejections are returned, never silently discarded."""
    accepted, rejected = [], []
    for candidate in candidates:
        radius = float(candidate["max_radius_m"])
        if 0 < radius <= max_radius_m:
            accepted.append(candidate)
        else:
            rejected.append({**candidate, "rejected_because": "max_radius_m outside the demo jaw geometry"})
    return accepted, rejected


def select_targets(accepted_by_tree: dict[int, list[dict]], per_tree: int, seed: int) -> list[dict]:
    """Sample a fixed number per tree with one seeded generator, order fixed by vertex ID."""
    selected = []
    for tree_index in sorted(accepted_by_tree):
        pool = sorted(accepted_by_tree[tree_index], key=lambda item: item["component_first_vertex"])
        if len(pool) < per_tree:
            raise ValueError(f"tree{tree_index} has {len(pool)} screened spurs, fewer than the requested {per_tree}")
        generator = random.Random(f"{seed}:tree{tree_index}")
        chosen = generator.sample(pool, per_tree)
        for item in sorted(chosen, key=lambda entry: entry["component_first_vertex"]):
            selected.append(
                {
                    "target_tree_index": tree_index,
                    "component_first_vertex": item["component_first_vertex"],
                    "max_radius_m": item["max_radius_m"],
                    "length_m": item["length_m"],
                    "axis": item["axis"],
                    "source_center_m": item["center_m"],
                }
            )
    return selected


def enumerate_tree(export_dir: Path, tree_index: int) -> list[dict]:
    """Measure every connected component of one tree's spur mesh. Requires USD."""
    from pxr import Gf, Usd, UsdGeom

    object_name = f"tree{tree_index}_SPUR"
    path = export_dir / f"tree{tree_index}.usdc"
    stage = Usd.Stage.Open(str(path))
    matches = [
        prim for prim in stage.Traverse() if prim.IsA(UsdGeom.Mesh) and prim.GetParent().GetName() == object_name
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {object_name} mesh, found {len(matches)}")

    mesh = UsdGeom.Mesh(matches[0])
    source_points = mesh.GetPointsAttr().Get()
    transform = UsdGeom.XformCache().GetLocalToWorldTransform(matches[0])
    points = [tuple(transform.Transform(Gf.Vec3d(point))) for point in source_points]
    components = connected_components(
        len(source_points), mesh.GetFaceVertexCountsAttr().Get(), mesh.GetFaceVertexIndicesAttr().Get()
    )

    measured = []
    for members in components:
        try:
            geometry = component_geometry(points, members)
        except ValueError:
            continue
        geometry["object_name"] = object_name
        geometry.pop("source_vertex_indices", None)
        measured.append(geometry)
    return measured


def verify_export(export_dir: Path, tree_indexes) -> dict:
    """Refuse to enumerate an export whose meshes do not match its own manifest."""
    manifest = json.loads((export_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("units") != "meters" or manifest.get("up_axis") != "Z":
        raise ValueError("Export must explicitly use meters and Z-up")
    recorded = {item["path"]: item["sha256"] for item in manifest["artifacts"]}
    digests = {}
    for tree_index in tree_indexes:
        name = f"tree{tree_index}.usdc"
        digest = sha256(export_dir / name)
        if recorded.get(name) != digest:
            raise ValueError(f"Export provenance mismatch: {name}")
        digests[name] = digest
    return digests


def build(export_dir: Path, trees, per_tree: int, seed: int, max_radius_m: float) -> dict:
    digests = verify_export(export_dir, trees)
    populations, accepted_by_tree, rejected_all = {}, {}, {}
    for tree_index in trees:
        measured = enumerate_tree(export_dir, tree_index)
        accepted, rejected = screen_candidates(measured, max_radius_m)
        accepted_by_tree[tree_index] = accepted
        rejected_all[f"tree{tree_index}"] = [item["component_first_vertex"] for item in rejected]
        populations[f"tree{tree_index}"] = {
            "measured_components": len(measured),
            "screened_in": len(accepted),
            "screened_out": len(rejected),
        }

    targets = select_targets(accepted_by_tree, per_tree, seed)
    return {
        "schema_version": SCHEMA_VERSION,
        "scope": SCOPE,
        "export_dir": str(export_dir),
        "export_sha256": digests,
        "enumerator_sha256": sha256(Path(__file__).resolve()),
        "selection_rule": (
            "Measure every connected component of each tree's SPUR mesh; keep those with "
            f"0 < max_radius_m <= {max_radius_m}; sample {per_tree} per tree with seed "
            f"'{seed}:tree<N>' over the pool ordered by component_first_vertex."
        ),
        "seed": seed,
        "targets_per_tree": per_tree,
        "max_radius_m": max_radius_m,
        "populations": populations,
        "screened_out_component_first_vertexes": rejected_all,
        "target_count": len(targets),
        "targets": targets,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-dir", required=True, type=Path, help="Two-tree Blender export directory")
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    parser.add_argument("--trees", default="0,1", help="Comma-separated tree indexes to enumerate")
    parser.add_argument("--per-tree", type=int, default=10, help="Targets sampled from each tree")
    parser.add_argument("--seed", type=int, default=20260923, help="Fixed selection seed")
    parser.add_argument("--max-radius-m", type=float, default=DEFAULT_MAX_RADIUS_M, help="Jaw-fit screen")
    args = parser.parse_args(argv)

    trees = [int(value) for value in args.trees.split(",") if value.strip() != ""]
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {args.output}")
        document = build(args.export_dir, trees, args.per_tree, args.seed, args.max_radius_m)
        serialized = json.dumps(document, indent=2, allow_nan=False) + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(serialized)
    except (OSError, ValueError, TypeError, KeyError) as error:
        parser.error(str(error))
    else:
        print(serialized, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
