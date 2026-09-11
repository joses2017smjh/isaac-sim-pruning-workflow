#!/usr/bin/env python3
"""Export an existing Blender orchard environment without rendering or executing it.

Run with ``blender -b --factory-startup --disable-autoexec --python THIS -- ...``.
The source .blend is read-only; all generated files stay in --output-dir.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_role(name: str, object_type: str) -> str | None:
    """Explicit static scenery allowlist; never include cameras, robots or trees."""
    label = name.lower()
    if object_type == "LIGHT":
        return "light"
    if object_type not in {"MESH", "CURVE"}:
        return None
    if label.startswith("post"):
        return "post"
    if label.startswith("wire"):
        return "wire"
    if "ground" in label:
        return "ground"
    return None


def texture_path(original: str, texture_root: Path) -> Path:
    """Relocate a Blender texture without allowing paths outside the given root."""
    marker = "/textures/"
    portable = original.replace("\\", "/")
    if marker not in portable:
        raise ValueError(f"Image has no explicit textures directory: {original}")
    relative = Path(portable.split(marker, 1)[1])
    root = texture_root.resolve()
    candidate = (root / relative).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError(f"Texture escapes texture root: {original}")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def connected_vertices(vertex_count: int, edges) -> list[list[int]]:
    """Deterministic mesh components, preserving source vertex identifiers."""
    parents = list(range(vertex_count))

    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for first, second in edges:
        if not 0 <= first < vertex_count or not 0 <= second < vertex_count:
            raise ValueError("Mesh edge index outside vertex array")
        parents[root(second)] = root(first)
    components = {}
    for index in range(vertex_count):
        components.setdefault(root(index), []).append(index)
    return sorted(components.values(), key=lambda indices: indices[0])


def branch_candidates(obj, origin, limit: int = 10) -> dict:
    """Measured connected-component geometry, not a perception result or cut label."""
    import numpy as np

    points = np.asarray([list(obj.matrix_world @ vertex.co - origin) for vertex in obj.data.vertices])
    components = connected_vertices(len(points), [tuple(edge.vertices) for edge in obj.data.edges])
    candidates = []
    for indices in components:
        if len(indices) < 6:
            continue
        sample = points[indices]
        center = sample.mean(axis=0)
        _, _, axes = np.linalg.svd(sample - center, full_matrices=False)
        axis = axes[0]
        if axis[np.argmax(np.abs(axis))] < 0:
            axis *= -1
        projections = (sample - center) @ axis
        length = float(np.ptp(projections))
        radius = float(np.max(np.linalg.norm(sample - center - projections[:, None] * axis, axis=1)))
        if not 0.025 <= length <= 0.60 or radius > 0.04:
            continue
        candidates.append(
            {
                "object_name": obj.name,
                "component_first_vertex": indices[0],
                "source_vertex_indices": indices,
                "center_m": center.tolist(),
                "axis": axis.tolist(),
                "axis_endpoints_m": [
                    (center + axis * value).tolist() for value in (projections.min(), projections.max())
                ],
                "length_m": length,
                "max_radius_m": radius,
                "bounds_min_m": sample.min(axis=0).tolist(),
                "bounds_max_m": sample.max(axis=0).tolist(),
            }
        )
    # Prefer lateral extremities: they tend to be less occluded in an orchard row.
    candidates.sort(key=lambda item: (-abs(item["center_m"][0]), item["component_first_vertex"]))
    return {"component_count": len(components), "eligible_count": len(candidates), "candidates": candidates[:limit]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", required=True, type=Path)
    parser.add_argument("--texture-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--inspect-only", action="store_true")
    args = parser.parse_args(argv)
    # Check before even writing the inventory: provenance belongs to its export.
    if not args.inspect_only:
        for filename in ("environment.usdc", "tree0.usdc", "manifest.json"):
            if (args.output_dir / filename).exists():
                raise FileExistsError(args.output_dir / filename)
    import bpy
    from mathutils import Vector

    bpy.ops.wm.open_mainfile(filepath=str(args.template.resolve()), load_ui=False, use_scripts=False)
    inventory = []
    for obj in bpy.context.scene.objects:
        corners = [obj.matrix_world @ Vector(corner) for corner in obj.bound_box]
        item = {
            "name": obj.name,
            "type": obj.type,
            "role": export_role(obj.name, obj.type),
            "location": list(obj.location),
            "rotation_euler": list(obj.rotation_euler),
            "scale": list(obj.scale),
            "bounds_min": [min(corner[i] for corner in corners) for i in range(3)],
            "bounds_max": [max(corner[i] for corner in corners) for i in range(3)],
            "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
            "uv_layers": list(obj.data.uv_layers.keys()) if obj.type == "MESH" else [],
            "hidden_render": obj.hide_render,
            "mesh_counts": {"vertices": len(obj.data.vertices), "polygons": len(obj.data.polygons)}
            if obj.type == "MESH"
            else None,
            "used_materials": sorted(
                {
                    obj.data.materials[poly.material_index].name
                    for poly in obj.data.polygons
                    if poly.material_index < len(obj.data.materials) and obj.data.materials[poly.material_index]
                }
            )
            if obj.type == "MESH"
            else [],
        }
        if obj.type == "LIGHT":
            item["light"] = {"type": obj.data.type, "energy": obj.data.energy, "color": list(obj.data.color)}
        inventory.append(item)
    materials = {}
    for material in bpy.data.materials:
        materials[material.name] = (
            [
                {
                    "type": node.bl_idname,
                    "name": node.name,
                    "image": node.image.filepath if getattr(node, "image", None) else None,
                    "inputs": {
                        socket.name: (
                            list(socket.default_value)
                            if hasattr(socket.default_value, "__len__")
                            else socket.default_value
                        )
                        for socket in node.inputs
                        if hasattr(socket, "default_value")
                        and isinstance(socket.default_value, (float, int, bool, str))
                        or hasattr(getattr(socket, "default_value", None), "__len__")
                    },
                    "links": [
                        {"from": link.from_node.name, "socket": link.from_socket.name, "input": link.to_socket.name}
                        for link in material.node_tree.links
                        if link.to_node == node
                    ],
                }
                for node in material.node_tree.nodes
            ]
            if material.node_tree
            else []
        )
    export_options = list(bpy.ops.wm.usd_export.get_rna_type().properties.keys())
    evidence = {
        "blender_version": bpy.app.version_string,
        "template": str(args.template.resolve()),
        "template_sha256": sha256(args.template),
        "objects": inventory,
        "materials": materials,
        "world_nodes": [
            {
                "type": node.bl_idname,
                "name": node.name,
                "image": node.image.filepath if getattr(node, "image", None) else None,
            }
            for node in bpy.context.scene.world.node_tree.nodes
        ]
        if bpy.context.scene.world and bpy.context.scene.world.node_tree
        else [],
        "usd_export_options": export_options,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_dir / "inventory.json"
    inventory_path.write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps({"inventory": str(inventory_path), "objects": len(inventory)}), flush=True)
    if args.inspect_only:
        return 0

    tree = [obj for obj in bpy.context.scene.objects if obj.type == "MESH" and obj.name.startswith("tree0_")]
    environment = [obj for obj in bpy.context.scene.objects if export_role(obj.name, obj.type)]
    trunk = bpy.data.objects.get("tree0_TRUNK")
    if trunk is None or len(tree) != 3:
        raise ValueError("Expected original tree0_TRUNK, tree0_BRANCH and tree0_SPUR meshes")
    origin = trunk.matrix_world.translation.copy()
    candidates = {obj.name: branch_candidates(obj, origin) for obj in tree if not obj.name.endswith("TRUNK")}
    used_materials = {
        obj.material_slots[polygon.material_index].material
        for obj in tree + environment
        if obj.type == "MESH"
        for polygon in obj.data.polygons
        if polygon.material_index < len(obj.material_slots) and obj.material_slots[polygon.material_index].material
    }
    repairs = []
    seen_images = set()
    for material in used_materials:
        if not material.node_tree:
            continue
        for node in material.node_tree.nodes:
            image = getattr(node, "image", None)
            if image is None or image.name in seen_images:
                continue
            seen_images.add(image.name)
            original = image.filepath
            relocated = texture_path(original, args.texture_root)
            image.filepath = str(relocated)
            image.reload()
            repairs.append(
                {"image": image.name, "original": original, "resolved": str(relocated), "sha256": sha256(relocated)}
            )

    for obj in environment + tree:
        matrix = obj.matrix_world.copy()
        matrix.translation -= origin
        obj.matrix_world = matrix
    bpy.context.view_layer.update()

    def export(objects, filename, root_path):
        bpy.ops.object.select_all(action="DESELECT")
        for obj in objects:
            obj.hide_set(False)
            obj.hide_viewport = False
            obj.hide_render = False
            obj.select_set(True)
        bpy.context.view_layer.objects.active = objects[0]
        result = bpy.ops.wm.usd_export(
            filepath=str((args.output_dir / filename).resolve()),
            selected_objects_only=True,
            export_animation=False,
            export_uvmaps=True,
            export_normals=True,
            export_materials=True,
            export_textures=True,
            overwrite_textures=True,
            relative_paths=True,
            generate_preview_surface=True,
            generate_materialx_network=False,
            convert_world_material=False,
            export_cameras=False,
            root_prim_path=root_path,
            export_custom_properties=False,
        )
        if "FINISHED" not in result:
            raise RuntimeError(f"USD export failed: {result}")
        return {
            "file": filename,
            "root_prim": root_path,
            "objects": [
                {
                    "name": obj.name,
                    "prim_path": f"{root_path}/{obj.name}",
                    "bounds_min_m": [
                        min((obj.matrix_world @ Vector(corner))[i] for corner in obj.bound_box) for i in range(3)
                    ],
                    "bounds_max_m": [
                        max((obj.matrix_world @ Vector(corner))[i] for corner in obj.bound_box) for i in range(3)
                    ],
                }
                for obj in objects
            ],
        }

    exports = [export(environment, "environment.usdc", "/BlenderOrchard"), export(tree, "tree0.usdc", "/BlenderTree")]
    manifest = {
        "schema_version": 1,
        "source_template": str(args.template.resolve()),
        "source_sha256": evidence["template_sha256"],
        "exporter_sha256": sha256(Path(__file__)),
        "blender_version": bpy.app.version_string,
        "units": "meters",
        "up_axis": "Z",
        "source_origin_m": list(origin),
        "export_translation_m": list(-origin),
        "source_scene_units": {
            "system": bpy.context.scene.unit_settings.system,
            "scale_length": bpy.context.scene.unit_settings.scale_length,
        },
        "exports": exports,
        "texture_repairs": repairs,
        "branch_geometry_candidates": candidates,
        "limitations": [
            "Static visual meshes: the importer must author collision and pruning physics.",
            "Blender procedural sky is not converted; the original Sun light is exported.",
            "Preview Surface export does not reproduce arbitrary Cycles shader nodes or displacement.",
            "Candidates use known mesh connected components, not a detector or verified safe cut sites.",
            "Original posts and wires use solid Principled materials, not image textures.",
        ],
        "artifacts": [
            {"path": str(path.relative_to(args.output_dir)), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(args.output_dir.rglob("*"))
            if path.is_file()
        ],
    }
    if sha256(args.template) != evidence["template_sha256"]:
        raise RuntimeError("Source template changed during export")
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(
        json.dumps(
            {"ok": True, "output_dir": str(args.output_dir), "exports": exports, "textures_relocated": len(repairs)}
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None))
