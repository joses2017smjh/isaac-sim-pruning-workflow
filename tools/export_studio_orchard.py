#!/usr/bin/env python3
"""Export two original Blender trees and orchard scenery to a LOCAL browser GLB.

Run with ``blender -b --factory-startup --disable-autoexec --python THIS -- ...``.
Source assets remain read-only. The GLB is not licensed for redistribution by this
script; keep it and its manifest under ignored artifacts/, outside public/.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export_blender_orchard import export_role, sha256, source_tree_groups, texture_path

TREE_NAMES = {f"tree{index}_{part}" for index in (0, 1) for part in ("TRUNK", "BRANCH", "SPUR")}


def selected_objects(objects):
    """Require two distinct original trees and exclude every light/camera/robot."""
    trees = [obj for group in source_tree_groups(objects, 2) for obj in group]
    scenery = [obj for obj in objects if export_role(obj.name, obj.type) in {"ground", "post", "wire"}]
    roles = {export_role(obj.name, obj.type) for obj in scenery}
    if roles != {"ground", "post", "wire"}:
        raise ValueError("The orchard must contain original ground, post and wire objects")
    return sorted(trees + scenery, key=lambda obj: obj.name)


def inspect_glb(path: Path) -> dict:
    """Inspect GLB provenance without Blender or a browser; reject external assets."""
    blob = path.read_bytes()
    if len(blob) < 20:
        raise ValueError("Truncated GLB header")
    magic, version, length = struct.unpack_from("<4sII", blob)
    if magic != b"glTF" or version != 2 or length != len(blob):
        raise ValueError("Invalid GLB header or length")
    offset = 12
    chunks = []
    while offset < len(blob):
        if offset + 8 > len(blob):
            raise ValueError("Truncated GLB chunk header")
        size, kind = struct.unpack_from("<I4s", blob, offset)
        offset += 8
        if size % 4 or offset + size > len(blob):
            raise ValueError("Invalid GLB chunk length")
        chunks.append((kind, blob[offset : offset + size]))
        offset += size
    if len(chunks) != 2 or chunks[0][0] != b"JSON" or chunks[1][0] != b"BIN\x00":
        raise ValueError("Expected one JSON and one embedded binary GLB chunk")
    gltf = json.loads(chunks[0][1])
    buffers = gltf.get("buffers", [])
    if len(buffers) != 1 or "uri" in buffers[0] or buffers[0].get("byteLength", 0) > len(chunks[1][1]):
        raise ValueError("GLB must use exactly one embedded buffer")
    images = gltf.get("images", [])
    if not images or any("bufferView" not in item or "uri" in item for item in images):
        raise ValueError("GLB must contain embedded image textures, without external URIs")
    if gltf.get("cameras") or "KHR_lights_punctual" in gltf.get("extensions", {}):
        raise ValueError("Cameras and lights must not be exported")
    nodes = gltf.get("nodes", [])
    named_trees = [node for node in nodes if node.get("name") in TREE_NAMES]
    if len(named_trees) != 6 or {node["name"] for node in named_trees} != TREE_NAMES:
        raise ValueError("GLB must preserve all six distinct original tree object nodes")
    if any("mesh" not in node for node in named_trees) or len({node["mesh"] for node in named_trees}) != 6:
        raise ValueError("Original tree nodes must reference six distinct meshes")
    if any("camera" in node or "KHR_lights_punctual" in node.get("extensions", {}) for node in nodes):
        raise ValueError("A GLB node contains a camera or light")
    allowed = TREE_NAMES | {
        node.get("name") for node in nodes if export_role(node.get("name", ""), "MESH") in {"ground", "post", "wire"}
    }
    if any(node.get("name") not in allowed for node in nodes):
        raise ValueError("GLB contains a node outside the orchard allowlist")
    return {
        "bytes": len(blob),
        "sha256": sha256(path),
        "node_count": len(nodes),
        "mesh_count": len(gltf.get("meshes", [])),
        "material_count": len(gltf.get("materials", [])),
        "embedded_image_count": len(images),
        "image_names": [item.get("name") for item in images],
        "tree_nodes": named_trees,
        "node_names": [node.get("name") for node in nodes],
        "extensions_used": gltf.get("extensionsUsed", []),
        "has_cameras": False,
        "has_lights": False,
        "has_external_asset_uris": False,
    }


def object_inventory(obj, vector_type) -> dict:
    corners = [obj.matrix_world @ vector_type(corner) for corner in obj.bound_box]
    return {
        "name": obj.name,
        "type": obj.type,
        "role": "tree" if obj.name in TREE_NAMES else export_role(obj.name, obj.type),
        "matrix_world": [list(row) for row in obj.matrix_world],
        "bounds_min_m": [min(corner[i] for corner in corners) for i in range(3)],
        "bounds_max_m": [max(corner[i] for corner in corners) for i in range(3)],
        "materials": [slot.material.name if slot.material else None for slot in obj.material_slots],
        "uv_layers": list(obj.data.uv_layers.keys()) if obj.type == "MESH" else [],
        "mesh_vertices": len(obj.data.vertices) if obj.type == "MESH" else None,
        "mesh_polygons": len(obj.data.polygons) if obj.type == "MESH" else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--texture-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing export directory: {args.output_dir}")
    template = args.template.resolve(strict=True)
    source_hash = sha256(template)
    import bpy
    from mathutils import Vector

    bpy.ops.wm.open_mainfile(filepath=str(template), load_ui=False, use_scripts=False)
    if not math.isclose(bpy.context.scene.unit_settings.scale_length, 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("This exporter requires source units of one meter; refusing an implicit unit conversion")
    objects = selected_objects(list(bpy.context.scene.objects))
    origin = bpy.data.objects["tree0_TRUNK"].matrix_world.translation.copy()
    inventory = [object_inventory(obj, Vector) for obj in bpy.context.scene.objects]
    used_materials = {slot.material for obj in objects for slot in obj.material_slots if slot.material is not None}
    repairs = []
    seen_images = set()
    for material in sorted(used_materials, key=lambda item: item.name):
        if not material.node_tree:
            continue
        for node in material.node_tree.nodes:
            image = getattr(node, "image", None)
            if image is None or image.name in seen_images:
                continue
            seen_images.add(image.name)
            relocated = texture_path(image.filepath, args.texture_root)
            image.filepath = str(relocated)
            image.reload()
            repairs.append(
                {
                    "image": image.name,
                    "texture_relative_path": str(relocated.relative_to(args.texture_root.resolve())),
                    "sha256": sha256(relocated),
                }
            )
    # Freeze source world transforms before disconnecting any parent relations.
    matrices = {obj.name: obj.matrix_world.copy() for obj in objects}
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.parent = None
        matrix = matrices[obj.name]
        matrix.translation -= origin
        obj.matrix_world = matrix
        obj.hide_set(False)
        obj.hide_viewport = False
        obj.hide_render = False
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.context.view_layer.update()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output = args.output_dir / "orchard.glb"
    result = bpy.ops.export_scene.gltf(
        filepath=str(output.resolve()),
        export_format="GLB",
        use_selection=True,
        export_yup=True,
        export_apply=True,
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
        export_image_format="AUTO",
        export_animations=False,
        export_cameras=False,
        export_lights=False,
        export_extras=False,
    )
    if "FINISHED" not in result:
        raise RuntimeError(f"GLB export did not finish: {result}")
    inspection = inspect_glb(output)
    if sha256(template) != source_hash:
        raise RuntimeError("Source Blender template changed during export")
    (args.output_dir / "inventory.json").write_text(json.dumps(inventory, indent=2, allow_nan=False) + "\n")
    manifest = {
        "schema_version": 1,
        "distribution": "local-only; original asset redistribution rights have not been established",
        "source_name": template.name,
        "source_sha256": source_hash,
        "source_unchanged_after_export": True,
        "exporter_sha256": sha256(Path(__file__)),
        "blender_version": bpy.app.version_string,
        "units": "meters",
        "up_axis": "Y",
        "source_up_axis": "Z",
        "source_origin_m": list(origin),
        "export_translation_source_z_up_m": list(-origin),
        "source_scene_units": {
            "system": bpy.context.scene.unit_settings.system,
            "scale_length": bpy.context.scene.unit_settings.scale_length,
        },
        "tree_count": 2,
        "tree_layout": "Two distinct original trees; shared translation by negative tree0_TRUNK origin only",
        "viewer_transform_not_baked": {
            "restore_z_up_rotation_x_degrees": 90,
            "scene_yaw_z_degrees": 150,
            "scene_translation_z_up_m": [0.31827075, 1.09701897, -0.08344020],
        },
        "export_options": {"selected_only": True, "apply_modifiers": True, "embedded_images": True},
        "rebased_objects": [object_inventory(obj, Vector) for obj in objects],
        "texture_repairs": repairs,
        "glb": inspection,
        "limitations": [
            "Visualization asset only; no collision, dynamics, sensors or simulated robot control.",
            "No cameras, lights, robot, procedural sky or animation are included.",
            "glTF PBR cannot reproduce arbitrary Cycles procedural shaders or shader displacement.",
            "UVs and exportable image materials are retained; unsupported shader links may be omitted.",
            "The viewer must apply its own lighting; this is not pixel-equivalent to Isaac RTX or Cycles.",
            "Original posts and wires use solid Principled materials, not image textures.",
        ],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"ok": True, "output": str(output), "glb": inspection}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else None))
