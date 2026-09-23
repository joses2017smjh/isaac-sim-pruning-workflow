#!/usr/bin/env python3
"""Small paired Blender study using generate_tree2 geometry/material functions.

Run with Blender --background orchard_template.blend --python this.py -- ...
All lights share exactly the same geometry, six camera poses, seed and renderer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Vector


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_exr(path):
    image = bpy.data.images.load(str(path))
    width, height = image.size
    arr = np.empty(width * height * image.channels, dtype=np.float32)
    image.pixels.foreach_get(arr)
    result = arr.reshape(height, width, image.channels)[::-1, :, 0].copy()
    bpy.data.images.remove(image)
    return result


def setup_outputs(scene, folder, stem, write_geometry):
    scene.use_nodes = True
    scene.node_tree.nodes.clear()
    nodes, links = scene.node_tree.nodes, scene.node_tree.links
    layer = nodes.new("CompositorNodeRLayers")
    if write_geometry:
        output = nodes.new("CompositorNodeOutputFile")
        output.base_path = str(folder)
        output.format.file_format = "OPEN_EXR"
        output.format.color_depth = "32"
        output.file_slots[0].path = stem + "_depth_"
        links.new(layer.outputs["Depth"], output.inputs[0])
        mask = nodes.new("CompositorNodeIDMask")
        mask.index = 1
        links.new(layer.outputs["IndexOB"], mask.inputs[0])
        output = nodes.new("CompositorNodeOutputFile")
        output.base_path = str(folder)
        output.format.file_format = "PNG"
        output.format.color_mode = "BW"
        output.file_slots[0].path = stem + "_mask_"
        links.new(mask.outputs[0], output.inputs[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--tree-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--lights", nargs="+", default=["source", "morning", "noon", "evening"])
    parser.add_argument("--one-view", action="store_true")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["COMPUTER_VISION_ROOT"] = str(args.companion)
    sys.path.insert(0, str(args.companion / "Dataloader"))
    import generate_tree2 as gen
    from daylight_presets import apply_preset

    random.seed(1729)
    np.random.seed(1729)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    gen.remove_placeholder_objects()
    gen.fix_ground_material()
    camera = bpy.data.objects[gen.CAM]
    camera.data.lens = 28.0
    camera.data.dof.use_dof = False
    scene.camera = camera
    original = bpy.data.objects[gen.TREE_OBJ.format(0)]
    saved = {0: (original.matrix_world.to_translation().copy(), None)}
    meta_path = args.companion / "trees/metadata" / f"{args.tree_id}_metadata.json"
    texture = next(x for x in gen.BARK_TEXTURES if x["name"] == "bark_brown_02")
    gen.remove_all_tree_objects()
    tree = gen.get_tree_object_for_metadata(args.tree_id, str(meta_path), 0, saved, texture)
    if tree is None:
        raise RuntimeError("Tree failed to load")
    tree.pass_index = 1
    for obj in scene.objects:
        if obj != tree:
            obj.pass_index = 0
    scene.render.engine = "CYCLES"
    gen.set_cycles_gpu(scene, prefer_optix=True)
    scene.cycles.samples = args.samples
    scene.cycles.seed = 1729
    scene.cycles.use_animated_seed = False
    scene.cycles.use_adaptive_sampling = False
    scene.render.resolution_x = args.width
    scene.render.resolution_y = round(args.width * 9 / 16)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    bpy.context.view_layer.use_pass_z = True
    bpy.context.view_layer.use_pass_object_index = True
    # Fixed settings across lights, recorded explicitly rather than inherited silently.
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0
    height = scene.render.resolution_y
    # Read actual Blender projection, avoiding the generator's hard-coded K_REF.
    projection = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=args.width,
        y=height,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    K = [
        [float(projection[0][0]) * args.width / 2, 0, args.width / 2 * (1 - float(projection[0][2])) - 0.5],
        [0, float(projection[1][1]) * height / 2, height / 2 * (1 + float(projection[1][2])) - 0.5],
        [0, 0, 1],
    ]
    bpy.context.view_layer.update()
    metadata = json.loads(meta_path.read_text())
    cylinders = gen.get_cylinder_data_from_metadata(metadata)
    world = gen.transform_cylinders_world(tree, cylinders)
    # Candidate selection is geometric and frozen before any model inference.
    candidates = [
        c for c in world if c["part_name"].startswith("spur") and 0.002 <= c["radius"] <= 0.012 and c["length"] >= 0.008
    ]
    candidates.sort(key=lambda c: (abs(c["centroid"][2] - 0.85), c["part_name"], c["centroid"]))
    target = candidates[0] if candidates else None
    geometry_sha = hashlib.sha256(
        np.asarray([v.co[:] for v in tree.data.vertices], dtype=np.float32).tobytes()
    ).hexdigest()
    poses = []
    for rig, z in enumerate([0.85, 1.0, 1.15]):
        camera.rotation_euler = gen.ROTATION_REF
        bpy.context.view_layer.update()
        right = camera.matrix_world.to_3x3().col[0].normalized()
        for side, dx in [("l", -0.12), ("r", 0.12)]:
            poses.append(
                {
                    "id": f"rig{rig}_{side}",
                    "location": list(Vector((gen.X_REF, gen.Y_REF, z)) + dx * right),
                    "rotation_euler": list(gen.ROTATION_REF),
                }
            )
    if args.one_view:
        poses = poses[:1]
    shared = args.output / "geometry"
    shared.mkdir()
    result = {
        "schema_version": 1,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "tree_id": args.tree_id,
        "family": args.tree_id.split("_")[1],
        "source_metadata_sha256": digest(meta_path),
        "source_generator_sha256": digest(args.companion / "Dataloader/generate_tree2.py"),
        "source_blend": bpy.data.filepath,
        "source_blend_sha256": digest(bpy.data.filepath),
        "geometry_sha256": geometry_sha,
        "tree_transform": [list(row) for row in tree.matrix_world],
        "target": target,
        "target_selection": (
            "Geometric spur radius 2-12 mm, segment length >=8 mm; closest world height to .85m, tie part  "
            "name/centroid. No visibility/error-based substitution."
        ),
        "renderer": {
            "engine": "CYCLES",
            "samples": args.samples,
            "seed": 1729,
            "width": args.width,
            "height": height,
            "view_transform": scene.view_settings.view_transform,
            "look": scene.view_settings.look,
            "exposure": 0.0,
            "gamma": 1.0,
        },
        "K": K,
        "poses": poses,
        "lighting": {},
        "frames": [],
        "depth_convention": "Blender Cycles Z pass, subject to plane sanity validation",
        "ok": False,
    }
    try:
        for li, light in enumerate(args.lights):
            result["lighting"][light] = apply_preset(scene, light, 1729)
            folder = args.output / light
            folder.mkdir()
            for index, pose in enumerate(poses):
                camera.location = pose["location"]
                camera.rotation_euler = pose["rotation_euler"]
                bpy.context.view_layer.update()
                stem = pose["id"]
                scene.frame_set(index + 1)
                setup_outputs(scene, shared, stem, li == 0)
                rgb = folder / f"{stem}.png"
                scene.render.filepath = str(rgb)
                bpy.ops.render.render(write_still=True)
                depth = shared / f"{stem}.npy"
                mask = shared / f"{stem}_mask.png"
                if li == 0:
                    exr = next(shared.glob(f"{stem}_depth_*.exr"))
                    arr = load_exr(exr)
                    np.save(depth, arr)
                    exr.unlink()
                    next(shared.glob(f"{stem}_mask_*.png")).rename(mask)
                target_pixel = None
                if target is not None:
                    projected = world_to_camera_view(scene, camera, Vector(target["centroid"]))
                    target_pixel = [float(projected.x * args.width) - 0.5, float((1 - projected.y) * height) - 0.5]
                target_visible = False
                if target_pixel is not None:
                    x, y = [int(round(v)) for v in target_pixel]
                    optical_z = -(camera.matrix_world.inverted() @ Vector(target["centroid"])).z
                    gt = np.load(depth)
                    if 0 <= x < args.width and 0 <= y < height:
                        target_visible = bool(abs(float(gt[y, x]) - optical_z) <= 2 * target["radius"] + 0.01)
                result["frames"].append(
                    {
                        "rgb": str(rgb),
                        "depth": str(depth),
                        "mask": str(mask),
                        "tree_id": args.tree_id,
                        "family": result["family"],
                        "lighting": light,
                        "view_id": stem,
                        "rig": index // 2,
                        "projected_target_pixel_xy": target_pixel,
                        "target_pixel_xy": target_pixel if target_visible else None,
                        "target_visible": target_visible,
                        "pixel_coordinate_convention": "zero-based pixel centers; Blender edge coordinates minus 0.5",
                        "target_expected_optical_z_m": float(optical_z) if target is not None else None,
                        "target_metric_scope": (
                            "Metadata spur-center projection; visibility check required before interpreting it as  "
                            "that branch surface."
                        ),
                        "camera_location": pose["location"],
                        "camera_rotation_euler": pose["rotation_euler"],
                        "K": K,
                        "geometry_sha256": geometry_sha,
                    }
                )
        result["ok"] = True
    finally:
        (args.output / "render_manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
