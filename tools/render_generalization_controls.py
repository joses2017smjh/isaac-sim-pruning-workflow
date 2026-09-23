#!/usr/bin/env python3
"""Render the generalization controls for one L-Py tree with Blender Cycles.

Run with Blender --background orchard_template.blend --python this.py -- ...
Same geometry, seed, texture, sampler and tone mapping as the matrix renderer
(``render_family_lighting.py``, whose helpers this reuses). What changes per
condition is exactly one of: the lighting preset, the camera model, or the pose
set, as listed in ``generalization_controls.CONDITIONS``. Depth and mask are
rendered once per (pose set, camera model) and shared by every lighting cell.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
from pathlib import Path

import bpy
import numpy as np
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generalization_controls as gc  # noqa: E402
import render_family_lighting as rfl  # noqa: E402

CLIP_START_M = 0.01


def apply_camera(scene, camera, model, scale):
    spec = gc.CAMERA_MODELS[model]
    camera.data.sensor_fit = spec["sensor_fit"]
    camera.data.sensor_width = spec["sensor_width_mm"]
    camera.data.lens = spec["lens_mm"]
    camera.data.clip_start = CLIP_START_M
    width, height = max(8, round(spec["width"] * scale)), max(8, round(spec["height"] * scale))
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    bpy.context.view_layer.update()
    projection = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=width,
        y=height,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    K = [
        [float(projection[0][0]) * width / 2, 0, width / 2 * (1 - float(projection[0][2])) - 0.5],
        [0, float(projection[1][1]) * height / 2, height / 2 * (1 + float(projection[1][2])) - 0.5],
        [0, 0, 1],
    ]
    return width, height, K


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--tree-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--scale", type=float, default=1.0, help="Resolution scale; below 1 only for smoke renders")
    parser.add_argument("--plan", type=Path, help="Frozen batch plan; its condition list replaces the module default")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ["COMPUTER_VISION_ROOT"] = str(args.companion)
    sys.path.insert(0, str(args.companion / "Dataloader"))
    import generate_tree2 as gen

    conditions = gc.CONDITIONS
    if args.plan is not None:
        conditions = json.loads(args.plan.read_text())["conditions"]

    random.seed(1729)
    np.random.seed(1729)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    gen.remove_placeholder_objects()
    gen.fix_ground_material()
    camera = bpy.data.objects[gen.CAM]
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
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.film_transparent = False
    bpy.context.view_layer.use_pass_z = True
    bpy.context.view_layer.use_pass_object_index = True
    scene.view_settings.view_transform = "Filmic"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0

    metadata = json.loads(meta_path.read_text())
    cylinders = gen.get_cylinder_data_from_metadata(metadata)
    world = gen.transform_cylinders_world(tree, cylinders)
    # Same geometric target rule as the matrix: frozen before any model inference.
    candidates = [
        c for c in world if c["part_name"].startswith("spur") and 0.002 <= c["radius"] <= 0.012 and c["length"] >= 0.008
    ]
    candidates.sort(key=lambda c: (abs(c["centroid"][2] - 0.85), c["part_name"], c["centroid"]))
    target = candidates[0] if candidates else None
    geometry_sha = hashlib.sha256(
        np.asarray([v.co[:] for v in tree.data.vertices], dtype=np.float32).tobytes()
    ).hexdigest()

    camera.rotation_euler = gen.ROTATION_REF
    bpy.context.view_layer.update()
    axes = camera.matrix_world.to_3x3()
    right, up, forward = axes.col[0].normalized(), axes.col[1].normalized(), (-axes.col[2]).normalized()
    pitched = axes @ Matrix.Rotation(math.radians(gc.ISAAC_PITCH_DEG), 3, "X")
    pitched_rotation = list(pitched.to_euler("XYZ"))
    expected = gc.pitched_forward(list(forward), list(up), gc.ISAAC_PITCH_DEG)
    if (Vector(expected) - (-pitched.col[2]).normalized()).length > 1e-6:
        raise RuntimeError("Pitched camera rotation does not match the registered pitch")

    pose_sets = {}
    skipped = []
    for spec_name, spec in gc.POSE_SETS.items():
        if spec["kind"] != "heights" and target is None:
            continue
        pose_sets[spec_name] = gc.poses(
            spec,
            reference_xy=(gen.X_REF, gen.Y_REF),
            rotation_ref=list(gen.ROTATION_REF),
            forward=list(forward),
            right=list(right),
            up=list(up),
            target_centroid=list(target["centroid"]) if target else None,
            pitched_rotation=pitched_rotation,
        )

    result = {
        "schema_version": gc.SCHEMA_VERSION,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "tree_id": args.tree_id,
        "family": args.tree_id.split("_")[1],
        "source_metadata_sha256": rfl.digest(meta_path),
        "source_generator_sha256": rfl.digest(args.companion / "Dataloader/generate_tree2.py"),
        "source_presets_sha256": rfl.digest(args.companion / "Dataloader/daylight_presets.py"),
        "source_blend": bpy.data.filepath,
        "source_blend_sha256": rfl.digest(bpy.data.filepath),
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
            "scale": args.scale,
            "clip_start_m": CLIP_START_M,
            "view_transform": scene.view_settings.view_transform,
            "look": scene.view_settings.look,
            "exposure": 0.0,
            "gamma": 1.0,
        },
        "camera_models": gc.CAMERA_MODELS,
        "pose_sets": {name: {**gc.POSE_SETS[name], "poses": poses} for name, poses in pose_sets.items()},
        "reference_axes": {"forward": list(forward), "right": list(right), "up": list(up)},
        "conditions": conditions,
        "lighting": {},
        "skipped": skipped,
        "frames": [],
        "depth_convention": "Blender Cycles Z pass, subject to plane sanity validation",
        "ok": False,
    }
    geometry_root = args.output / "geometry"
    geometry_root.mkdir()
    rendered_geometry = set()
    frame_counter = 0
    try:
        for cond in conditions:
            if cond["pose_set"] not in pose_sets:
                skipped.append({"condition": cond["id"], "reason": "no_geometric_target"})
                continue
            if cond["lighting"] not in result["lighting"]:
                result["lighting"][cond["lighting"]] = gc.apply_lighting(scene, cond["lighting"], 1729)
            else:
                gc.apply_lighting(scene, cond["lighting"], 1729)
            width, height, K = apply_camera(scene, camera, cond["camera_model"], args.scale)
            key = gc.geometry_key(cond["camera_model"], cond["pose_set"])
            shared = geometry_root / key
            write_geometry = key not in rendered_geometry
            if write_geometry:
                shared.mkdir()
            folder = args.output / "conditions" / cond["id"].replace("/", "__")
            folder.mkdir(parents=True)
            spec = gc.POSE_SETS[cond["pose_set"]]
            for index, pose in enumerate(pose_sets[cond["pose_set"]]):
                camera.location = pose["location"]
                camera.rotation_euler = pose["rotation_euler"]
                bpy.context.view_layer.update()
                stem = pose["id"]
                frame_counter += 1
                scene.frame_set(frame_counter)
                rfl.setup_outputs(scene, shared, stem, write_geometry)
                rgb = folder / f"{stem}.png"
                scene.render.filepath = str(rgb)
                bpy.ops.render.render(write_still=True)
                depth = shared / f"{stem}.npy"
                mask = shared / f"{stem}_mask.png"
                if write_geometry:
                    exr = next(shared.glob(f"{stem}_depth_*.exr"))
                    np.save(depth, rfl.load_exr(exr))
                    exr.unlink()
                    next(shared.glob(f"{stem}_mask_*.png")).rename(mask)
                target_pixel, optical_z, target_visible = None, None, False
                if target is not None:
                    projected = world_to_camera_view(scene, camera, Vector(target["centroid"]))
                    target_pixel = [float(projected.x * width) - 0.5, float((1 - projected.y) * height) - 0.5]
                    optical_z = -(camera.matrix_world.inverted() @ Vector(target["centroid"])).z
                    x, y = [int(round(v)) for v in target_pixel]
                    if 0 <= x < width and 0 <= y < height and optical_z > 0:
                        gt = np.load(depth)
                        target_visible = bool(abs(float(gt[y, x]) - optical_z) <= 2 * target["radius"] + 0.01)
                distance = None
                if spec["kind"] != "heights":
                    distance = spec["distances_m"][index if spec["kind"] == "sweep" else index // 2]
                result["frames"].append(
                    {
                        "rgb": str(rgb),
                        "depth": str(depth),
                        "mask": str(mask),
                        "tree_id": args.tree_id,
                        "family": result["family"],
                        "condition": cond["id"],
                        "condition_group": cond["group"],
                        "lighting": cond["lighting"],
                        "camera_model": cond["camera_model"],
                        "pose_set": cond["pose_set"],
                        "geometry_key": key,
                        "dino_rig_compatible": gc.dino_rig_compatible(cond["camera_model"], cond["pose_set"]),
                        "view_id": stem,
                        "rig": index // 2 if spec["kind"] != "sweep" else index,
                        "nominal_distance_m": distance,
                        "pitch_deg": spec.get("pitch_deg", 0.0),
                        "width": width,
                        "height": height,
                        "projected_target_pixel_xy": target_pixel,
                        "target_pixel_xy": target_pixel if target_visible else None,
                        "target_visible": target_visible,
                        "pixel_coordinate_convention": "zero-based pixel centers; Blender edge coordinates minus 0.5",
                        "target_expected_optical_z_m": float(optical_z) if optical_z is not None else None,
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
            rendered_geometry.add(key)
        result["ok"] = True
    finally:
        (args.output / "render_manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
