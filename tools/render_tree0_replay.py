#!/usr/bin/env python3
"""Render the original orchard tree0 in Blender Cycles at the recorded Isaac wrist poses.

Run with Blender --background orchard_template.blend --python this.py -- ...
The renderer control for the Isaac Stage A failure: the same tree the Isaac
scene was exported from, seen by the same camera model from the very poses the
Stage A wrist recorded, rendered by the training renderer. The Isaac-to-Blender
chain (export translation, 150 degree yaw, ROS optical basis to Blender camera)
was verified on September 23 to 1e-6 m against the recorded RTX depth.

Two bark textures per pose: the palm bark the export carried (what Isaac
rendered) and bark_brown_02 (what every training render used). Depth and the
tree0 mask are shared by the two. The robot, tool jaws and Isaac dome light are
not in the template and are not reproduced; lighting uses the companion preset
named by the recorded run, applied through the same code path as the matrix.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generalization_controls as gc  # noqa: E402
import render_family_lighting as rfl  # noqa: E402

TREE0_OBJECTS = ("tree0_TRUNK", "tree0_BRANCH", "tree0_SPUR")
BARKS = {"palm": "bark_palm_tree", "bark_brown_02": "bark_brown_02"}
WIDTH, HEIGHT = 480, 320


def rz(deg):
    a = np.radians(deg)
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def isaac_wrist_to_blender_camera(position_w_m, rotation_w_ros, translation_w_m, yaw_deg, source_origin_m):
    """Blender camera location and rotation matrix from a recorded Isaac wrist pose."""
    export = rz(-yaw_deg) @ (np.asarray(position_w_m, float) - np.asarray(translation_w_m, float))
    location = export + np.asarray(source_origin_m, float)
    rotation = rz(-yaw_deg) @ np.asarray(rotation_w_ros, float) @ np.diag([1.0, -1.0, -1.0])
    return location, rotation


def relocate_textures(texture_root):
    repairs = []
    for image in bpy.data.images:
        path = image.filepath.replace("\\", "/")
        if "/textures/" in path:
            new = Path(texture_root) / path.split("/textures/", 1)[1]
            if new.is_file():
                image.filepath = str(new)
                image.reload()
                repairs.append({"image": image.name, "path": str(new)})
            else:
                raise FileNotFoundError(f"Template texture missing from the companion: {new}")
    return repairs


def material_slot_for(obj, texture_folder):
    for index, slot in enumerate(obj.material_slots):
        material = slot.material
        if material is None or not material.node_tree:
            continue
        paths = [n.image.filepath for n in material.node_tree.nodes if n.type == "TEX_IMAGE" and n.image]
        if paths and all(f"/{texture_folder}/" in p.replace("\\", "/") for p in paths):
            return index
    raise LookupError(f"No material slot on {obj.name} uses only {texture_folder}")


def assign_bark(bark_key):
    for name in TREE0_OBJECTS:
        obj = bpy.data.objects[name]
        index = material_slot_for(obj, BARKS[bark_key])
        for polygon in obj.data.polygons:
            polygon.material_index = index


def mute_displacement():
    """Isaac's preview surface carried no displacement; match that so geometry is identical."""
    muted = []
    for material in bpy.data.materials:
        if not material.node_tree:
            continue
        for link in list(material.node_tree.links):
            if link.to_node.type == "OUTPUT_MATERIAL" and link.to_socket.name == "Displacement":
                material.node_tree.links.remove(link)
                muted.append(material.name)
    return muted


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True, help="A recorded Stage A run")
    parser.add_argument("--manifest", type=Path, required=True, help="orchard_two_trees_v1/manifest.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--frame-stride", type=int, default=3)
    parser.add_argument("--max-frame", type=int, default=77, help="Last frame before the recorded detachment")
    parser.add_argument("--scale", type=float, default=1.0, help="Resolution scale; below 1 only for smoke renders")
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(args.companion / "Dataloader"))

    manifest = json.loads(args.manifest.read_text())
    source_origin = np.asarray(manifest["source_origin_m"], float)
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    repairs = relocate_textures(args.companion / "textures")
    muted = mute_displacement()
    camera = bpy.data.objects["Camera"]
    camera.data.type = "PERSP"
    camera.data.lens = 20.0
    camera.data.sensor_width = 30.0
    camera.data.sensor_height = 20.0
    camera.data.sensor_fit = "HORIZONTAL"
    camera.data.shift_x = camera.data.shift_y = 0.0
    camera.data.clip_start, camera.data.clip_end = 0.01, 20.0
    camera.data.dof.use_dof = False
    camera.rotation_mode = "XYZ"
    scene.camera = camera
    width, height = max(8, round(WIDTH * args.scale)), max(8, round(HEIGHT * args.scale))
    scene.render.resolution_x, scene.render.resolution_y = width, height
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1.0
    scene.render.engine = "CYCLES"
    try:
        import generate_tree2 as gen

        gen.set_cycles_gpu(scene, prefer_optix=True)
    except Exception as error:  # noqa: BLE001 - CPU fallback is acceptable for a smoke render
        print(f"Cycles GPU setup skipped: {error}")
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
    for obj in scene.objects:
        obj.pass_index = 1 if obj.name in TREE0_OBJECTS else 0
    fx = 20.0 / 30.0 * width
    K = [[fx, 0, width / 2 - 0.5], [0, fx, height / 2 - 0.5], [0, 0, 1]]

    result = {
        "schema_version": 1,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "tree_id": "original_orchard_tree0",
        "family": "original_orchard",
        "source_blend": bpy.data.filepath,
        "source_blend_sha256": rfl.digest(bpy.data.filepath),
        "manifest_sha256": rfl.digest(args.manifest),
        "source_presets_sha256": rfl.digest(args.companion / "Dataloader/daylight_presets.py"),
        "texture_repairs": repairs,
        "displacement_muted_materials": muted,
        "camera_model": {**gc.CAMERA_MODELS["isaac_wrist"], "K": K, "scale": args.scale},
        "mapping": {
            "chain": (
                "p_blender = Rz(-yaw) (p_isaac - translation_w_m) + source_origin_m; "
                "R_cam = Rz(-yaw) R_ros diag(1,-1,-1)"
            ),
            "source_origin_m": source_origin.tolist(),
            "verified": "September 23: recorded RTX depth reproduced to 1e-6 m by ray cast and 4e-6 m by a Cycles tile",
        },
        "barks": BARKS,
        "renderer": {"engine": "CYCLES", "samples": args.samples, "seed": 1729, "view_transform": "Filmic"},
        "runs": [],
        "lighting": {},
        "frames": [],
        "depth_convention": "Blender Cycles Z pass (planar optical Z), the RTX distance_to_image_plane convention",
        "ok": False,
    }
    geometry_root = args.output / "geometry"
    geometry_root.mkdir()
    counter = 0
    try:
        for run_dir in args.run_dir:
            report = json.loads((run_dir / "report.json").read_text())
            frames = json.loads((run_dir / "frames.json").read_text())["frames"]
            scene_info = report["blender_scene"]
            daylight = scene_info["daylight"]["preset"]
            translation = np.asarray(scene_info["translation_w_m"], float)
            yaw = float(scene_info["yaw_degrees"])
            target_isaac = np.asarray(scene_info["target"]["position_w_m"], float)
            target_blender = rz(-yaw) @ (target_isaac - translation) + source_origin
            result["runs"].append(
                {
                    "run_dir": str(run_dir),
                    "report_sha256": rfl.digest(run_dir / "report.json"),
                    "daylight": daylight,
                    "target_id": scene_info["target"]["id"],
                    "target_blender_m": target_blender.tolist(),
                    "frames_selected": list(range(0, args.max_frame + 1, args.frame_stride)),
                }
            )
            for index in range(0, args.max_frame + 1, args.frame_stride):
                frame = frames[index]
                measurement = (frame.get("live_vision") or {}).get("measurement") or {}
                location, rotation = isaac_wrist_to_blender_camera(
                    frame["wrist_position_w_m"], frame["wrist_rotation_w_ros"], translation, yaw, source_origin
                )
                matrix = Matrix.Identity(4)
                for i in range(3):
                    for j in range(3):
                        matrix[i][j] = float(rotation[i, j])
                    matrix[i][3] = float(location[i])
                camera.matrix_world = matrix
                bpy.context.view_layer.update()
                pc = rotation.T @ (target_blender - location)
                optical_z = float(-pc[2])
                projected = [
                    float(fx * pc[0] / optical_z + width / 2 - 0.5),
                    float(fx * -pc[1] / optical_z + height / 2 - 0.5),
                ]
                tracker_pixel = measurement.get("pixel_xy")
                if tracker_pixel is not None:
                    tracker_pixel = [float(v) * args.scale for v in tracker_pixel]
                stem = f"{run_dir.name}_f{index:03d}"
                shared = geometry_root / stem
                shared.mkdir()
                for bark_index, bark in enumerate(BARKS):
                    assign_bark(bark)
                    key = f"{daylight}/{bark}"
                    if key not in result["lighting"]:
                        result["lighting"][key] = gc.apply_lighting(scene, daylight, 1729)
                    else:
                        gc.apply_lighting(scene, daylight, 1729)
                    counter += 1
                    scene.frame_set(counter)
                    rfl.setup_outputs(scene, shared, "z", bark_index == 0)
                    folder = args.output / "conditions" / f"isaac_pose__{bark}"
                    folder.mkdir(parents=True, exist_ok=True)
                    rgb = folder / f"{stem}.png"
                    scene.render.filepath = str(rgb)
                    bpy.ops.render.render(write_still=True)
                    depth, mask = shared / "depth.npy", shared / "mask.png"
                    if bark_index == 0:
                        exr = next(shared.glob("z_depth_*.exr"))
                        np.save(depth, rfl.load_exr(exr))
                        exr.unlink()
                        next(shared.glob("z_mask_*.png")).rename(mask)
                    gt = np.load(depth)
                    visible = False
                    pixel = tracker_pixel if tracker_pixel is not None else projected
                    x, y = int(round(pixel[0])), int(round(pixel[1]))
                    if 0 <= x < width and 0 <= y < height:
                        visible = bool(abs(float(gt[y, x]) - optical_z) <= 0.03)
                    result["frames"].append(
                        {
                            "rgb": str(rgb),
                            "depth": str(depth),
                            "mask": str(mask),
                            "tree_id": "original_orchard_tree0",
                            "family": "original_orchard",
                            "condition": f"isaac_pose/{bark}/{daylight}",
                            "condition_group": "renderer_control",
                            "bark": bark,
                            "lighting": daylight,
                            "camera_model": "isaac_wrist",
                            "pose_set": "isaac_recorded",
                            "dino_rig_compatible": False,
                            "view_id": stem,
                            "isaac_run": run_dir.name,
                            "isaac_frame_index": index,
                            "isaac_rgb": str(run_dir / "frames" / f"wrist_{index:05d}.png"),
                            "isaac_depth": str(run_dir / "frames" / f"depth_{index:05d}.npy"),
                            "width": width,
                            "height": height,
                            "projected_target_pixel_xy": projected,
                            "tracker_pixel_xy": tracker_pixel,
                            "target_pixel_xy": pixel if visible else None,
                            "target_visible": visible,
                            "pixel_coordinate_convention": "zero-based pixel centers; Isaac K minus 0.5",
                            "target_expected_optical_z_m": optical_z,
                            "camera_location": location.tolist(),
                            "camera_rotation_matrix": rotation.tolist(),
                            "K": K,
                        }
                    )
        result["ok"] = True
    finally:
        (args.output / "render_manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
