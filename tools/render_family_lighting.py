#!/usr/bin/env python3
"""Small paired Blender study using generate_tree2 geometry/material functions.

Run with Blender --background orchard_template.blend --python this.py -- ...
All lights share exactly the same geometry, six camera poses, seed and renderer.

Two renderers share the camera rigs, the tree placement, the lights
(``daylight_presets.apply_preset`` with seed 1729), the geometry outputs and
the manifest format:

* ``--renderer matrix`` (the default, unchanged in its outputs; both modes now also stop Python writing
  bytecode into the companion): the family-matrix renderer,
  ``--width`` (512) by 9/16, ``--samples`` (16) fixed samples with the
  template's denoiser, Cycles seed 1729, Filmic set explicitly, no world fix.
* ``--renderer training``: the companion training render's settings.
  ``generate_tree2.fix_world_background()`` runs before any light is applied,
  ``generate_tree2.render_tree(scene)`` sets engine, GPU and resolution
  (1920x1080), and the template's sampler is never changed: it is asserted
  against ``TEMPLATE_SAMPLER``, the check ``render_training_lighting.py``
  makes, and a difference raises. A CPU fallback is refused, as the training
  render refuses it. View transform, look, exposure, gamma and depth of field
  are left as the template holds them and recorded with the full sampler.
  Under the first light, each view's depth and tree mask are also rendered
  once on the matrix grid (512x288) with the matrix renderer's sampler (16
  samples, seed 1729, no adaptive sampling) into ``geometry_matrix_grid/``,
  and recorded as ``matrix_grid_geometry``. That RGB is discarded, every
  changed setting is restored and checked before the next frame, and the
  render-gap evaluation compares this pass, never the 1920x1080 geometry, with
  the published matrix ground truth: Cycles takes depth and object index at
  pixel centres, and 512 does not divide 1920, so no 1920x1080 pixel samples a
  matrix pixel centre. The manifest is rewritten after every frame, so a task
  stopped by its time limit keeps the record of what it rendered.

``--smoke`` (training renderer only) is a CPU code-path check: 64 px wide,
1 sample, no adaptive sampling, CPU device, no template assertion; its
matrix-grid pass keeps the 512x288 grid at 1 sample. Its manifest says
``smoke``, and the render-gap evaluation refuses it.

The module imports Blender only inside the functions that need it, so the
argument rules and the template check can be tested without Blender;
``render_generalization_controls.py`` reuses ``digest``, ``load_exr`` and
``setup_outputs``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path

import numpy as np

MATRIX_WIDTH = 512
MATRIX_SAMPLES = 16
MATRIX_SEED = 1729
MATRIX_GRID_FOLDER = "geometry_matrix_grid"
#: The scene settings a matrix-grid geometry pass changes, by owner; each is restored after the pass.
GRID_STATE_KEYS = {
    "render": ("resolution_x", "resolution_y", "resolution_percentage"),
    "cycles": ("samples", "seed", "use_animated_seed", "use_adaptive_sampling"),
}
#: The template's training sampler, as ``render_training_lighting.TEMPLATE_SAMPLER`` asserts it (a test keeps
#: the two equal). The view transform and the 100% resolution of generate_tree2's W x H are checked with it.
TEMPLATE_SAMPLER = {
    "samples": 4096,
    "use_adaptive_sampling": True,
    "use_denoising": True,
    "denoiser": "OPENIMAGEDENOISE",
    "seed": 0,
    "use_animated_seed": False,
    "filter_width": 1.5,
}
TEMPLATE_VIEW_TRANSFORM = "Filmic"
SMOKE_WIDTH = 64
SMOKE_SAMPLES = 1
#: Recorded with the sampler when this Blender has them; never compared.
EXTRA_CYCLES_SETTINGS = (
    "sampling_pattern",
    "time_limit",
    "denoising_input_passes",
    "denoising_prefilter",
    "denoising_quality",
    "denoising_use_gpu",
    "max_bounces",
    "diffuse_bounces",
    "glossy_bounces",
    "transmission_bounces",
    "volume_bounces",
    "transparent_max_bounces",
    "sample_clamp_direct",
    "sample_clamp_indirect",
    "blur_glossy",
    "caustics_reflective",
    "caustics_refractive",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_exr(path):
    import bpy

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


def parse_args(argv):
    """The command line; the matrix defaults are filled in here so the training renderer can refuse them."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--tree-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--renderer",
        choices=("matrix", "training"),
        default="matrix",
        help="matrix: the family-matrix renderer (default); training: the companion training render's settings",
    )
    parser.add_argument("--width", type=int, default=None, help=f"Matrix renderer only (default {MATRIX_WIDTH})")
    parser.add_argument("--samples", type=int, default=None, help=f"Matrix renderer only (default {MATRIX_SAMPLES})")
    parser.add_argument("--lights", nargs="+", default=["source", "morning", "noon", "evening"])
    parser.add_argument("--one-view", action="store_true")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=f"Training renderer only: {SMOKE_WIDTH} px, {SMOKE_SAMPLES} sample, CPU; a code-path check, never data",
    )
    args = parser.parse_args(argv)
    if args.renderer == "matrix":
        if args.smoke:
            parser.error("--smoke checks the training renderer's code path; the matrix renderer has no smoke mode")
        args.width = MATRIX_WIDTH if args.width is None else args.width
        args.samples = MATRIX_SAMPLES if args.samples is None else args.samples
    elif args.width is not None or args.samples is not None:
        parser.error("--width and --samples are matrix-renderer options; the training renderer keeps the template's")
    return args


def sampler_settings(scene):
    """The settings the training check compares, plus the rest of the sampler, as Blender holds them."""
    cycles, view, render = scene.cycles, scene.view_settings, scene.render
    settings = {
        "samples": cycles.samples,
        "use_adaptive_sampling": bool(cycles.use_adaptive_sampling),
        "adaptive_threshold": float(cycles.adaptive_threshold),
        "adaptive_min_samples": int(cycles.adaptive_min_samples),
        "use_denoising": bool(cycles.use_denoising),
        "denoiser": str(cycles.denoiser),
        "seed": cycles.seed,
        "use_animated_seed": bool(cycles.use_animated_seed),
        "filter_width": float(cycles.filter_width),
        "view_transform": view.view_transform,
        "look": view.look,
        "exposure": float(view.exposure),
        "gamma": float(view.gamma),
        "resolution": [render.resolution_x, render.resolution_y, render.resolution_percentage],
        "device": str(cycles.device),
    }
    for name in EXTRA_CYCLES_SETTINGS:
        value = getattr(cycles, name, None)
        if value is not None:
            settings[name] = value if isinstance(value, (bool, int, float)) else str(value)
    return settings


def template_differences(settings, width, height):
    """Each setting that differs from the training render's template, as (found, expected); empty if none."""
    wrong = {k: (settings.get(k), v) for k, v in TEMPLATE_SAMPLER.items() if settings.get(k) != v}
    if settings.get("view_transform") != TEMPLATE_VIEW_TRANSFORM:
        wrong["view_transform"] = (settings.get("view_transform"), TEMPLATE_VIEW_TRANSFORM)
    if settings.get("resolution") != [width, height, 100]:
        wrong["resolution"] = (settings.get("resolution"), [width, height, 100])
    return wrong


def apply_smoke(scene):
    """The CPU code-path check: 64 px wide at 16:9, one sample, no adaptive sampling, CPU device."""
    scene.cycles.device = "CPU"
    scene.render.resolution_x = SMOKE_WIDTH
    scene.render.resolution_y = round(SMOKE_WIDTH * 9 / 16)
    scene.render.resolution_percentage = 100
    scene.cycles.samples = SMOKE_SAMPLES
    scene.cycles.use_adaptive_sampling = False


def matrix_grid_state(smoke=False):
    """The matrix renderer's grid and sampler for the training renderer's geometry pass (one sample in a smoke)."""
    return {
        "resolution_x": MATRIX_WIDTH,
        "resolution_y": round(MATRIX_WIDTH * 9 / 16),
        "resolution_percentage": 100,
        "samples": SMOKE_SAMPLES if smoke else MATRIX_SAMPLES,
        "seed": MATRIX_SEED,
        "use_animated_seed": False,
        "use_adaptive_sampling": False,
    }


def scene_state(scene):
    """The current values of the settings a matrix-grid pass changes."""
    return {key: getattr(getattr(scene, owner), key) for owner, keys in GRID_STATE_KEYS.items() for key in keys}


def set_scene_state(scene, state):
    for owner, keys in GRID_STATE_KEYS.items():
        for key in keys:
            setattr(getattr(scene, owner), key, state[key])


def intrinsics(camera, scene, width, height):
    """Zero-based pixel-centre K from Blender's own projection, avoiding the generator's hard-coded K_REF."""
    import bpy

    projection = camera.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=width,
        y=height,
        scale_x=scene.render.pixel_aspect_x,
        scale_y=scene.render.pixel_aspect_y,
    )
    return [
        [float(projection[0][0]) * width / 2, 0, width / 2 * (1 - float(projection[0][2])) - 0.5],
        [0, float(projection[1][1]) * height / 2, height / 2 * (1 + float(projection[1][2])) - 0.5],
        [0, 0, 1],
    ]


def render_matrix_grid_geometry(scene, folder, stem, state):
    """Depth and tree mask of the current view on the matrix grid; the RGB is discarded and every setting restored.

    Returns the depth and mask paths and the sampler as Blender held it for this render.
    """
    import bpy

    saved = scene_state(scene)
    try:
        set_scene_state(scene, state)
        held = sampler_settings(scene)
        setup_outputs(scene, folder, stem, True)
        bpy.ops.render.render(write_still=False)
    finally:
        set_scene_state(scene, saved)
    # The compositor still writes to ``folder``; the next frame's setup_outputs clears and rebuilds it.
    exr = next(folder.glob(f"{stem}_depth_*.exr"))
    depth = folder / f"{stem}.npy"
    np.save(depth, load_exr(exr))
    exr.unlink()
    mask = folder / f"{stem}_mask.png"
    next(folder.glob(f"{stem}_mask_*.png")).rename(mask)
    return depth, mask, held


def write_manifest(output, result):
    """Atomic rewrite of the render manifest with the frames recorded so far."""
    temporary = output / "render_manifest.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "render_manifest.json")


def compute_devices():
    """The Cycles compute devices this process can use, as Blender reports them."""
    import bpy

    addon = bpy.context.preferences.addons.get("cycles")
    if addon is None:
        return {"compute_device_type": None, "devices": []}
    preferences = addon.preferences
    return {
        "compute_device_type": str(preferences.compute_device_type),
        "devices": [{"name": d.name, "type": d.type, "use": bool(d.use)} for d in preferences.devices],
    }


def camera_settings(camera):
    dof = camera.data.dof
    return {
        "lens_mm": float(camera.data.lens),
        "sensor_width_mm": float(camera.data.sensor_width),
        "sensor_fit": str(camera.data.sensor_fit),
        "use_dof": bool(dof.use_dof),
        "dof_focus_distance_m": float(dof.focus_distance),
        "dof_aperture_fstop": float(dof.aperture_fstop),
    }


def main():
    args = parse_args(sys.argv[sys.argv.index("--") + 1 :])
    import bpy
    from bpy_extras.object_utils import world_to_camera_view
    from mathutils import Vector

    sys.dont_write_bytecode = True  # Blender ignores PYTHONDONTWRITEBYTECODE; never write into the companion.
    training = args.renderer == "training"
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
    if training:
        # As the training render: the world fix comes before any light; each preset then rebuilds the world nodes.
        gen.fix_world_background()
    gen.fix_ground_material()
    camera = bpy.data.objects[gen.CAM]
    camera.data.lens = 28.0
    if not training:
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
    if training:
        # Engine, GPU, 1920x1080 at 100%, 8-bit RGB PNG, opaque film; the sampler stays the template's.
        gen.render_tree(scene)
        if args.smoke:
            apply_smoke(scene)
    else:
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
    if not training:
        # Fixed settings across lights, recorded explicitly rather than inherited silently.
        scene.view_settings.view_transform = "Filmic"
        scene.view_settings.look = "None"
        scene.view_settings.exposure = 0.0
        scene.view_settings.gamma = 1.0
    width, height = scene.render.resolution_x, scene.render.resolution_y
    training_record = {}
    if training:
        settings = sampler_settings(scene)
        devices = compute_devices()
        if not args.smoke:
            wrong = template_differences(settings, gen.W, gen.H)
            if wrong:
                raise RuntimeError(f"Template sampler differs from the training render: {wrong} {settings}")
            if settings["device"] != "GPU":
                raise RuntimeError(
                    f"Cycles found no GPU and fell back to {settings['device']}; refusing a CPU render: {devices}"
                )
        training_record = {
            "renderer_mode": "training",
            "smoke": bool(args.smoke),
            "template_sampler": dict(TEMPLATE_SAMPLER, view_transform=TEMPLATE_VIEW_TRANSFORM),
            "template_check": "skipped: smoke render" if args.smoke else "passed",
            "sampler_settings": settings,
            "compute_devices": devices,
            "camera_settings": camera_settings(camera),
            "world_fix": "generate_tree2.fix_world_background() before the presets",
            "blender": bpy.app.version_string,
            "code_revision": os.environ.get("PRUNING_CODE_REVISION"),
        }
        grid_state = matrix_grid_state(args.smoke)
        grid_width, grid_height = grid_state["resolution_x"], grid_state["resolution_y"]
        training_record["matrix_grid_geometry"] = {
            "scope": (
                "Depth and tree mask of every view rendered once more on the matrix grid with the matrix renderer's "
                "sampler, under the first light, in this session; its RGB is discarded. The render-gap evaluation "
                "compares this pass with the published matrix ground truth (pose identity)."
            ),
            "width": grid_width,
            "height": grid_height,
            "sampler": {key: grid_state[key] for key in GRID_STATE_KEYS["cycles"]},
            "K": intrinsics(camera, scene, grid_width, grid_height),
            "lighting": args.lights[0],
            "views": {},
        }
    K = intrinsics(camera, scene, width, height)
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
    if training:
        grid_folder = args.output / MATRIX_GRID_FOLDER
        grid_folder.mkdir()
        renderer = {
            "engine": scene.render.engine,
            "mode": "training",
            "samples": settings["samples"],
            "seed": settings["seed"],
            "width": width,
            "height": height,
            "view_transform": settings["view_transform"],
            "look": settings["look"],
            "exposure": settings["exposure"],
            "gamma": settings["gamma"],
        }
    else:
        renderer = {
            "engine": "CYCLES",
            "samples": args.samples,
            "seed": 1729,
            "width": args.width,
            "height": height,
            "view_transform": scene.view_settings.view_transform,
            "look": scene.view_settings.look,
            "exposure": 0.0,
            "gamma": 1.0,
        }
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
        "renderer": renderer,
        **training_record,
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
                    target_pixel = [float(projected.x * width) - 0.5, float((1 - projected.y) * height) - 0.5]
                target_visible = False
                if target_pixel is not None:
                    x, y = [int(round(v)) for v in target_pixel]
                    optical_z = -(camera.matrix_world.inverted() @ Vector(target["centroid"])).z
                    gt = np.load(depth)
                    if 0 <= x < width and 0 <= y < height:
                        target_visible = bool(abs(float(gt[y, x]) - optical_z) <= 2 * target["radius"] + 0.01)
                if training and li == 0:
                    before = sampler_settings(scene)
                    grid_depth, grid_mask, held = render_matrix_grid_geometry(scene, grid_folder, stem, grid_state)
                    if sampler_settings(scene) != before:
                        raise RuntimeError(f"The matrix-grid pass left the render settings changed: {before}")
                    grid_record = result["matrix_grid_geometry"]
                    if grid_record.setdefault("sampler_settings", held) != held:
                        raise RuntimeError(f"The matrix-grid sampler differs between views: {held}")
                    grid_record["views"][stem] = {"depth": str(grid_depth), "mask": str(grid_mask)}
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
                if training:
                    # A task stopped by its time limit never reaches ``finally``; keep what it rendered.
                    write_manifest(args.output, result)
        result["ok"] = True
    finally:
        if training:
            # Atomic, like the per-frame writes: a kill here must not leave a truncated manifest.
            write_manifest(args.output, result)
        else:
            (args.output / "render_manifest.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
