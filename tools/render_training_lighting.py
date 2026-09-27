#!/usr/bin/env python3
"""Re-render one Envy training tree at its recorded poses under seeded lighting draws.

Run with Blender --background orchard_template.blend --python this.py -- ...
The companion generator's own functions set the scene up exactly as the
August training render did (placeholders removed, overcast world, dirt ground,
28 mm lens, 1920x1080, the template's 4096-sample adaptive sampler with CPU
OpenImageDenoise, Filmic); this script asserts those settings and never
changes them. The camera is placed from each frame's annotation JSON; for the
box_cam sets the camera box is placed at the centre pose and not moved for the
left and right views, as the generator did. The only change is the light: one
draw of ``training_lighting.sample`` per frame, applied through the companion's
``daylight_presets.apply_preset`` code path.

Only the RGB is kept. Depth and the tree mask are rendered to node-local
scratch, compared with the surviving files the training row will reuse, and
deleted; a frame whose geometry does not reproduce is kept in the manifest with
``ok`` false and is never used for training. No companion file is written.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True  # Blender ignores PYTHONDONTWRITEBYTECODE; never write into the companion.

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import bpy  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import training_lighting as tl  # noqa: E402

#: The template's training sampler, read on CPU from orchard_template.blend after the generator's setup.
TEMPLATE_SAMPLER = {
    "samples": 4096,
    "use_adaptive_sampling": True,
    "use_denoising": True,
    "denoiser": "OPENIMAGEDENOISE",
    "seed": 0,
    "use_animated_seed": False,
    "filter_width": 1.5,
}


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_channel(path):
    """First channel of an image Blender can read, top row first."""
    image = bpy.data.images.load(str(path))
    width, height = image.size
    values = np.empty(width * height * image.channels, dtype=np.float32)
    image.pixels.foreach_get(values)
    array = values.reshape(height, width, image.channels)[::-1].copy()
    bpy.data.images.remove(image)
    return array


def luma(array):
    rgb = np.clip(array[..., :3], 0.0, 1.0)
    return float(255.0 * (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]).mean())


def sampler_settings(scene):
    cycles = scene.cycles
    return {
        "samples": cycles.samples,
        "use_adaptive_sampling": bool(cycles.use_adaptive_sampling),
        "adaptive_threshold": float(cycles.adaptive_threshold),
        "use_denoising": bool(cycles.use_denoising),
        "denoiser": str(cycles.denoiser),
        "seed": cycles.seed,
        "use_animated_seed": bool(cycles.use_animated_seed),
        "filter_width": float(cycles.filter_width),
        "view_transform": scene.view_settings.view_transform,
        "resolution": [scene.render.resolution_x, scene.render.resolution_y, scene.render.resolution_percentage],
        "device": str(cycles.device),
    }


def setup_compositor(scene, folder, stem, pass_index):
    scene.use_nodes = True
    nodes, links = scene.node_tree.nodes, scene.node_tree.links
    nodes.clear()
    layer = nodes.new("CompositorNodeRLayers")
    depth = nodes.new("CompositorNodeOutputFile")
    depth.base_path = str(folder)
    depth.format.file_format = "OPEN_EXR"
    depth.format.color_depth = "32"
    depth.file_slots[0].path = stem + "_depth_"
    links.new(layer.outputs["Depth"], depth.inputs[0])
    mask = nodes.new("CompositorNodeIDMask")
    mask.index = pass_index
    links.new(layer.outputs["IndexOB"], mask.inputs[0])
    out = nodes.new("CompositorNodeOutputFile")
    out.base_path = str(folder)
    out.format.file_format = "PNG"
    out.format.color_mode = "BW"
    out.file_slots[0].path = stem + "_mask_"
    links.new(mask.outputs[0], out.inputs[0])


def place_camera(gen, cam, companion, frame):
    """Camera at the annotated pose; the box_cam camera box at the centre pose, as the generator placed it."""
    annotation = json.loads(tl.annotation_path(companion, frame).read_text())
    rect_present = False
    if frame["set_id"].startswith("box_cam"):
        centre = json.loads(tl.annotation_path(companion, {**frame, "view": "c"}).read_text())
        cam.location = centre["camera"]["location"]
        cam.rotation_mode = "XYZ"
        cam.rotation_euler = centre["camera"]["rotation_euler"]
        bpy.context.view_layer.update()
        rect_present = gen.update_camera_rect(cam) is not None
    else:
        rect = bpy.data.objects.get("camera_ground_rect")
        if rect is not None:
            bpy.data.objects.remove(rect, do_unlink=True)
    cam.location = annotation["camera"]["location"]
    cam.rotation_mode = "XYZ"
    cam.rotation_euler = annotation["camera"]["rotation_euler"]
    bpy.context.view_layer.update()
    return annotation, rect_present


def apply_lighting(daylight_presets, scene, name, config, seed):
    """Apply a drawn config through the companion's own preset code path, injected for one call."""
    if config is None:
        return daylight_presets.apply_preset(scene, name, seed)
    daylight_presets.PRESETS[name] = config
    try:
        return daylight_presets.apply_preset(scene, name, seed)
    finally:
        daylight_presets.PRESETS.pop(name, None)


def write_manifest(output, result):
    """Atomic rewrite of the render manifest with the frames recorded so far."""
    result["frames_ok"] = sum(1 for f in result["frames"] if f.get("ok"))
    temporary = output / "render_manifest.json.tmp"
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(output / "render_manifest.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--companion", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True, help="Frozen batch plan listing this tree's frames")
    parser.add_argument("--tree-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scratch", type=Path, required=True, help="Node-local directory for depth and mask checks")
    parser.add_argument(
        "--source-checks", type=int, default=0, help="Also render N frames under 'source' vs surviving RGB"
    )
    parser.add_argument("--smoke", action="store_true", help="64 px, 1 sample, CPU: code path only, no assertions")
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args(sys.argv[sys.argv.index("--") + 1 :])
    args.output.mkdir(parents=True, exist_ok=False)
    args.scratch.mkdir(parents=True, exist_ok=True)
    os.environ["COMPUTER_VISION_ROOT"] = str(args.companion)
    sys.path.insert(0, str(args.companion / "Dataloader"))
    import daylight_presets
    import generate_tree2 as gen

    plan = json.loads(args.plan.read_text())
    frames = plan["trees"][args.tree_id]["frames"]
    if args.max_frames:
        frames = frames[: args.max_frames]
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.scale_length = 1.0
    gen.remove_placeholder_objects()
    gen.fix_world_background()
    gen.fix_ground_material()
    cam = bpy.data.objects[gen.CAM]
    cam.data.lens = 28.0
    scene.camera = cam
    gen.render_tree(scene)
    if args.smoke:
        scene.cycles.device = "CPU"
        scene.render.resolution_x, scene.render.resolution_y = 64, 36
        scene.cycles.samples = 1
        scene.cycles.use_adaptive_sampling = False
        scene.cycles.use_denoising = False
        scene.render.threads_mode = "FIXED"
        scene.render.threads = 2
    settings = sampler_settings(scene)
    if not args.smoke:
        wrong = {k: (settings[k], v) for k, v in TEMPLATE_SAMPLER.items() if settings[k] != v}
        if wrong or settings["view_transform"] != "Filmic" or settings["resolution"] != [gen.W, gen.H, 100]:
            raise RuntimeError(f"Template sampler differs from the training render: {wrong} {settings}")
    template_tree = bpy.data.objects.get(gen.TREE_OBJ.format(0))
    saved = {0: (template_tree.matrix_world.to_translation().copy(), None)}
    gen.remove_all_tree_objects()
    texture = next(t for t in gen.BARK_TEXTURES if t["name"] == "bark_brown_02")
    metadata = args.companion / "trees/metadata" / f"{args.tree_id}_metadata.json"
    tree = gen.get_tree_object_for_metadata(args.tree_id, str(metadata), 0, saved, texture)
    if tree is None:
        raise RuntimeError("Tree failed to load")
    gen.set_pass_indices(tree, gen.get_background_objects())
    view_layer = bpy.context.view_layer
    view_layer.use_pass_z = True
    view_layer.use_pass_object_index = True

    result = {
        "schema_version": 1,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "tree_id": args.tree_id,
        "smoke": args.smoke,
        "sampler_version": tl.SAMPLER_VERSION,
        "render_settings": settings,
        "blender": bpy.app.version_string,
        "source_blend_sha256": digest(bpy.data.filepath),
        "companion_sha256": {
            name: digest(args.companion / name)
            for name in ("Dataloader/generate_tree2.py", "Dataloader/daylight_presets.py", "Dataloader/move_camera.py")
        },
        "metadata_sha256": digest(metadata),
        "frames": [],
        "source_checks": [],
        "ok": False,
    }
    try:
        for index, frame in enumerate(frames):
            started = time.time()
            annotation, rect_present = place_camera(gen, cam, args.companion, frame)
            draw = tl.sample(
                frame["tree"], frame["set_id"], frame["shot"], frame["view"], annotation["camera"]["rotation_euler"]
            )
            label = tl.lighting_label(draw)
            applied = apply_lighting(daylight_presets, scene, label, draw["config"], draw["seed"] % (2**31))
            stem = f"{frame['tree']}_{frame['set_id']}_{frame['shot']}_{frame['view']}"
            setup_compositor(scene, args.scratch, stem, gen.PASS_INDEX_TREE)
            scene.frame_set(int(frame["shot"].replace("shot", "")))
            folder = args.output / frame["set_id"]
            folder.mkdir(exist_ok=True)
            rgb = folder / f"{stem}.png"
            scene.render.filepath = str(rgb)
            bpy.ops.render.render(write_still=True)
            exr = next(args.scratch.glob(f"{stem}_depth_*.exr"))
            mask_png = next(args.scratch.glob(f"{stem}_mask_*.png"))
            new_depth = load_channel(exr)[..., 0]
            new_mask = load_channel(mask_png)[..., 0] > 0.5
            exr.unlink()
            mask_png.unlink()
            record = {
                "index": index,
                **frame,
                "rgb": str(rgb),
                "rgb_sha256": digest(rgb),
                "lighting_label": label,
                "lighting": draw,
                "applied_sun": applied.get("sun"),
                "rect_present": rect_present,
                "mean_luma": luma(load_channel(rgb)),
                "render_seconds": time.time() - started,
            }
            if args.smoke:
                record["agreement"] = {"ok": None, "reason": "smoke render at 64 px; the surviving files are 1920x1080"}
                record["ok"] = True
            else:
                old_depth = np.load(frame["depth_path"], allow_pickle=False)
                old_mask = load_channel(frame["mask_path"])[..., 0] > 0.5
                record["agreement"] = tl.agreement(new_depth, new_mask, old_depth, old_mask)
                record["ok"] = bool(record["agreement"]["ok"])
            result["frames"].append(record)
            # Written after every frame: a task stopped by its time limit keeps the record of what it rendered.
            write_manifest(args.output, result)
            print(
                json.dumps(
                    {k: record[k] for k in ("index", "set_id", "shot", "view", "mean_luma", "render_seconds", "ok")}
                ),
                flush=True,
            )
        for frame in frames[: args.source_checks]:
            place_camera(gen, cam, args.companion, frame)
            apply_lighting(daylight_presets, scene, "source", None, 1729)
            scene.use_nodes = False
            check = args.scratch / f"source_check_{frame['set_id']}_{frame['shot']}_{frame['view']}.png"
            scene.render.filepath = str(check)
            bpy.ops.render.render(write_still=True)
            new_rgb = load_channel(check)[..., :3]
            old_rgb = load_channel(frame["rgb_path"])[..., :3]
            old_mask = load_channel(frame["mask_path"])[..., 0] > 0.5
            if new_rgb.shape == old_rgb.shape:
                diff = float(np.abs(new_rgb - old_rgb)[old_mask].mean() * 255.0) if old_mask.any() else None
            else:
                diff = None
            result["source_checks"].append(
                {
                    "set_id": frame["set_id"],
                    "shot": frame["shot"],
                    "view": frame["view"],
                    "tree_mask_mean_abs_diff_0_255": diff,
                }
            )
            check.unlink()
        result["ok"] = True
    finally:
        write_manifest(args.output, result)


if __name__ == "__main__":
    main()
