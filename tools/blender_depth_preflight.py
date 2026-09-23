"""Empirical optical-Z and lighting determinism preflight inside Blender."""

import json
import sys
from pathlib import Path

import bpy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from render_family_lighting import load_exr, setup_outputs

out = Path(sys.argv[sys.argv.index("--") + 1])
out.mkdir(parents=True, exist_ok=False)
original_lights = [
    {
        "name": o.name,
        "type": o.data.type,
        "energy": o.data.energy,
        "color": list(o.data.color),
        "angle": o.data.angle,
        "rotation": list(o.rotation_euler),
    }
    for o in bpy.context.scene.objects
    if o.type == "LIGHT"
]
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
bpy.ops.object.camera_add(location=(0, 0, 0))
scene.camera = bpy.context.object
scene.camera.data.lens = 20
bpy.ops.mesh.primitive_plane_add(size=20, location=(0, 0, -2))
scene.render.engine = "CYCLES"
scene.cycles.device = "CPU"
scene.cycles.samples = 1
scene.render.threads_mode = "FIXED"
scene.render.threads = 2
scene.render.resolution_x = 128
scene.render.resolution_y = 72
scene.render.resolution_percentage = 100
bpy.context.view_layer.use_pass_z = True
bpy.context.view_layer.use_pass_object_index = True
setup_outputs(scene, out, "plane", True)
scene.render.filepath = str(out / "plane.png")
bpy.ops.render.render(write_still=True)
depth = load_exr(next(out.glob("plane_depth_*.exr")))
np.save(out / "plane.npy", depth)
error = float(np.max(np.abs(depth - 2)))
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Computer_Vision/Dataloader")
from daylight_presets import apply_preset

lights = {}
for name in ["source", "morning", "noon", "evening"]:
    one = apply_preset(scene, name, 1729)
    two = apply_preset(scene, name, 1729)
    assert one == two
    lights[name] = one
result = {
    "ok": error < 1e-4,
    "plane_optical_z_m": 2,
    "min_pass_m": float(depth.min()),
    "max_pass_m": float(depth.max()),
    "max_abs_error_m": error,
    "conclusion": "Cycles Depth is optical Z, not Euclidean ray distance, for this perspective camera",
    "template_lights": original_lights,
    "lighting_deterministic": True,
    "lighting": lights,
}
(out / "preflight.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
assert result["ok"]
