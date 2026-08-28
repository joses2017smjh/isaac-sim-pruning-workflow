"""Headless Isaac Sim 6.0 cube + plane smoke for the pruning workflow.

Reuses the v60 stack already proven on this cluster (A40, job 21036831):
isaacsim 6.0.0.1 + isaaclab 3.0.0b2. Isaac Sim 5.1's RTX path segfaults here.

Stage units are metres. The cube capture matches the BHL probe's camera
convention (Y-up default, look along +Y). The plane is captured after hiding
the cube so the camera is not inside the 1 m volume.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True, "renderer": "RayTracedLighting", "width": 256, "height": 256})

import numpy as np  # noqa: E402

import omni.replicator.core as rep  # noqa: E402
import omni.usd  # noqa: E402
from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade  # noqa: E402


def _material(stage, path: str, color: tuple[float, float, float]):
    mat = UsdShade.Material.Define(stage, path)
    shader = UsdShade.Shader.Define(stage, f"{path}/surface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    return mat


def _capture(prim_path: str):
    product = rep.create.render_product(prim_path, (256, 256))
    rgb = rep.AnnotatorRegistry.get_annotator("rgb")
    depth = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb.attach(product)
    depth.attach(product)
    for _ in range(16):
        rep.orchestrator.step(rt_subframes=4)
    color = np.asarray(rgb.get_data())[..., :3].astype(np.float32)
    z = np.asarray(depth.get_data()).astype(np.float32)
    return color, z


def _median_finite(z: np.ndarray) -> tuple[float, float, float]:
    finite = np.isfinite(z) & (z > 0) & (z < 1e10)
    if not finite.any():
        return float("nan"), 0.0, float("nan")
    values = z[finite]
    spread = float(np.nanpercentile(values, 95) - np.nanpercentile(values, 5))
    return float(np.median(values)), float(finite.mean()), spread


def _jsonable(value):
    """numpy.bool_ reports as class name 'bool' and is not JSON-serializable."""
    if isinstance(value, dict):
        return {key: _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (str, int, float, type(None))):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


omni.usd.get_context().new_stage()
stage = omni.usd.get_context().get_stage()
UsdGeom.SetStageMetersPerUnit(stage, 1.0)
UsdGeom.Xform.Define(stage, "/World")

cube = UsdGeom.Cube.Define(stage, "/World/cube")
cube.CreateSizeAttr(1.0)
UsdShade.MaterialBindingAPI(cube.GetPrim()).Bind(_material(stage, "/World/mat_cube", (0.9, 0.45, 0.2)))

key = UsdLux.DistantLight.Define(stage, "/World/key")
key.CreateIntensityAttr(5000.0)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 45.0))
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(800.0)

cube_cam = UsdGeom.Camera.Define(stage, "/World/cube_cam")
UsdGeom.Xformable(cube_cam).AddTranslateOp().Set(Gf.Vec3d(0.0, -2.0, 0.0))
UsdGeom.Xformable(cube_cam).AddRotateXYZOp().Set(Gf.Vec3f(90.0, 0.0, 0.0))
cube_cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))

cube_rgb, cube_z = _capture("/World/cube_cam")
cube_center, cube_frac, cube_spread = _median_finite(cube_z)

UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()

plane = UsdGeom.Mesh.Define(stage, "/World/plane")
plane.CreatePointsAttr([(-2, -2, -2), (2, -2, -2), (2, 2, -2), (-2, 2, -2)])
plane.CreateFaceVertexCountsAttr([4])
plane.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
plane.CreateNormalsAttr([(0, 0, 1)] * 4)
UsdShade.MaterialBindingAPI(plane.GetPrim()).Bind(_material(stage, "/World/mat_plane", (0.2, 0.6, 0.9)))

plane_cam = UsdGeom.Camera.Define(stage, "/World/plane_cam")
UsdGeom.Xformable(plane_cam).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
plane_cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))

plane_rgb, plane_z = _capture("/World/plane_cam")
plane_center, plane_frac, plane_spread = _median_finite(plane_z)

# Front face of a 1 m cube at the origin, viewed from y=-2: 1.5 m.
cube_ok = bool(cube_frac > 0.05 and abs(cube_center - 1.5) < 0.05)
# Fronto-parallel plane at z=-2, default camera looking -Z: constant 2.0 m.
plane_ok = bool(plane_frac > 0.2 and abs(plane_center - 2.0) < 0.05 and plane_spread < 0.05)
rgb_ok = bool(float(cube_rgb.std()) > 1.0)
ok = cube_ok and plane_ok and rgb_ok

print(
    f"VERDICT cube={cube_center:.4f}m (expect 1.5) plane={plane_center:.4f}m "
    f"(expect 2.0, spread {plane_spread:.4f}) rgb_std={float(cube_rgb.std()):.2f} "
    f"ok={ok}",
    flush=True,
)

report = _jsonable({
    "ok": ok,
    "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
    "isaacsim": "6.0.0.1",
    "isaaclab": "3.0.0b2",
    "venv": "/nfs/hpc/share/sanchej7/Humanoid_Lite/venv-isaac60",
    "sif": "/nfs/hpc/share/sanchej7/Humanoid_Lite/container/bhl.sif",
    "meters_per_unit": UsdGeom.GetStageMetersPerUnit(stage),
    "cube_depth_m": cube_center,
    "cube_finite_frac": cube_frac,
    "cube_spread_m": cube_spread,
    "cube_expected_m": 1.5,
    "plane_depth_m": plane_center,
    "plane_finite_frac": plane_frac,
    "plane_spread_m": plane_spread,
    "plane_expected_m": 2.0,
    "cube_rgb_mean": float(cube_rgb.mean()),
    "cube_rgb_std": float(cube_rgb.std()),
    "checks": {"cube": cube_ok, "plane": plane_ok, "rgb": rgb_ok},
})

print(json.dumps(report, indent=2), flush=True)
out = Path(os.environ.get("BENCH_OUT", "/tmp/pruning_isaac_smoke.json"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

if ok:
    try:
        pruning_root = Path(os.environ["PRUNING_ROOT"])
        sys.path.insert(0, str(pruning_root / "source" / "isaaclab_pruning"))
        from isaaclab_pruning.geometry.cylinders import load_cylinders, transform_cylinders
        from isaaclab_pruning.usd.cylinders import DEFAULT_TREE_TILT_X_DEG, _local_transform, write_cylinder_tree_usd

        metadata_dir = Path(
            os.environ.get("TREE_METADATA", "/nfs/hpc/share/sanchej7/Computer_Vision/trees/metadata")
        )
        usd_dir = pruning_root / "artifacts" / "trees"
        debug_ids = [f"lpy_envy_{i:05d}" for i in range(10)]
        converted = []
        for tree_id in debug_ids:
            metadata = metadata_dir / f"{tree_id}_metadata.json"
            if not metadata.is_file():
                continue
            tilt = _local_transform(DEFAULT_TREE_TILT_X_DEG, (0, 0, 0))
            cylinders = transform_cylinders(load_cylinders(metadata), tilt)
            converted.append(
                write_cylinder_tree_usd(cylinders, usd_dir / f"{tree_id}.usda", tree_id=tree_id)
            )
        report["trees"] = _jsonable({
            "converted": [item["tree_id"] for item in converted],
            "count": len(converted),
            "output_dir": str(usd_dir),
        })
        print(json.dumps(report["trees"], indent=2), flush=True)
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except Exception as error:  # noqa: BLE001 — record and still emit the cube/plane evidence
        report["trees"] = {"converted": [], "error": repr(error)}
        out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

app.close()
sys.exit(0 if ok else 1)
