"""Headless Isaac Sim 6.0: 30 cm camera-rect depth vs CAMERA_RECT_DEPTH_M.

Fronto-parallel quad at z = -0.30 m, default camera looking -Z. Median
distance_to_image_plane must land within 5 mm of 0.30 m.
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

CAMERA_RECT_DEPTH_M = 0.30


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

depth = -CAMERA_RECT_DEPTH_M
half = 0.10
rect = UsdGeom.Mesh.Define(stage, "/World/camera_rect")
rect.CreatePointsAttr([(-half, -half, depth), (half, -half, depth), (half, half, depth), (-half, half, depth)])
rect.CreateFaceVertexCountsAttr([4])
rect.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
rect.CreateNormalsAttr([(0, 0, 1)] * 4)
UsdShade.MaterialBindingAPI(rect.GetPrim()).Bind(_material(stage, "/World/mat_rect", (0.85, 0.75, 0.2)))

key = UsdLux.DistantLight.Define(stage, "/World/key")
key.CreateIntensityAttr(5000.0)
UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 45.0))
dome = UsdLux.DomeLight.Define(stage, "/World/dome")
dome.CreateIntensityAttr(800.0)

cam = UsdGeom.Camera.Define(stage, "/World/cam")
UsdGeom.Xformable(cam).AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, 0.0))
cam.CreateClippingRangeAttr(Gf.Vec2f(0.05, 20.0))

rgb, z = _capture("/World/cam")
center, frac, spread = _median_finite(z)
depth_ok = bool(frac > 0.2 and abs(center - CAMERA_RECT_DEPTH_M) < 0.005 and spread < 0.01)
rgb_ok = bool(float(rgb.std()) > 1.0)
ok = depth_ok and rgb_ok

report = _jsonable(
    {
        "ok": ok,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "isaacsim": "6.0.0.1",
        "expected_m": CAMERA_RECT_DEPTH_M,
        "depth_m": center,
        "finite_frac": frac,
        "spread_m": spread,
        "rgb_std": float(rgb.std()),
        "checks": {"depth": depth_ok, "rgb": rgb_ok},
    }
)
print(json.dumps(report, indent=2), flush=True)
out = Path(os.environ.get("BENCH_OUT", "/tmp/pruning_camera_rect.json"))
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
app.close()
sys.exit(0 if ok else 1)
