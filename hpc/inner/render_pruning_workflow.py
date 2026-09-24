"""Record the real UR5e articulation, RTX cameras, and dual live ToF on v60.

Run through the repository's GPU wrapper, never on a login node. Default mode
records scripted inspection. PRUNING_RENDER_MODE=blender_vision instead uses
the exported orchard, causal RGB-D tracking and a discrete simulated detachment.
Neither mode is a learned policy or a claim that the task's training gates pass.
Rendered metric depth is simulator ground truth. The camera mounting is a
simulation-defined wrist camera; the two ToF offsets retain reviewed CAD poses.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import sys
import traceback
from pathlib import Path

WRIST_POSITION_IN_TOOL_M = (0.0, -0.09, -0.025)
RENDER_UPDATES_PER_FRAME = 4
INITIAL_JOINT_POSITION_RAD = (0.0, -0.65, 1.20, -0.55, -0.70, 0.0)
TARGET_POSITION_M = (0.25123947, 0.63413405, 0.81517712)


def sensor_guard(ranges, valid, consecutive_missing, *, minimum_clearance_m=0.06):
    """Safety gate on measured live ToF; the scripted goal is still geometry-derived."""
    import numpy as np

    ranges, valid = np.asarray(ranges, dtype=float), np.asarray(valid, dtype=bool)
    samples = ranges[valid & np.isfinite(ranges) & (ranges > 0.0)]
    missing = consecutive_missing + 1 if not samples.size else 0
    if samples.size and samples.min() < minimum_clearance_m:
        return "tof_minimum_clearance", missing
    if missing >= 4:
        return "both_tof_missing_4_frames", missing
    return None, missing


def manual_capture_interval(current_step, frame_count, physics_steps_per_frame):
    """Put the automatic-render boundary beyond this bounded capture sequence."""
    values = (current_step, frame_count, physics_steps_per_frame)
    if any(type(value) is not int for value in values) or current_step < 0 or min(values[1:]) <= 0:
        raise ValueError("Expected nonnegative current step and positive integer capture sizes")
    return current_step + frame_count * physics_steps_per_frame + 1


def command_tracking_error_m(tool_position_w, command_position_b, root_position_w, root_rotation_w):
    """Measure against the issued command, including a latched stop command.

    Commands are root-relative, while measured tool positions are world-space.
    The scheduled path may continue after a stop and must not enter this metric.
    """
    import numpy as np

    tool = np.asarray(tool_position_w, dtype=float)
    command = np.asarray(command_position_b, dtype=float)
    root = np.asarray(root_position_w, dtype=float)
    rotation = np.asarray(root_rotation_w, dtype=float)
    if tool.shape != (3,) or command.shape != (3,) or root.shape != (3,) or rotation.shape != (3, 3):
        raise ValueError("Expected three-vector positions and a 3x3 root rotation.")
    if not all(np.isfinite(value).all() for value in (tool, command, root, rotation)):
        raise ValueError("Tracking poses must be finite.")
    return float(np.linalg.norm(tool - (root + rotation @ command)))


def optical_rotation_from_forward(direction):
    """Right-handed ROS optical rotation; +Z looks forward, +Y points down.

    The returned rotation is fixed in the tool frame at initialization. This
    is a target-informed simulation mount, not a calibrated hardware camera
    and not a per-frame oracle look-at camera.
    """
    import numpy as np

    forward = np.asarray(direction, dtype=float)
    if forward.shape != (3,) or not np.isfinite(forward).all() or np.linalg.norm(forward) < 1.0e-8:
        raise ValueError("Expected a finite nonzero three-vector camera direction.")
    forward = forward / np.linalg.norm(forward)
    nominal_down = np.array([0.0, 1.0, 0.0])
    right = np.cross(nominal_down, forward)
    if np.linalg.norm(right) < 1.0e-6:
        raise ValueError("Camera direction cannot be parallel to the tool's Y axis.")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)
    return np.column_stack((right, down, forward))


def seed_visibility(depth, pixel_xy, expected_optical_z_m, *, tolerance_m=0.02):
    """Validate the known selection ONCE; never supply oracle positions to servoing."""
    import numpy as np

    array = np.asarray(depth)
    pixel = np.asarray(pixel_xy, dtype=float)
    result = {
        "visible": False,
        "expected_optical_z_m": float(expected_optical_z_m),
        "measured_optical_z_m": None,
        "tolerance_m": float(tolerance_m),
    }
    if array.ndim != 2 or pixel.shape != (2,) or not np.isfinite(pixel).all():
        return {**result, "reason": "invalid_seed_projection"}
    x, y = np.rint(pixel).astype(int)
    if not (0 <= x < array.shape[1] and 0 <= y < array.shape[0]):
        return {**result, "reason": "seed_outside_image"}
    z = float(array[y, x])
    if not np.isfinite(z) or z <= 0 or not np.isfinite(expected_optical_z_m) or expected_optical_z_m <= 0:
        return {**result, "reason": "invalid_seed_depth"}
    result["measured_optical_z_m"] = z
    visible = abs(z - expected_optical_z_m) <= tolerance_m
    return {
        **result,
        "visible": bool(visible),
        "reason": None if visible else "selected_branch_occluded_or_wrong_surface",
    }


def episode_command(frame: int, count: int) -> tuple[str, tuple[float, float, float]]:
    """Smooth root-frame translation relative to the measured settled tool."""
    if count < 30 or not 0 <= frame < count:
        raise ValueError("Expected at least 30 frames and an in-range frame index.")
    fraction = frame / (count - 1)
    if fraction < 0.08:
        phase, progress = "observe", 0.0
    elif fraction < 0.50:
        phase = "approach"
        progress = (fraction - 0.08) / 0.42
    elif fraction < 0.64:
        phase, progress = "align", 1.0
    elif fraction < 0.78:
        phase, progress = "inspect_at_standoff", 1.0
    else:
        phase = "retreat"
        progress = 1.0 - (fraction - 0.78) / 0.22
    progress = max(0.0, min(1.0, progress))
    blend = progress * progress * (3.0 - 2.0 * progress)
    offset = [-0.090 * blend, 0.250 * blend, 0.020 * blend]
    if phase == "align":
        offset[0] += 0.006 * math.sin(2.0 * math.pi * (fraction - 0.50) / 0.14)
    return phase, tuple(offset)


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "tolist"):
        return _json_safe(value.tolist())
    return value


def main() -> int:  # noqa: C901 - the simulator is imported only after AppLauncher.
    from dataclasses import asdict

    from isaaclab_pruning.sim.render_quality import CaptureQuality, apply_capture_quality, settings_readback
    from isaaclab_pruning.sim.vision_demo_controller import ApproachStrategy

    root = Path(os.environ["PRUNING_ROOT"])
    quality = None
    blender_mode = os.environ.get("PRUNING_RENDER_MODE") == "blender_vision"
    photometric_normalization = os.environ.get("PRUNING_PHOTOMETRIC_NORMALIZATION", "raw")
    if photometric_normalization not in ("raw", "clahe"):
        raise ValueError("PRUNING_PHOTOMETRIC_NORMALIZATION must be raw or clahe")
    # Labelled approach strategy; defaults are the September 23 baseline. Gates
    # and thresholds are not reachable from the environment.
    approach = ApproachStrategy(
        mode=os.environ.get("PRUNING_APPROACH_MODE", "straight"),
        standoff_m=float(os.environ.get("PRUNING_STANDOFF_M", "0")),
        max_step_m=float(os.environ.get("PRUNING_MAX_STEP_M", "0.004")),
    )
    wrist_mount = (0.0, -0.14, -0.025) if blender_mode else WRIST_POSITION_IN_TOOL_M
    if os.environ.get("PRUNING_RENDER_QUALITY") == "pathtraced":
        quality = CaptureQuality(
            overview_width=int(os.environ.get("PRUNING_OVERVIEW_WIDTH", "1280")),
            total_samples=int(os.environ.get("PRUNING_RENDER_SAMPLES", "64")),
        )
    overview_resolution = quality.overview_resolution if quality else (960, 640)
    close_resolution = (640, 480)
    wrist_resolution = (480, 320)
    output = Path(os.environ["PRUNING_RENDER_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    if (output / "frames.json").exists() or (output / "frames").exists():
        raise FileExistsError(f"Refusing to overwrite an earlier capture: {output}")
    frame_dir = output / "frames"
    frame_dir.mkdir()
    frame_count = int(os.environ.get("PRUNING_RENDER_FRAMES", "140"))
    if frame_count < 30:
        raise ValueError("PRUNING_RENDER_FRAMES must be at least 30.")
    fps = 10
    records = []
    report = {
        "ok": False,
        "rendering_ok": False,
        "stage": "launch",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.environ.get("SLURMD_NODENAME") or os.uname().nodename,
        "task_outcome": "not_started",
        "frame_count": 0,
        "photometric_normalization": photometric_normalization,
        "approach_strategy": asdict(approach),
        "provenance": {
            "robot": "UR5e + reviewed BDS mock-pruner, six actuated UR joints",
            "asset_id": os.environ.get("PRUNING_ASSET_ID"),
            "usd": os.environ.get("PRUNING_USD"),
            "usd_evidence": os.environ.get("PRUNING_USD_EVIDENCE"),
            "runtime": "Isaac Sim 6.0.0.1 / Isaac Lab 3.0.0b2",
            "controller": (
                "known-geometry scripted Cartesian differential IK with physical implicit joint drives "
                "and measured ToF stop gate; no learned CV input"
            ),
            "robot_state": "measured PhysX articulation state, not teleported joint replay",
            "overview_rgb": "RTX rgb annotator",
            "wrist_rgb": (
                "RTX rgb annotator; exterior tool-mounted camera with fixed simulation-defined toe-in; "
                "not hardware calibration"
            ),
            "close_rgb": "RTX rgb annotator; static external close-up camera",
            "depth": "RTX distance_to_image_plane; simulator ground truth, not learned depth",
            "tof": "two live MultiMeshRayCasterCamera 8x8 15Hz at reviewed sensor offsets; noise disabled",
            "environment": "procedural finite-cylinder pruning fixture and tree backdrop",
            "cutting": "inspection at standoff only; no wood severing or trained policy",
            "contact": "PhysX measured forces; coverage recorded explicitly",
        },
        "artifact_inventory": {
            "overview": "frames/overview_*.png",
            "close": "frames/close_*.png",
            "wrist": "frames/wrist_*.png",
            "metric_depth": "frames/depth_*.npy",
            "telemetry": "frames.json",
            "stage": "scene.usda",
        },
    }
    if blender_mode:
        report["provenance"].update(
            controller="Causal pyramidal-LK RGB-D visual servo, bounded Cartesian IK; seed pixel supplied once",
            environment="Original Blender orchard USD meshes, UVs and image maps; see scene provenance",
            cutting="Visual jaw surrogate and gated rigid-piece detachment; no wood fracture or actuated blade CAD",
            recognition="Branch identity, axis and radius supplied by selected mesh metadata; not learned recognition",
        )

    def flush():
        (output / "report.json").write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
        (output / "frames.json").write_text(
            json.dumps(_json_safe({"fps": fps, "frames": records}), allow_nan=False) + "\n"
        )

    flush()
    shadow = None
    if os.environ.get("PRUNING_SHADOW_DIR"):
        sys.path.insert(0, str(root / "tools"))
        from shadow_depth import ShadowClient

        shadow = ShadowClient(os.environ["PRUNING_SHADOW_DIR"])
        report["learned_depth_shadow"] = shadow.model
        report["learned_depth_controls_robot"] = False
    simulation_app = None
    env = None
    try:
        from isaaclab.app import AppLauncher

        launcher = AppLauncher(headless=True, enable_cameras=True, kit_args=quality.startup_kit_args if quality else "")
        simulation_app = launcher.app
        if os.environ.get("PRUNING_FAMILY_IMPORT_CHECK") == "1":
            from pxr import Usd, UsdGeom

            asset_checks = []
            for family in ("envy", "ufo"):
                path = root / f"artifacts/trees/lpy_{family}_00000.usda"
                imported = Usd.Stage.Open(str(path))
                meshes = [p for p in imported.Traverse() if p.IsA(UsdGeom.Gprim)]
                bound = (
                    UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
                    .ComputeWorldBound(imported.GetDefaultPrim())
                    .ComputeAlignedRange()
                )
                entry = {
                    "family": family,
                    "path": str(path),
                    "renderable_primitive_count": len(meshes),
                    "meters_per_unit": UsdGeom.GetStageMetersPerUnit(imported),
                    "bounds_min": list(bound.GetMin()),
                    "bounds_max": list(bound.GetMax()),
                    "scope": "Independent USD stage opened in Isaac process; not a closed-loop tree trial",
                }
                entry["ok"] = (
                    bool(meshes)
                    and entry["meters_per_unit"] == 1.0
                    and all(math.isfinite(v) for v in (*entry["bounds_min"], *entry["bounds_max"]))
                )
                asset_checks.append(entry)
            report["family_import_checks"] = asset_checks
            flush()
            if not all(item["ok"] for item in asset_checks):
                raise RuntimeError("Family USD import preflight failed")
        simulation_app = launcher.app
        root = Path(os.environ["PRUNING_ROOT"])
        sys.path.insert(0, str(root / "source" / "isaaclab_pruning"))
        import numpy as np
        import torch
        from PIL import Image

        import carb
        import omni.replicator.core as rep
        import omni.timeline
        from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdPhysics, UsdShade

        from isaaclab.sim import RenderCfg
        from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

        from isaaclab_pruning.geometry.cut_point import CutPoint
        from isaaclab_pruning.sim.lab3_compat import apply, as_torch
        from isaaclab_pruning.sim.pose_conventions import pose_xyzw_to_wxyz, quaternion_wxyz_to_xyzw
        from isaaclab_pruning.sim.pruning_env import make_pruning_env_cls
        from isaaclab_pruning.sim.pruning_env_cfg import PruningEnvCfg
        from isaaclab_pruning.task.loop import episode_start_target

        report["lab3_compat"] = apply()
        BaseEnv = make_pruning_env_cls()
        # The original zero pose puts the tool ~64 mm above ground. Raising
        # the fixed base creates a bench-mounted cell; this is not a pass of
        # the separate original-pose hold smoke test.
        base_height = 0.70
        # CPU URDF FK places this fixture 0.35 m along the reviewed tool +Z
        # from the nonsingular initial posture. Spawn before Warp mesh caches.
        target_position = np.array(TARGET_POSITION_M)

        def material(stage, path, color):
            mat = UsdShade.Material.Define(stage, path)
            shader = UsdShade.Shader.Define(stage, f"{path}/Surface")
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.72)
            mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return mat

        def cylinder(stage, path, start, end, radius, mat, collision=True):
            start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
            axis = end - start
            length = float(np.linalg.norm(axis))
            prim = UsdGeom.Cylinder.Define(stage, path)
            prim.CreateRadiusAttr(float(radius))
            prim.CreateHeightAttr(length)
            prim.CreateAxisAttr("Z")
            xform = UsdGeom.Xformable(prim)
            xform.AddTranslateOp().Set(Gf.Vec3d(*((start + end) * 0.5)))
            xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
                Gf.Rotation(Gf.Vec3d(0, 0, 1), Gf.Vec3d(*(axis / length))).GetQuat()
            )
            UsdShade.MaterialBindingAPI.Apply(prim.GetPrim()).Bind(mat)
            if collision:
                UsdPhysics.CollisionAPI.Apply(prim.GetPrim())
            return prim

        class RenderEnv(BaseEnv):
            def _spawn_task_geometry(self):
                stage = self.sim.stage
                UsdGeom.SetStageUpAxis(stage, "Z")
                if blender_mode:
                    from isaaclab.sensors import MultiMeshRayCasterCfg

                    from isaaclab_pruning.sim.blender_demo_scene import spawn_blender_demo_scene

                    self.blender_scene = spawn_blender_demo_scene(
                        stage,
                        Path(
                            os.environ.get(
                                "PRUNING_BLENDER_SCENE_DIR", str(root / "artifacts/blender_scene/orchard_two_trees_v1")
                            )
                        ),
                        target_position_w=target_position,
                        base_height=base_height,
                        yaw_degrees=150.0,
                        component_first_vertex=int(os.environ.get("PRUNING_COMPONENT_VERTEX", "8235")),
                        target_tree_index=int(os.environ.get("PRUNING_TARGET_TREE", "0")),
                        tree_count=2,
                    )
                    selected = self.blender_scene.selected_mesh_prim_path
                    targets = [path for path in self.blender_scene.evidence["collision_mesh_paths"] if path != selected]
                    targets.append(
                        MultiMeshRayCasterCfg.RaycastTargetCfg(
                            prim_expr=str(self.blender_scene.body.GetPath()),
                            merge_prim_meshes=True,
                            track_mesh_transforms=True,
                        )
                    )
                    self.cfg.tof0_cfg.mesh_prim_paths = targets
                    self.cfg.tof1_cfg.mesh_prim_paths = targets
                    # Keep the original collision plane but hide its grid, which
                    # otherwise protrudes through the exported terrain relief.
                    UsdGeom.Imageable(stage.GetPrimAtPath("/World/ground")).MakeInvisible()
                    ground_mapping = UsdShade.Shader(
                        stage.GetPrimAtPath("/World/Orchard/_materials/mat_ground_color/Mapping")
                    )
                    ground_mapping.GetInput("scale").Set(Gf.Vec2f(500, 500))
                    sun = UsdLux.DistantLight(stage.GetPrimAtPath("/World/Orchard/sun/sun_data"))
                    sun.GetIntensityAttr().Set(2000.0)
                    self.blender_scene.evidence["presentation_overrides"] = {
                        "ground_uv_scale": [500, 500],
                        "ground_texture_repeat_m": [1, 1],
                        "exported_sun_intensity": 0.25,
                        "capture_sun_intensity": 2000.0,
                        "source_export_unchanged": True,
                        "reason": "Meter-scale ground detail and visible RTX sunlight; not Blender lighting parity",
                    }
                    metal = material(stage, "/World/Looks/Pedestal", (0.11, 0.16, 0.21))
                    cylinder(stage, "/World/Pedestal", (0, 0, 0), (0, 0, base_height), 0.14, metal)
                    dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
                    dome.CreateIntensityAttr(450.0)
                    dome.CreateColorAttr(Gf.Vec3f(0.82, 0.90, 1.0))
                    from isaaclab_pruning.sim.daylight import apply_daylight

                    self.blender_scene.evidence["daylight"] = apply_daylight(
                        sun, dome, os.environ.get("PRUNING_DAYLIGHT", "source")
                    )
                    self.blender_scene.evidence["presentation_overrides"]["capture_sun_intensity"] = float(
                        sun.GetIntensityAttr().Get()
                    )
                    self.blender_scene.evidence["sky_fill"] = (
                        "Constant blue-white dome at intensity 450; not original procedural sky"
                    )
                    return
                tree_path = "/World/envs/env_0/Tree"
                UsdGeom.Xform.Define(stage, tree_path)
                bark = material(stage, "/World/Looks/Bark", (0.25, 0.12, 0.055))
                spur = material(stage, "/World/Looks/Target", (0.88, 0.29, 0.055))
                metal = material(stage, "/World/Looks/Pedestal", (0.11, 0.16, 0.21))
                floor_material = material(stage, "/World/Looks/Floor", (0.30, 0.36, 0.29))
                # The stock ground asset has a high-contrast grid. This visual
                # plane hides that grid without changing the collision ground.
                floor = UsdGeom.Mesh.Define(stage, "/World/VisualFloor")
                floor.CreatePointsAttr([(-100, -100, 0.002), (100, -100, 0.002), (100, 100, 0.002), (-100, 100, 0.002)])
                floor.CreateFaceVertexCountsAttr([4])
                floor.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
                floor.CreateSubdivisionSchemeAttr("none")
                UsdShade.MaterialBindingAPI.Apply(floor.GetPrim()).Bind(floor_material)
                cylinder(stage, "/World/Pedestal", (0, 0, 0), (0, 0, base_height), 0.14, metal)
                trunk_base = np.array([target_position[0] - 0.20, target_position[1] + 0.25, 0.0])
                cylinder(stage, f"{tree_path}/trunk", trunk_base, trunk_base + [0.0, 0.0, 1.50], 0.055, bark)
                cylinder(
                    stage,
                    f"{tree_path}/support_branch",
                    trunk_base + [0, 0, 1.02],
                    target_position + [0, 0.025, 0.13],
                    0.026,
                    bark,
                )
                cylinder(
                    stage,
                    f"{tree_path}/target_spur",
                    target_position + [0, 0, -0.18],
                    target_position + [0, 0, 0.18],
                    0.020,
                    spur,
                )
                for index, (offset, end) in enumerate(
                    [((0, 0, 1.18), (-0.40, 0.16, 1.52)), ((0, 0, 0.94), (0.33, 0.15, 1.21))]
                ):
                    cylinder(stage, f"{tree_path}/branch_{index}", trunk_base + offset, trunk_base + end, 0.026, bark)
                # A second unmeasured tree establishes the virtual environment
                # without polluting the target fixture's ToF mesh collection.
                backdrop = "/World/Backdrop"
                UsdGeom.Xform.Define(stage, backdrop)
                cylinder(stage, f"{backdrop}/trunk", (-0.50, 1.50, 0), (-0.50, 1.50, 1.50), 0.04, bark)
                cylinder(stage, f"{backdrop}/branch0", (-0.50, 1.50, 0.85), (-0.86, 1.48, 1.3), 0.02, bark)
                cylinder(stage, f"{backdrop}/branch1", (-0.50, 1.50, 1.10), (-0.22, 1.55, 1.5), 0.02, bark)
                dome = UsdLux.DomeLight.Define(stage, "/World/Dome")
                dome.CreateIntensityAttr(450.0)
                dome.CreateColorAttr(Gf.Vec3f(0.82, 0.90, 1.0))
                key = UsdLux.DistantLight.Define(stage, "/World/Key")
                key.CreateIntensityAttr(1700.0)
                key.CreateAngleAttr(4.0)
                UsdGeom.Xformable(key).AddRotateXYZOp().Set(Gf.Vec3f(-40, 10, -35))

            def _ensure_target(self):
                scene = getattr(self, "blender_scene", None)
                self.target = episode_start_target(
                    CutPoint(
                        record_id=scene.target_id if scene else "render_fixture_vertical_spur",
                        part_name="target_spur",
                        position_w=target_position,
                        axis_w=scene.target_axis_w if scene else np.array([0.0, 0.0, 1.0]),
                        radius_m=scene.target_radius_m if scene else 0.020,
                        length_m=0.05 if scene else 0.36,
                        neighbor_count=1,
                    ),
                    batch=self.num_envs,
                    device=self.device,
                )

            def _get_dones(self):
                # No target/reset teleport is permitted in this recording.
                done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
                return done, done.clone()

            def _get_rewards(self):
                return torch.zeros(self.num_envs, device=self.device)

            def _apply_action(self):
                if getattr(self, "settling", False):
                    self.robot.set_joint_position_target(as_torch(self.robot.data.default_joint_pos))
                else:
                    super()._apply_action()

        cfg = PruningEnvCfg(observation_variant="B_tof")
        # These fields and their setting names are from the pinned v60
        # RenderCfg, not an assumed Isaac 5 renderer API. Extra render-only
        # updates below allow temporal denoising to settle without physics.
        cfg.sim.render = RenderCfg(
            rendering_mode="quality",
            antialiasing_mode=None if quality else "DLSS",
            dlss_mode=2,
            enable_dl_denoiser=True,
            enable_dlssg=False,
            enable_direct_lighting=True,
            samples_per_pixel=4,
            enable_shadows=True,
            enable_reflections=False,
            enable_global_illumination=False,
            ambient_light_intensity=0.5,
        )
        report["render_settings"] = cfg.sim.render.to_dict()
        report["render_updates_per_frame"] = RENDER_UPDATES_PER_FRAME
        cfg.robot_cfg.init_state.pos = (0.0, 0.0, base_height)
        joint_names = (
            "shoulder_pan_joint",
            "shoulder_lift_joint",
            "elbow_joint",
            "wrist_1_joint",
            "wrist_2_joint",
            "wrist_3_joint",
        )
        # Literal joint names below are derived from the promoted articulation
        # config; retain fixed-joint defaults without broad regex replacement.
        arm_joints = tuple(
            env_name
            for env_name in cfg.robot_cfg.init_state.joint_pos
            if any(env_name.endswith(name) for name in joint_names)
        )
        if len(arm_joints) != 6:
            raise RuntimeError(f"Expected six explicit UR joint defaults; found {arm_joints}.")
        for suffix, value in zip(joint_names, INITIAL_JOINT_POSITION_RAD):
            name = next(name for name in arm_joints if name.endswith(suffix))
            cfg.robot_cfg.init_state.joint_pos[name] = value
        report["initial_joint_position_rad"] = dict(cfg.robot_cfg.init_state.joint_pos)
        cfg.tof_noise_enabled = False
        report["stage"] = "construct_scene"
        flush()
        env = RenderEnv(cfg=cfg)
        env.settling = True
        env.reset(seed=7)
        report["stage"] = "settle_physical_joint_drives"
        flush()
        for _ in range(120):
            env.step(torch.zeros((1, 7), device=env.device))
        env.settling = False
        if quality:
            report["capture_quality"] = apply_capture_quality(carb.settings.get_settings(), quality)
        initial_tool = env._control_tool_pose_w().clone()
        root_pose = as_torch(env.robot.data.root_pose_w)
        tool_pos_b, tool_quat_b = subtract_frame_transforms(
            root_pose[:, :3],
            root_pose[:, 3:7],
            initial_tool[:, :3],
            quaternion_wxyz_to_xyzw(initial_tool[:, 3:7]),
        )
        hold = pose_xyzw_to_wxyz(torch.cat((tool_pos_b, tool_quat_b), dim=-1)).contiguous()
        root_rotation_w = matrix_from_quat(root_pose[:, 3:7])[0].detach().cpu().numpy()
        tool_rotation_w = matrix_from_quat(quaternion_wxyz_to_xyzw(initial_tool[:, 3:7]))[0].detach().cpu().numpy()
        # Reference the actual settled tool rather than the nominal zero-pose
        # CAD point: the deliberately unchanged 800/40 drives sag under load.
        # The fixture target is known geometry (not a CV-estimated goal).
        standoff_position_w = target_position - 0.10 * tool_rotation_w[:, 2]
        approach_delta_b = root_rotation_w.T @ (standoff_position_w - initial_tool[0, :3].detach().cpu().numpy())
        report["approach_delta_root_m"] = approach_delta_b.tolist()
        report["commanded_standoff_position_w_m"] = standoff_position_w.tolist()
        report["initial_tool_pose_wxyz"] = initial_tool.detach().cpu().tolist()
        demo = None
        if blender_mode:
            from isaaclab_pruning.sim.blender_demo_scene import gap_aligned_camera_mount
            from isaaclab_pruning.sim.vision_demo_controller import VisionPruningDemo

            initial_contact_n = max(
                float(np.linalg.norm(as_torch(sensor.data.net_forces_w).detach().cpu().numpy(), axis=-1).max())
                for sensor in env.contact_sensors.values()
            )
            report["startup_contact_force_n"] = initial_contact_n
            report["blender_scene"] = env.blender_scene.evidence
            if initial_contact_n > 5.0:
                raise RuntimeError(f"Orchard layout rejected: startup robot contact {initial_contact_n:.3f} N > 5 N")
            report["initial_piece_pose_wxyz"] = env.blender_scene.initialize_physics_tracking()
            proxy_closing_w = np.cross(tool_rotation_w[:, 2], np.asarray(env.blender_scene.target_axis_w))
            proxy_closing_w /= np.linalg.norm(proxy_closing_w)
            proxy_closing_tool = tool_rotation_w.T @ proxy_closing_w
            wrist_mount = gap_aligned_camera_mount(proxy_closing_tool)
            env.blender_scene.set_proxy_closing_axis_tool(proxy_closing_tool)
            demo = VisionPruningDemo(
                env.blender_scene.target_id,
                env.blender_scene.target_axis_w,
                env.blender_scene.target_radius_m,
                initial_tool[0].detach().cpu().numpy(),
                closing_axis_tool=proxy_closing_tool,
                photometric_normalization=photometric_normalization,
                approach=approach,
            )
            report["tracker_config"] = demo.evidence()["tracker_config"]
            report["blender_scene"] = env.blender_scene.evidence
            report["camera_mount_selection"] = {
                "method": "fixed offset perpendicular to proxy closing axis; looks along jaw opening",
                "closing_axis_tool": proxy_closing_tool.tolist(),
                "position_tool_m": list(wrist_mount),
                "radial_distance_m": 0.14,
                "hardware_calibration": False,
                "dynamic_reaiming": False,
            }
            env.blender_scene.update_tool_proxy(initial_tool[0].detach().cpu().numpy(), 0.0)
        report["target_position_m"] = target_position.tolist()
        report["bench_base_height_m"] = base_height
        report["contact_coverage"] = env.contact_state()
        if hasattr(env, "control_state"):
            report["initial_control_state"] = env.control_state()
        stage = env.sim.stage

        def camera(path, resolution):
            cam = UsdGeom.Camera.Define(stage, path)
            cam.CreateFocalLengthAttr(20.0)
            cam.CreateHorizontalApertureAttr(30.0)
            cam.CreateVerticalApertureAttr(30.0 * resolution[1] / resolution[0])
            cam.CreateClippingRangeAttr(Gf.Vec2f(0.01, 20.0))
            transform = UsdGeom.Xformable(cam).AddTransformOp()
            product = rep.create.render_product(path, resolution)
            rgb = rep.AnnotatorRegistry.get_annotator("rgb")
            rgb.attach(product)
            return transform, product, rgb

        overview_tf, overview_product, overview_rgb = camera("/World/OverviewCamera", overview_resolution)
        close_tf, close_product, close_rgb = camera("/World/CloseCamera", close_resolution)
        wrist_tf, wrist_product, wrist_rgb = camera("/World/WristCamera", wrist_resolution)
        overview_tf.Set(
            Gf.Matrix4d()
            .SetLookAt(
                Gf.Vec3d(*((4.8, -4.2, 3.2) if blender_mode else (1.75, -1.65, 1.75))),
                Gf.Vec3d(*((0.5, 1.1, 1.65) if blender_mode else (0.36, 0.47, 0.66))),
                Gf.Vec3d(0, 0, 1),
            )
            .GetInverse()
        )
        close_focus = (target_position + initial_tool[0, :3].detach().cpu().numpy()) * 0.5
        close_tf.Set(
            Gf.Matrix4d()
            .SetLookAt(Gf.Vec3d(*(close_focus + [0.55, -0.75, 0.35])), Gf.Vec3d(*close_focus), Gf.Vec3d(0, 0, 1))
            .GetInverse()
        )
        depth_ann = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth_ann.attach(wrist_product)
        target_position_tool = tool_rotation_w.T @ (target_position - initial_tool[0, :3].detach().cpu().numpy())
        if blender_mode:
            # Aim the fixed exterior camera at the midpoint of the approach
            # depth so both initial and near-mouth views remain in its FOV.
            target_position_tool[2] = 0.20
        wrist_rotation_tool = optical_rotation_from_forward(target_position_tool - np.asarray(wrist_mount))
        report["camera"] = {
            "wrist_resolution": [480, 320],
            "overview_resolution": list(overview_resolution),
            "close_resolution": [640, 480],
            "focal_length_mm": 20.0,
            "horizontal_aperture_mm": 30.0,
            "wrist_intrinsics": [[320.0, 0.0, 240.0], [0.0, 320.0, 160.0], [0.0, 0.0, 1.0]],
            "wrist_position_in_tool_m": list(wrist_mount),
            "wrist_rotation_in_tool_ros": wrist_rotation_tool.tolist(),
            "wrist_mount": (
                "simulation-defined exterior mount; fixed optical toe-in initialized using the known fixture target"
            ),
            "wrist_optical_convention": (
                "+Z forward, +X right, +Y down (ROS); fixed rotation relative to tool recorded above"
            ),
            "hardware_camera_calibrated": False,
        }

        def update_wrist():
            measured = env._control_tool_pose_w()
            rotation = matrix_from_quat(quaternion_wxyz_to_xyzw(measured[:, 3:7]))[0].detach().cpu().numpy()
            position = measured[0, :3].detach().cpu().numpy() + rotation @ np.asarray(wrist_mount)
            camera_rotation = rotation @ wrist_rotation_tool
            usd_rotation = camera_rotation @ np.diag([1.0, -1.0, -1.0])
            transform = np.eye(4)
            transform[:3, :3] = usd_rotation
            transform[:3, 3] = position
            wrist_tf.Set(Gf.Matrix4d(*transform.T.reshape(-1).tolist()))
            return position, camera_rotation

        update_wrist()
        report["stage"] = "warmup_rtx"
        flush()
        # Lab owns physical stepping. Replicator's independent orchestrator
        # waited indefinitely for its own timeline schedule in job 21201586.
        # Passive render-product annotators need only Lab's Fabric sync plus
        # bounded Kit render ticks, with timeline auto-advance disabled.
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_auto_update(False)
        rep.orchestrator.set_capture_on_play(False)

        def render_tick():
            env.sim.render()
            simulation_app.update()

        def capture_frozen_pose():
            if quality:
                # Flush Fabric once. Repeated pre-render flushes can reset PT
                # accumulation even though the physical pose did not change.
                env.sim.render(skip_app_pumping=True)
                for _ in range(quality.updates_per_capture):
                    simulation_app.update()
            else:
                for _ in range(RENDER_UPDATES_PER_FRAME):
                    render_tick()

        warmup_ready = False
        for tick in range(30):
            print(f"RTX_WARMUP_BEGIN {tick + 1}/30", flush=True)
            capture_frozen_pose() if quality else render_tick()
            shapes = {}
            for name, annotator in (("overview", overview_rgb), ("close", close_rgb), ("wrist", wrist_rgb)):
                data = np.asarray(annotator.get_data())
                shapes[name] = list(data.shape)
                if data.ndim == 3 and data.shape[-1] >= 3 and data.size:
                    Image.fromarray(data[..., :3].astype(np.uint8)).save(output / f"preview_{name}.png")
            depth_data = np.asarray(depth_ann.get_data())
            shapes["depth"] = list(depth_data.shape)
            warmup_ready = (
                shapes["overview"][:2] == list(reversed(overview_resolution))
                and shapes["close"][:2] == [480, 640]
                and shapes["wrist"][:2] == [320, 480]
                and list(depth_data.squeeze().shape) == [320, 480]
            )
            print(f"RTX_WARMUP_END {tick + 1}/30 shapes={shapes}", flush=True)
            report["warmup_ticks"] = tick + 1
            report["warmup_shapes"] = shapes
            flush()
            if warmup_ready and tick >= 7:
                break
        if not warmup_ready:
            raise RuntimeError("RTX annotators remained empty after 30 bounded Lab render ticks.")
        if quality:
            report["capture_quality_after_warmup"] = settings_readback(
                carb.settings.get_settings(), quality.carb_settings()
            )
        preview_depth = np.asarray(depth_ann.get_data(), dtype=np.float32).squeeze()
        np.save(output / "preview_depth.npy", preview_depth)
        camera_matrix = np.asarray(report["camera"]["wrist_intrinsics"])

        def optical_transform(position, rotation):
            transform = np.eye(4)
            transform[:3, :3], transform[:3, 3] = rotation, position
            return transform

        if demo:
            position, rotation = update_wrist()
            optical_target = rotation.T @ (target_position - position)
            projected = camera_matrix @ optical_target
            seed_pixel = projected[:2] / projected[2]
            preview_rgb = np.asarray(wrist_rgb.get_data())[..., :3].copy()
            visibility = seed_visibility(preview_depth, seed_pixel, optical_target[2])
            if not visibility["visible"]:
                demo.stop("initial_target_not_visible")
            report["vision_initialization"] = {
                "pixel_xy": seed_pixel.tolist(),
                "source": "Known selected component projected once; no per-frame oracle update or automatic reseeding",
                "visibility": visibility,
                "tracker": demo.initialize(preview_rgb, preview_depth, seed_pixel)
                if visibility["visible"]
                else {"state": "initialization_rejected", "reason": visibility["reason"]},
            }
            report["initial_live_vision"] = demo.observe(
                preview_rgb,
                preview_depth,
                camera_matrix,
                optical_transform(position, rotation),
                0.0,
                initial_tool[0].detach().cpu().numpy(),
            )
        report["artifact_inventory"]["preview"] = [
            "preview_overview.png",
            "preview_close.png",
            "preview_wrist.png",
            "preview_depth.npy",
        ]
        report["stage"] = "record"
        flush()
        steps_per_frame = round(1.0 / (fps * env.step_dt))
        if abs(steps_per_frame * env.step_dt - 1.0 / fps) > 1.0e-8:
            raise ValueError("Capture cadence must be an integer multiple of environment dt.")
        report["capture_dt_s"] = steps_per_frame * env.step_dt
        if quality:
            # DirectRLEnv checks cfg.sim.render_interval at every physics step.
            # RTX annotators here are explicitly rendered at capture cadence;
            # the intervening render calls do not provide controller inputs.
            # Apply after initialization so camera timing/setup is unchanged.
            report["previous_automatic_render_interval"] = env.cfg.sim.render_interval
            env.cfg.sim.render_interval = manual_capture_interval(
                int(env._sim_step_counter),
                frame_count,
                steps_per_frame * env.cfg.decimation,
            )
            report["automatic_render_interval_during_capture"] = env.cfg.sim.render_interval
        capture_timeline_start = float(timeline.get_current_time())
        capture_physics_step_start = int(env.sim.get_physics_step_count())
        stopped_reason = None
        missing_sensor_frames = 0
        last_safe_command = hold.clone()
        first_overview = None
        max_overview_change = 0.0
        for index in range(frame_count):
            phase, scheduled_offset = episode_command(index, frame_count)
            progress = scheduled_offset[1] / 0.25
            offset = approach_delta_b * progress
            # Preserve the small inspection alignment excursion while mapping
            # the nominal schedule onto a target-relative endpoint.
            offset[0] += scheduled_offset[0] + 0.09 * progress
            action = hold.clone()
            action[:, :3] += action.new_tensor(offset)
            vision_decision = None
            if demo:
                command, phase, vision_decision = demo.command(
                    env._control_tool_pose_w()[0].detach().cpu().numpy(),
                    index / fps,
                )
                command_w = hold.new_tensor(command).reshape(1, 7)
                command_p, command_q = subtract_frame_transforms(
                    root_pose[:, :3],
                    root_pose[:, 3:7],
                    command_w[:, :3],
                    quaternion_wxyz_to_xyzw(command_w[:, 3:7]),
                )
                action = pose_xyzw_to_wxyz(torch.cat((command_p, command_q), dim=-1)).contiguous()
            if stopped_reason is None and (demo is None or index == 0):
                guard_reason, missing_sensor_frames = sensor_guard(
                    np.concatenate((env.tof0.detach().cpu().numpy(), env.tof1.detach().cpu().numpy())),
                    np.concatenate((env.tof0_valid.detach().cpu().numpy(), env.tof1_valid.detach().cpu().numpy())),
                    missing_sensor_frames,
                )
                if guard_reason is not None:
                    stopped_reason = guard_reason
                    report["sensor_stop_frame"] = index
            if stopped_reason is not None:
                phase, action = "stopped_failure", last_safe_command
                if demo:
                    demo.stop(stopped_reason)
                    measured_w = env._control_tool_pose_w()
                    held_p, held_q = subtract_frame_transforms(
                        root_pose[:, :3],
                        root_pose[:, 3:7],
                        measured_w[:, :3],
                        quaternion_wxyz_to_xyzw(measured_w[:, 3:7]),
                    )
                    action = pose_xyzw_to_wxyz(torch.cat((held_p, held_q), dim=-1)).contiguous()
                    vision_decision = {"state": "hold", "reason": stopped_reason}
            render_generation_before_physics = env.sim.render_generation
            for _ in range(steps_per_frame):
                env.step(action)
            automatic_render_calls = env.sim.render_generation - render_generation_before_physics
            if demo:
                demo.command_applied(phase, vision_decision)
            tool = env._control_tool_pose_w().detach().cpu().numpy()[0]
            joints = as_torch(env.robot.data.joint_pos).detach().cpu().numpy()[0]
            if not np.isfinite(tool).all() or not np.isfinite(joints).all():
                raise RuntimeError("Nonfinite physical articulation state.")
            root_pose_now = as_torch(env.robot.data.root_pose_w)
            error_m = command_tracking_error_m(
                tool[:3],
                action[0, :3].detach().cpu().numpy(),
                root_pose_now[0, :3].detach().cpu().numpy(),
                matrix_from_quat(root_pose_now[:, 3:7])[0].detach().cpu().numpy(),
            )
            if error_m > 0.25 and stopped_reason is None:
                stopped_reason = f"tool_tracking_error_{error_m:.3f}m"
                phase = "stopped_failure"
                if hasattr(env, "control_state"):
                    report["failure_control_state"] = env.control_state()
            else:
                last_safe_command = action.clone()
            camera_position, camera_rotation = update_wrist()
            proxy_closure = 0.0
            if demo:
                proxy_closure = demo.cut_step.closure_progress if demo.cut_step else 0.0
                env.blender_scene.update_tool_proxy(tool, proxy_closure)
            capture_step_before = int(env.sim.get_physics_step_count())
            capture_time_before = float(timeline.get_current_time())
            capture_frozen_pose()
            capture_step_after = int(env.sim.get_physics_step_count())
            capture_time_after = float(timeline.get_current_time())
            overview = np.asarray(overview_rgb.get_data())[..., :3]
            close = np.asarray(close_rgb.get_data())[..., :3]
            wrist = np.asarray(wrist_rgb.get_data())[..., :3]
            depth = np.asarray(depth_ann.get_data(), dtype=np.float32).squeeze()
            if (
                overview.shape != (*reversed(overview_resolution), 3)
                or close.shape != (480, 640, 3)
                or wrist.shape != (320, 480, 3)
                or depth.shape != (320, 480)
            ):
                raise RuntimeError(
                    f"Unexpected rendered shapes: {overview.shape}, {close.shape}, {wrist.shape}, {depth.shape}"
                )
            Image.fromarray(overview.astype(np.uint8)).save(frame_dir / f"overview_{index:05d}.png")
            Image.fromarray(close.astype(np.uint8)).save(frame_dir / f"close_{index:05d}.png")
            Image.fromarray(wrist.astype(np.uint8)).save(frame_dir / f"wrist_{index:05d}.png")
            np.save(frame_dir / f"depth_{index:05d}.npy", depth)
            if first_overview is None:
                first_overview = overview.astype(np.float32)
            max_overview_change = max(
                max_overview_change, float(np.abs(overview.astype(np.float32) - first_overview).mean())
            )
            sensor_state = env.tof_state()
            contact = env.contact_state()
            contact_sensors = getattr(env, "contact_sensors", {"legacy_contact": env.contact})
            forces = {
                name: as_torch(sensor.data.net_forces_w).detach().cpu().tolist()
                for name, sensor in contact_sensors.items()
            }
            contact_force_n = max(
                float(np.linalg.norm(np.asarray(values), axis=-1).max()) for values in forces.values()
            )
            live_vision = None
            detach_requested = False
            if demo:
                # Validate the freshly captured range sample before authorizing
                # closure/detachment, not on the next command frame. For live
                # mode the pre-step guard runs only on the initial sample.
                fresh_guard_reason, missing_sensor_frames = sensor_guard(
                    np.concatenate((env.tof0.detach().cpu().numpy(), env.tof1.detach().cpu().numpy())),
                    np.concatenate((env.tof0_valid.detach().cpu().numpy(), env.tof1_valid.detach().cpu().numpy())),
                    missing_sensor_frames,
                )
                if fresh_guard_reason and stopped_reason is None:
                    stopped_reason = fresh_guard_reason
                    report["sensor_stop_frame"] = index
                if stopped_reason:
                    demo.stop(stopped_reason)
                live_vision = demo.observe(
                    wrist.copy(),
                    depth.copy(),
                    camera_matrix,
                    optical_transform(camera_position, camera_rotation),
                    (index + 1) / fps,
                    tool,
                    hazard_contact=contact_force_n > 5.0,
                )
                if demo.cut_step.detach_event:
                    detach_requested = env.blender_scene.detach()
                if demo.cut_step.phase == "stopped" and stopped_reason is None:
                    stopped_reason = demo.cut_step.stopped_reason
            record = {
                "index": index,
                "time_s": (index + 1) * steps_per_frame * env.step_dt,
                "timeline_time_s": float(timeline.get_current_time()),
                "timeline_elapsed_s": float(timeline.get_current_time()) - capture_timeline_start,
                "physics_step_count": int(env.sim.get_physics_step_count()),
                "capture_physics_steps_advanced": capture_step_after - capture_step_before,
                "automatic_render_calls_during_physics": automatic_render_calls,
                "capture_timeline_advanced_s": capture_time_after - capture_time_before,
                "physics_elapsed_s": (int(env.sim.get_physics_step_count()) - capture_physics_step_start)
                * env.physics_dt,
                "phase": phase,
                "sensor_guard_consecutive_missing_frames": missing_sensor_frames,
                "sensor_stop_reason": stopped_reason
                if stopped_reason and stopped_reason.startswith(("tof_", "both_tof"))
                else None,
                "tool_position_m": tool[:3].tolist(),
                "tool_pose_wxyz": tool.tolist(),
                "command_pose_root_wxyz": action.detach().cpu().tolist()[0],
                "target_position_m": target_position.tolist(),
                "target_distance_m": float(np.linalg.norm(tool[:3] - target_position)),
                "joint_position_rad": joints.tolist(),
                "joint_velocity_rad_s": as_torch(env.robot.data.joint_vel).detach().cpu().tolist()[0],
                "tracking_error_m": error_m,
                "tof_left_m": env.tof0.detach().cpu().tolist()[0],
                "tof_right_m": env.tof1.detach().cpu().tolist()[0],
                "tof_left_valid": env.tof0_valid.detach().cpu().tolist()[0],
                "tof_right_valid": env.tof1_valid.detach().cpu().tolist()[0],
                "tof_state": sensor_state,
                "contact_force_n": contact_force_n,
                "contact_forces_w_n": forces,
                "contact": contact,
                "wrist_position_w_m": camera_position.tolist(),
                "wrist_rotation_w_ros": camera_rotation.tolist(),
                "rgb_std": float(overview.std()),
                "close_rgb_std": float(close.std()),
                "wrist_rgb_std": float(wrist.std()),
                "depth_finite_fraction": float((np.isfinite(depth) & (depth > 0)).mean()),
            }
            if demo:
                record.update(
                    live_vision=live_vision,
                    visual_servo_decision=vision_decision,
                    controller_source_frame_index=index - 1,
                    visual_jaw_closure_progress=proxy_closure,
                    detachment_requested_after_capture=detach_requested,
                    selected_piece_pose_wxyz=env.blender_scene.measured_piece_pose_wxyz(),
                )
            if shadow is not None:
                tracker = (live_vision or {}).get("measurement", {})
                pixel = tracker.get("pixel_xy") if tracker.get("state") == "tracking" else None
                record["learned_depth_shadow"] = shadow.observe(index, wrist, depth, pixel)
            records.append(record)
            if index in (0, frame_count // 2, frame_count - 1) and hasattr(env, "control_state"):
                report.setdefault("control_snapshots", {})[str(index)] = env.control_state()
            report["frame_count"] = len(records)
            if index % 10 == 0:
                print(
                    f"CAPTURE {index + 1}/{frame_count} phase={phase} tool={tool[:3]} error={error_m:.4f}m", flush=True
                )
                flush()
        positions = np.asarray([record["tool_position_m"] for record in records])
        displacements = np.linalg.norm(positions - positions[0], axis=1)
        distances = np.asarray([record["target_distance_m"] for record in records])
        tof_variation = {}
        for name, key in (("tof0", "tof_left_m"), ("tof1", "tof_right_m")):
            array = np.asarray([record[key] for record in records], dtype=float)
            valid = np.isfinite(array)
            medians = [float(np.median(frame[np.isfinite(frame)])) for frame in array if np.isfinite(frame).any()]
            tof_variation[name] = {
                "valid_samples": int(valid.sum()),
                "valid_fraction": float(valid.mean()),
                "median_range_span_m": max(medians) - min(medians) if medians else 0.0,
            }
        checks = {
            "frames_complete": len(records) == frame_count,
            "overview_nonblank": min(record["rgb_std"] for record in records) > 3.0,
            "close_nonblank": min(record["close_rgb_std"] for record in records) > 3.0,
            "wrist_nonblank": min(record["wrist_rgb_std"] for record in records) > 3.0,
            "metric_depth_finite": max(record["depth_finite_fraction"] for record in records) > 0.05,
            "robot_physically_moved": float(displacements.max()) > 0.020,
            "overview_changed": max_overview_change > 0.10,
            "capture_did_not_advance_physics": all(
                record["capture_physics_steps_advanced"] == 0 and abs(record["capture_timeline_advanced_s"]) < 1.0e-8
                for record in records
            ),
            "both_tof_live": all(item["valid_samples"] > 32 for item in tof_variation.values()),
            "both_tof_changed": all(item["median_range_span_m"] > 0.005 for item in tof_variation.values()),
        }
        approach = bool(distances[0] - distances.min() > 0.020)
        retreat = bool(distances[-1] - distances.min() > 0.020)
        report.update(
            checks=checks,
            ok=all(checks.values()),
            rendering_ok=all(
                checks[key] for key in ("frames_complete", "overview_nonblank", "wrist_nonblank", "metric_depth_finite")
            ),
            stage="complete",
            task_outcome=(
                "approach_inspect_retreat_no_cut"
                if approach and retreat and stopped_reason is None
                else "stopped_failure"
            ),
            stopped_reason=stopped_reason,
            tof_variation=tof_variation,
            metrics={
                "max_tool_displacement_m": float(displacements.max()),
                "initial_target_distance_m": float(distances[0]),
                "minimum_target_distance_m": float(distances.min()),
                "final_target_distance_m": float(distances[-1]),
                "retreat_return_error_m": float(displacements[-1]),
                "maximum_overview_mean_absolute_change_8bit": max_overview_change,
                "approach_observed": approach,
                "retreat_observed": retreat,
            },
        )
        if demo:
            report["final_live_vision"] = demo.evidence()
            report["metrics"]["retreat_return_error_m"] = float(
                np.linalg.norm(positions[-1] - initial_tool[0, :3].detach().cpu().numpy())
            )
            report["metrics"]["return_reference"] = "Measured initial home before the first vision command"
            piece_positions = np.asarray([record["selected_piece_pose_wxyz"][:3] for record in records])
            initial_piece = np.asarray(report["initial_piece_pose_wxyz"][:3])
            drop_m = float(initial_piece[2] - piece_positions[:, 2].min())
            report["metrics"]["selected_piece_maximum_drop_m"] = drop_m
            report["metrics"]["selected_piece_drop_observed"] = bool(env.blender_scene.detached and drop_m > 0.02)
            report["task_outcome"] = (
                "vision_guided_simulated_detachment_and_retreat"
                if env.blender_scene.detached and drop_m > 0.02 and retreat and stopped_reason is None
                else "simulated_detachment_requested"
                if env.blender_scene.detached
                else "vision_stopped_failure"
                if stopped_reason
                else "vision_approach_incomplete"
            )
            report["vision_command_count"] = demo.vision_command_count
        if quality:
            report["checks"]["no_unused_intermediate_renders"] = all(
                record["automatic_render_calls_during_physics"] == 0 for record in records
            )
            report["ok"] = all(report["checks"].values())
        if hasattr(env, "control_state"):
            report["final_control_state"] = env.control_state()
        if quality:
            report["capture_quality_at_end"] = settings_readback(carb.settings.get_settings(), quality.carb_settings())
        stage.GetRootLayer().Export(str(output / "scene.usda"))
        flush()
        (output / "manifest.json").write_text(json.dumps(_json_safe(report), indent=2, allow_nan=False) + "\n")
        print(json.dumps(_json_safe(report), indent=2, allow_nan=False), flush=True)
        return 0 if report["ok"] else 1
    except Exception as error:  # noqa: BLE001 - preserve the actual failure beside any partial frames.
        report["error"] = repr(error)
        report["traceback"] = traceback.format_exc()
        report["task_outcome"] = "runtime_failure"
        flush()
        print(report["traceback"], flush=True)
        return 1
    finally:
        if env is not None:
            with contextlib.suppress(Exception):
                env.close()
        if simulation_app is not None:
            simulation_app.close()


if __name__ == "__main__":
    sys.exit(main())
