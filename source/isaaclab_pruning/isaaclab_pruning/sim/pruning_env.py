"""Isaac Lab DirectRLEnv for dormant-spur pruning.

Importing this module requires Isaac Lab. Core reward, observation, ToF, and
success logic live in isaac-free packages and are unit-tested without Sim.
"""

from __future__ import annotations

ISAAC_IMPORT_ERROR = (
    "PruningEnv requires Isaac Lab on the v60 stack (Isaac Sim 6.0.0.1). "
    "Use isaaclab_pruning.task, .policies, and .baselines from a plain interpreter."
)


def require_isaaclab() -> None:
    try:
        import isaaclab  # noqa: F401
    except ImportError as error:
        raise RuntimeError(ISAAC_IMPORT_ERROR) from error


def make_pruning_env_cls():  # noqa: C901 - class is built lazily behind the Isaac import gate.
    """Build the DirectRLEnv subclass only after Isaac Lab is importable."""
    require_isaaclab()

    from isaaclab_pruning.sim.lab3_compat import apply as apply_lab3
    from isaaclab_pruning.sim.lab3_compat import as_torch

    apply_lab3()

    from collections.abc import Sequence

    import numpy as np
    import torch

    from pxr import UsdPhysics

    from isaaclab.assets import Articulation
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.envs import DirectRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor, MultiMeshRayCasterCamera
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_from_usd, spawn_ground_plane
    from isaaclab.sim.spawners.shapes import CuboidCfg
    from isaaclab.sim.utils import resolve_prim_pose, resolve_prim_scale

    from isaaclab_pruning.geometry.cut_point import CutPoint
    from isaaclab_pruning.geometry.cutter import cutter_boxes_from_spec
    from isaaclab_pruning.geometry.wood import nearby_wood_in_failure_zone
    from isaaclab_pruning.policies.observations import ObservationVariant, build_observation, proprioception
    from isaaclab_pruning.robot import (
        compose_physics_body_to_control_tool_pose,
        load_ur5e_pruner_spec,
        point_offset_in_jacobian_frame,
        repository_root,
        shift_spatial_jacobian_to_point,
    )
    from isaaclab_pruning.sensors.tof_noise import ToFNoiseConfig, ToFStatus, apply_tof_noise
    from isaaclab_pruning.sensors.tof_raycaster import (
        TOF_SITE_PRIM_EXPRS,
        VL53L8CX_DATA_TYPE,
    )
    from isaaclab_pruning.sim.pose_conventions import (
        pose_wxyz_to_xyzw,
        pose_xyzw_to_wxyz,
        quaternion_wxyz_to_xyzw,
    )
    from isaaclab_pruning.sim.tof_smoke_geometry import TOF_SMOKE_TARGET
    from isaaclab_pruning.task.loop import episode_start_target
    from isaaclab_pruning.task.reward import dense_pruning_reward
    from isaaclab_pruning.task.success import evaluate_cut_success

    from .pruning_env_cfg import PruningEnvCfg

    class PruningEnv(DirectRLEnv):
        cfg: PruningEnvCfg

        def __init__(self, cfg: PruningEnvCfg, render_mode: str | None = None, **kwargs):
            self.spec = load_ur5e_pruner_spec()
            self.variant = ObservationVariant(cfg.observation_variant)
            self._cut_cache = None
            super().__init__(cfg, render_mode, **kwargs)
            # Isaac Lab 3 creates PhysX articulation views during the
            # DirectRLEnv simulation reset, after _setup_scene returns.
            # Resolving joint/body names earlier dereferences an uninitialized
            # Articulation._root_view (job 21153411).
            self.robot_entity_cfg.resolve(self.scene)
            self.ik_controller.reset()

        def _setup_scene(self):
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
            tree_usd = repository_root() / "artifacts/trees/lpy_envy_00000.usda"
            if tree_usd.is_file():
                spawn_from_usd(
                    prim_path="/World/envs/env_0/Tree",
                    cfg=UsdFileCfg(usd_path=str(tree_usd.resolve())),
                    translation=(0.0, 1.0, 0.0),
                )
            self._tof_smoke_target_prim = None
            if self.cfg.tof_smoke_target_enabled:
                if int(self.cfg.scene.num_envs) != 1:
                    raise ValueError("The deterministic ToF smoke target is validated only for one environment.")
                expected_targets = [TOF_SMOKE_TARGET.prim_expr]
                if any(
                    list(sensor_cfg.mesh_prim_paths) != expected_targets
                    for sensor_cfg in (self.cfg.tof0_cfg, self.cfg.tof1_cfg)
                ):
                    raise ValueError(
                        "The ToF smoke target was enabled without routing both ray-casters exclusively to it."
                    )
                target_cfg = CuboidCfg(size=tuple(self.cfg.tof_smoke_target_size_m))
                self._tof_smoke_target_prim = target_cfg.func(
                    TOF_SMOKE_TARGET.prim_path,
                    target_cfg,
                    translation=tuple(self.cfg.tof_smoke_target_position_w_m),
                )
            self.robot = Articulation(self.cfg.robot_cfg)
            self.contact = ContactSensor(self.cfg.contact_cfg)
            self.tof_sensors = {
                "tof0": MultiMeshRayCasterCamera(self.cfg.tof0_cfg),
                "tof1": MultiMeshRayCasterCamera(self.cfg.tof1_cfg),
            }
            self.scene.clone_environments(copy_from_source=False)
            if self.device == "cpu":
                self.scene.filter_collisions(global_prim_paths=["/World/ground"])
            self.scene.articulations["robot"] = self.robot
            self.scene.sensors["arm_contact"] = self.contact
            self.scene.sensors.update(self.tof_sensors)
            self.robot_entity_cfg = SceneEntityCfg(
                name="robot",
                joint_names=[joint.name for joint in self.spec.arm_joints],
                body_names=[self.spec.physics_eef_body],
            )
            self._ensure_target()
            self._initialize_observation_buffers()
            self.ik_controller = DifferentialIKController(
                cfg=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=self.spec.ik_relative_mode,
                    ik_method=self.spec.ik_method,
                    ik_params={"lambda_val": self.spec.ik_lambda},
                ),
                num_envs=self.cfg.num_envs,
                device=self.device,
            )
            self.actions = torch.zeros((self.num_envs, self.spec.action_dim), device=self.device)

        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            self.actions[:] = actions
            self._cut_cache = None

        def _apply_action(self) -> None:
            # Public absolute-tool commands follow the robot spec's wxyz;
            # the pinned Lab 3 differential-IK controller consumes xyzw.
            self.ik_controller.set_command(pose_wxyz_to_xyzw(self.actions))
            eef_idx = self.robot_entity_cfg.body_ids[0]
            jacobi_idx = eef_idx - 1 if self.robot.is_fixed_base else eef_idx
            # Lab 3's backend-neutral articulation data exposes a ProxyArray;
            # use its Torch view instead of PhysX's raw Warp array.  The link
            # Jacobian is also referenced at body_pose_w's link origin, which
            # is the point shifted below to the reviewed control-tool frame.
            jacobians = as_torch(self.robot.data.body_link_jacobian_w)[
                :, jacobi_idx, :, self.robot_entity_cfg.joint_ids
            ]
            physics_body_pose_w = as_torch(self.robot.data.body_pose_w)[:, eef_idx]
            root_pose_w = as_torch(self.robot.data.root_pose_w)
            from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

            physics_body_pos_b, physics_body_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                physics_body_pose_w[:, 0:3],
                physics_body_pose_w[:, 3:7],
            )
            physics_body_pose_b = pose_xyzw_to_wxyz(torch.cat((physics_body_pos_b, physics_body_quat_b), dim=-1))
            # PhysX exposes the geometric Jacobian in world coordinates while
            # DifferentialIKController consumes it with a root-frame pose.
            # Rotate both spatial blocks before shifting the reference point.
            root_rotation_bw = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
            jacobians = jacobians.clone()
            jacobians[:, :3, :] = torch.bmm(root_rotation_bw, jacobians[:, :3, :])
            jacobians[:, 3:6, :] = torch.bmm(root_rotation_bw, jacobians[:, 3:6, :])
            control_tool_pose_b = compose_physics_body_to_control_tool_pose(
                physics_body_pose_b,
                self.spec.control_tool_translation_in_physics_body_m,
                self.spec.control_tool_quaternion_wxyz_in_physics_body,
            )
            tool_offset_b = point_offset_in_jacobian_frame(
                physics_body_pose_b[:, 3:7],
                self.spec.control_tool_translation_in_physics_body_m,
            )
            control_tool_jacobians = shift_spatial_jacobian_to_point(jacobians, tool_offset_b)
            joint_pos_des = self.ik_controller.compute(
                control_tool_pose_b[:, 0:3],
                quaternion_wxyz_to_xyzw(control_tool_pose_b[:, 3:7]),
                control_tool_jacobians,
                as_torch(self.robot.data.joint_pos)[:, self.robot_entity_cfg.joint_ids],
            )
            self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)

        def _arm_collision(self) -> torch.Tensor:
            forces = as_torch(self.contact.data.net_forces_w)
            if forces is None:
                raise RuntimeError("ContactSensor.net_forces_w is None; PhysX contact state was not read.")
            return torch.linalg.vector_norm(forces, dim=-1).amax(dim=-1) > float(
                self.cfg.contact_cfg.force_threshold or 1.0
            )

        def _cut_success(self):
            if self._cut_cache is None:
                self._cut_cache = self._compute_cut_success()
            return self._cut_cache

        def _compute_cut_success(self):
            eef_pose = self._control_tool_pose_w()
            mouth, failure = cutter_boxes_from_spec(
                eef_pose_w=eef_pose,
                mouth_half_extents=self.spec.mouth_half_extents_m,
                failure_half_extents=self.spec.failure_half_extents_m,
                failure_offset_eef=self.spec.failure_offset_m,
                mouth_offset_eef=self.spec.mouth_offset_m,
            )
            other_wood = None
            if hasattr(self, "wood_centroids"):
                other_wood = nearby_wood_in_failure_zone(
                    self.wood_centroids,
                    self.wood_axes,
                    self.wood_lengths,
                    failure,
                    exclude_mask=self.wood_target_mask,
                )
            closing_axis = eef_pose.new_tensor([1.0, 0.0, 0.0]).expand(self.num_envs, 3)
            closing_axis = torch.matmul(mouth.rotation_bw, closing_axis.unsqueeze(-1)).squeeze(-1)
            return evaluate_cut_success(
                branch_centroid_w=self.target.position_w,
                branch_axis_w=self.target.axis_w,
                branch_length_m=self.target.length_m,
                cutter_closing_axis_w=closing_axis,
                mouth_box=mouth,
                failure_box=failure,
                arm_collision=self._arm_collision(),
                other_wood_in_failure_zone=other_wood,
                perpendicularity_tolerance_deg=self.spec.perpendicularity_tolerance_deg,
            )

        def _get_rewards(self) -> torch.Tensor:
            eef = self._control_tool_pose_w()[:, 0:3]
            return dense_pruning_reward(eef, self.target.position_w, self._cut_success(), self.actions)

        def _get_dones(self):
            success = self._cut_success().success
            time_out = self.episode_length_buf >= self.max_episode_length
            return success, time_out

        def _initialize_observation_buffers(self) -> None:
            """Allocate observations; ToF starts invalid until a live frame arrives.

            Flow and metric depth remain explicit Phase-3 placeholders. Unlike
            the old smoke constants, both ToF channels are populated only from
            registered scene sensors.
            """
            n = self.num_envs
            th, tw = self.cfg.tof_hw
            fh, fw = self.cfg.flow_hw
            mh, mw = self.cfg.metric_hw
            self.flow = torch.zeros((n, fh, fw, 2), device=self.device)
            self.metric_student = torch.full((n, mh, mw), 1.20, device=self.device)
            self.metric_var = torch.full((n, mh, mw), 1e-2, device=self.device)
            self._tof_noise_cfg = ToFNoiseConfig(
                min_range_m=float(self.cfg.tof_min_range_m),
                max_range_m=float(self.cfg.tof_max_range_m),
                min_sigma_m=0.003 if self.cfg.tof_noise_enabled else 0.0,
                range_sigma_fraction=0.03 if self.cfg.tof_noise_enabled else 0.0,
                dropout_probability=0.05 if self.cfg.tof_noise_enabled else 0.0,
                thin_dropout_probability=0.30 if self.cfg.tof_noise_enabled else 0.0,
            )
            self.tof0 = torch.empty((n, th, tw), device=self.device)
            self.tof1 = torch.empty((n, th, tw), device=self.device)
            self.tof0_raw = torch.empty((n, th, tw), device=self.device)
            self.tof1_raw = torch.empty((n, th, tw), device=self.device)
            self.tof0_valid = torch.empty((n, th, tw), dtype=torch.bool, device=self.device)
            self.tof1_valid = torch.empty((n, th, tw), dtype=torch.bool, device=self.device)
            self.tof0_var = torch.empty((n, th, tw), device=self.device)
            self.tof1_var = torch.empty((n, th, tw), device=self.device)
            self.tof0_status = torch.empty((n, th, tw), dtype=torch.int8, device=self.device)
            self.tof1_status = torch.empty((n, th, tw), dtype=torch.int8, device=self.device)
            self._tof_last_frame = {
                name: torch.full((n,), -1, dtype=torch.int64, device=self.device) for name in self.tof_sensors
            }
            self._tof_source = "live_multi_mesh_ray_caster"
            self._reset_tof_buffers()

        def _reset_tof_buffers(self, env_ids: torch.Tensor | slice | None = None) -> None:
            ids = slice(None) if env_ids is None else env_ids
            for name in self.tof_sensors:
                ranges = getattr(self, name)
                raw = getattr(self, f"{name}_raw")
                valid = getattr(self, f"{name}_valid")
                variance = getattr(self, f"{name}_var")
                status = getattr(self, f"{name}_status")
                ranges[ids] = float("nan")
                raw[ids] = float("nan")
                valid[ids] = False
                variance[ids] = float("inf")
                status[ids] = int(ToFStatus.OUT_OF_RANGE)
                self._tof_last_frame[name][ids] = -1

        def _refresh_live_tof(self) -> None:
            """Consume each new 15 Hz ray-caster frame exactly once.

            The environment steps at 60 Hz. Frame counters prevent stochastic
            noise/dropout from being re-sampled three extra times while the
            underlying 15 Hz range image is unchanged.
            """
            expected_shape = (self.num_envs, *tuple(self.cfg.tof_hw), 1)
            refreshed = False
            for name, sensor in self.tof_sensors.items():
                data = sensor.data
                output = data.output
                if output is None or VL53L8CX_DATA_TYPE not in output:
                    raise RuntimeError(f"{name} produced no {VL53L8CX_DATA_TYPE!r} output.")
                raw_hwc = as_torch(output[VL53L8CX_DATA_TYPE])
                if tuple(raw_hwc.shape) != expected_shape:
                    raise RuntimeError(f"{name} output shape {tuple(raw_hwc.shape)} != expected {expected_shape}.")
                frame = as_torch(sensor.frame).reshape(-1).to(device=self.device, dtype=torch.int64)
                if frame.shape != self._tof_last_frame[name].shape:
                    raise RuntimeError(
                        f"{name} frame shape {tuple(frame.shape)} != {tuple(self._tof_last_frame[name].shape)}."
                    )
                changed = frame != self._tof_last_frame[name]
                if not bool(changed.any().item()):
                    continue
                ids = changed.nonzero(as_tuple=False).squeeze(-1)
                refreshed = True
                raw = raw_hwc[ids, ..., 0]
                observation = apply_tof_noise(raw, config=self._tof_noise_cfg)
                getattr(self, f"{name}_raw")[ids] = raw
                getattr(self, name)[ids] = observation.range_m
                getattr(self, f"{name}_valid")[ids] = observation.valid
                variance = observation.variance_m2
                if not self.cfg.tof_noise_enabled:
                    # Zero noise is useful for a geometry smoke, but a zero
                    # variance would make valid ToF disappear from fusion.
                    variance = torch.where(
                        observation.valid,
                        torch.full_like(variance, 1.0e-12),
                        torch.full_like(variance, float("inf")),
                    )
                getattr(self, f"{name}_var")[ids] = variance
                getattr(self, f"{name}_status")[ids] = observation.status
                self._tof_last_frame[name][ids] = frame[ids]
            if refreshed:
                self._tof_source = "live_multi_mesh_ray_caster"

        def _require_buffer(self, name: str):
            if not hasattr(self, name):
                raise RuntimeError(
                    f"Observation buffer {name!r} was never filled. "
                    "Cameras on the stage are not observations (BHL 194/194/194)."
                )
            return getattr(self, name)

        def _control_tool_pose_w(self) -> torch.Tensor:
            """Return the control-tool world pose as xyz + wxyz for core geometry."""
            physics_body_pose_w = as_torch(self.robot.data.body_pose_w)[:, self.robot_entity_cfg.body_ids[0], 0:7]
            return compose_physics_body_to_control_tool_pose(
                pose_xyzw_to_wxyz(physics_body_pose_w),
                self.spec.control_tool_translation_in_physics_body_m,
                self.spec.control_tool_quaternion_wxyz_in_physics_body,
            )

        def _proprio(self) -> torch.Tensor:
            joints = self.robot_entity_cfg.joint_ids
            eef = self._control_tool_pose_w()
            return proprioception(
                as_torch(self.robot.data.joint_pos)[:, joints],
                as_torch(self.robot.data.joint_vel)[:, joints],
                eef,
            )

        def observation_for_variant(self, variant: ObservationVariant) -> torch.Tensor:
            self._refresh_live_tof()
            return build_observation(
                variant,
                goal_w=self.target.position_w,
                proprio=self._proprio(),
                flow_hw2=self._require_buffer("flow"),
                tof0=self._require_buffer("tof0"),
                tof1=self._require_buffer("tof1"),
                tof0_valid=self._require_buffer("tof0_valid"),
                tof1_valid=self._require_buffer("tof1_valid"),
                tof0_var=self._require_buffer("tof0_var"),
                tof1_var=self._require_buffer("tof1_var"),
                metric_depth=self._require_buffer("metric_student"),
                metric_var=self._require_buffer("metric_var"),
            )

        def _get_observations(self) -> dict:
            return {"policy": self.observation_for_variant(self.variant)}

        def _ensure_target(self) -> None:
            cut = CutPoint(
                record_id="smoke",
                part_name="spur_smoke",
                position_w=np.array([0.4, 1.4, 0.5], dtype=np.float64),
                axis_w=np.array([0.0, 1.0, 0.0], dtype=np.float64),
                radius_m=0.01,
                length_m=0.1,
                neighbor_count=0,
            )
            self.target = episode_start_target(cut, batch=self.num_envs, device=self.device)

        def _reset_idx(self, env_ids: Sequence[int] | None):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            super()._reset_idx(env_ids)
            ids = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device)
            self.robot.write_joint_state_to_sim(
                as_torch(self.robot.data.default_joint_pos)[ids],
                as_torch(self.robot.data.default_joint_vel)[ids],
                env_ids=ids,
            )
            self._ensure_target()
            self._reset_tof_buffers(ids)
            self._cut_cache = None

        def apply_tof_observation(
            self, ranges0, ranges1, radii0=None, radii1=None, config: ToFNoiseConfig | None = None
        ):
            """Inject debug ToF tables until the next live ray-caster frame."""
            noisy0 = apply_tof_noise(ranges0, hit_radii_m=radii0, config=config)
            noisy1 = apply_tof_noise(ranges1, hit_radii_m=radii1, config=config)
            self.tof0, self.tof1 = noisy0.range_m, noisy1.range_m
            self.tof0_raw, self.tof1_raw = ranges0.clone(), ranges1.clone()
            self.tof0_valid, self.tof1_valid = noisy0.valid, noisy1.valid
            self.tof0_var, self.tof1_var = noisy0.variance_m2, noisy1.variance_m2
            self.tof0_status, self.tof1_status = noisy0.status, noisy1.status
            for name, sensor in self.tof_sensors.items():
                self._tof_last_frame[name][:] = (
                    as_torch(sensor.frame).reshape(-1).to(device=self.device, dtype=torch.int64)
                )
            self._tof_source = "external_debug_injection"

        def _range_stats(self, ranges: torch.Tensor, valid: torch.Tensor | None = None) -> dict:
            mask = torch.isfinite(ranges)
            if valid is not None:
                mask &= valid
            selected = ranges[mask]
            stats = {
                "shape": list(ranges.shape),
                "finite_fraction": float(torch.isfinite(ranges).to(dtype=torch.float32).mean().item()),
                "valid_fraction": float(mask.to(dtype=torch.float32).mean().item()),
                "min_m": None,
                "median_m": None,
                "max_m": None,
            }
            if selected.numel():
                stats.update(
                    min_m=float(selected.min().item()),
                    median_m=float(selected.median().item()),
                    max_m=float(selected.max().item()),
                )
            return stats

        def tof_state(self) -> dict:
            """Return JSON-safe live sensor provenance, poses, frames, and ranges."""
            self._refresh_live_tof()
            sensors = {}
            for name, sensor in self.tof_sensors.items():
                data = sensor.data
                frame = as_torch(sensor.frame).reshape(-1)
                sensors[name] = {
                    "class": type(sensor).__name__,
                    "tracking_prim_expr": sensor.cfg.prim_path,
                    "logical_site_prim_expr": TOF_SITE_PRIM_EXPRS[name],
                    "mesh_prim_paths": [
                        target if isinstance(target, str) else target.prim_expr for target in sensor.cfg.mesh_prim_paths
                    ],
                    "update_period_s": float(sensor.cfg.update_period),
                    "frame": frame.detach().cpu().tolist(),
                    "consumed_frame": self._tof_last_frame[name].detach().cpu().tolist(),
                    "position_w_m": as_torch(data.pos_w).detach().cpu().tolist(),
                    "quaternion_w_ros_xyzw": as_torch(data.quat_w_ros).detach().cpu().tolist(),
                    "raw": self._range_stats(getattr(self, f"{name}_raw")),
                    "observation": self._range_stats(getattr(self, name), getattr(self, f"{name}_valid")),
                }
            return {
                "source": self._tof_source,
                "noise_enabled": bool(self.cfg.tof_noise_enabled),
                "range_limits_m": [
                    float(self.cfg.tof_min_range_m),
                    float(self.cfg.tof_max_range_m),
                ],
                "sensors": sensors,
            }

        def tof_smoke_target_state(self) -> dict:
            """Return authored and actual stage state for the opt-in smoke wall."""
            state = TOF_SMOKE_TARGET.manifest()
            state.update(
                enabled=bool(self.cfg.tof_smoke_target_enabled),
                position_w_m=list(self.cfg.tof_smoke_target_position_w_m),
                size_m=list(self.cfg.tof_smoke_target_size_m),
                sensor_mesh_prim_paths={
                    name: [
                        target if isinstance(target, str) else target.prim_expr for target in sensor.cfg.mesh_prim_paths
                    ]
                    for name, sensor in self.tof_sensors.items()
                },
            )
            prim = self._tof_smoke_target_prim
            if prim is None:
                state.update(
                    stage_prim_valid=False,
                    stage_prim_type=None,
                    geometry_prim_path=None,
                    geometry_prim_type=None,
                    collision_api_applied=None,
                    rigid_body_api_applied=None,
                )
            else:
                geometry_path = f"{TOF_SMOKE_TARGET.prim_path}/geometry/mesh"
                geometry_prim = prim.GetStage().GetPrimAtPath(geometry_path)
                actual_position, actual_orientation = resolve_prim_pose(prim)
                cube_size = float(geometry_prim.GetAttribute("size").Get())
                geometry_scale = resolve_prim_scale(geometry_prim)
                state.update(
                    stage_prim_valid=bool(prim.IsValid()),
                    stage_prim_type=str(prim.GetTypeName()),
                    geometry_prim_path=geometry_path,
                    geometry_prim_type=(str(geometry_prim.GetTypeName()) if geometry_prim.IsValid() else None),
                    actual_position_w_m=[float(value) for value in actual_position],
                    actual_orientation_xyzw=[float(value) for value in actual_orientation],
                    actual_size_m=[cube_size * float(value) for value in geometry_scale],
                    collision_api_applied=bool(geometry_prim.HasAPI(UsdPhysics.CollisionAPI)),
                    rigid_body_api_applied=bool(prim.HasAPI(UsdPhysics.RigidBodyAPI)),
                )
            return state

        def contact_state(self) -> dict:
            forces = as_torch(self.contact.data.net_forces_w)
            if forces is None:
                raise RuntimeError("PhysX contact forces were not available.")
            tensor = forces.detach()
            return {
                "shape": list(tensor.shape),
                "max_abs": float(tensor.abs().max().item()),
                "finite": bool(torch.isfinite(tensor).all().item()),
                "n_bodies_in_contact": int((torch.linalg.vector_norm(tensor, dim=-1) > 1.0).sum().item()),
            }

    return PruningEnv
