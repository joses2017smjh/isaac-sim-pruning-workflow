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


def make_pruning_env_cls():
    """Build the DirectRLEnv subclass only after Isaac Lab is importable."""
    require_isaaclab()

    from isaaclab_pruning.sim.lab3_compat import apply as apply_lab3
    from isaaclab_pruning.sim.lab3_compat import as_torch

    apply_lab3()

    from collections.abc import Sequence

    import numpy as np
    import torch

    from isaaclab.assets import Articulation
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.envs import DirectRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sensors import ContactSensor
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, UsdFileCfg, spawn_from_usd, spawn_ground_plane

    from isaaclab_pruning.geometry.cut_point import CutPoint
    from isaaclab_pruning.geometry.cutter import cutter_boxes_from_spec
    from isaaclab_pruning.geometry.wood import nearby_wood_in_failure_zone
    from isaaclab_pruning.policies.observations import ObservationVariant, build_observation, proprioception
    from isaaclab_pruning.robot import load_ur5e_pruner_spec, repository_root
    from isaaclab_pruning.sensors.tof_noise import ToFNoiseConfig, apply_tof_noise
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
            self.robot = Articulation(self.cfg.robot_cfg)
            self.contact = ContactSensor(self.cfg.contact_cfg)
            self.scene.clone_environments(copy_from_source=False)
            if self.device == "cpu":
                self.scene.filter_collisions(global_prim_paths=["/World/ground"])
            self.scene.articulations["robot"] = self.robot
            self.scene.sensors["arm_contact"] = self.contact
            self.robot_entity_cfg = SceneEntityCfg(
                name="robot",
                joint_names=[joint.name for joint in self.spec.arm_joints],
                body_names=[self.spec.eef_body],
            )
            self.robot_entity_cfg.resolve(self.scene)
            self._ensure_target()
            self._fill_observation_buffers()
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
            self.ik_controller.set_command(self.actions)
            eef_idx = self.robot_entity_cfg.body_ids[0]
            jacobi_idx = eef_idx - 1 if self.robot.is_fixed_base else eef_idx
            jacobians = as_torch(self.robot.root_physx_view.get_jacobians())[
                :, jacobi_idx, :, self.robot_entity_cfg.joint_ids
            ]
            eef_pose_w = as_torch(self.robot.data.body_pose_w)[:, eef_idx]
            root_pose_w = as_torch(self.robot.data.root_pose_w)
            from isaaclab.utils.math import subtract_frame_transforms

            eef_pos_b, eef_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], eef_pose_w[:, 0:3], eef_pose_w[:, 3:7]
            )
            joint_pos_des = self.ik_controller.compute(
                eef_pos_b,
                eef_quat_b,
                jacobians,
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
            eef_pose = as_torch(self.robot.data.body_pose_w)[:, self.robot_entity_cfg.body_ids[0], 0:7]
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
            eef = as_torch(self.robot.data.body_pose_w)[:, self.robot_entity_cfg.body_ids[0], 0:3]
            return dense_pruning_reward(eef, self.target.position_w, self._cut_success(), self.actions)

        def _get_dones(self):
            success = self._cut_success().success
            time_out = self.episode_length_buf >= self.max_episode_length
            return success, time_out

        def _fill_observation_buffers(self) -> None:
            """ToF ≠ metric by construction so C vs D cannot be a silent copy."""
            n = self.num_envs
            th, tw = self.cfg.tof_hw
            fh, fw = self.cfg.flow_hw
            mh, mw = self.cfg.metric_hw
            self.flow = torch.zeros((n, fh, fw, 2), device=self.device)
            self.tof0 = torch.full((n, th, tw), 0.40, device=self.device)
            self.tof1 = torch.full((n, th, tw), 0.42, device=self.device)
            self.tof0_valid = torch.ones((n, th, tw), dtype=torch.bool, device=self.device)
            self.tof1_valid = torch.ones((n, th, tw), dtype=torch.bool, device=self.device)
            self.tof0_var = torch.full((n, th, tw), 1e-4, device=self.device)
            self.tof1_var = torch.full((n, th, tw), 1e-4, device=self.device)
            self.metric_student = torch.full((n, mh, mw), 1.20, device=self.device)
            self.metric_var = torch.full((n, mh, mw), 1e-2, device=self.device)

        def _require_buffer(self, name: str):
            if not hasattr(self, name):
                raise RuntimeError(
                    f"Observation buffer {name!r} was never filled. "
                    "Cameras on the stage are not observations (BHL 194/194/194)."
                )
            return getattr(self, name)

        def _proprio(self) -> torch.Tensor:
            joints = self.robot_entity_cfg.joint_ids
            eef = as_torch(self.robot.data.body_pose_w)[:, self.robot_entity_cfg.body_ids[0], 0:7]
            return proprioception(
                as_torch(self.robot.data.joint_pos)[:, joints],
                as_torch(self.robot.data.joint_vel)[:, joints],
                eef,
            )

        def observation_for_variant(self, variant: ObservationVariant) -> torch.Tensor:
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
            self._fill_observation_buffers()
            self._cut_cache = None

        def apply_tof_observation(
            self, ranges0, ranges1, radii0=None, radii1=None, config: ToFNoiseConfig | None = None
        ):
            noisy0 = apply_tof_noise(ranges0, hit_radii_m=radii0, config=config)
            noisy1 = apply_tof_noise(ranges1, hit_radii_m=radii1, config=config)
            self.tof0, self.tof1 = noisy0.range_m, noisy1.range_m
            self.tof0_valid, self.tof1_valid = noisy0.valid, noisy1.valid
            self.tof0_var, self.tof1_var = noisy0.variance_m2, noisy1.variance_m2

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
