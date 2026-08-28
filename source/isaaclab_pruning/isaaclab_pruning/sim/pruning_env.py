"""Isaac Lab DirectRLEnv for dormant-spur pruning.

Importing this module requires Isaac Lab. Core reward, observation, ToF, and
success logic live in isaac-free packages and are unit-tested without Sim.
"""

from __future__ import annotations

ISAAC_IMPORT_ERROR = (
    "PruningEnv requires Isaac Lab. Gate 0 (headless create_empty.py in Apptainer) "
    "is still open. Use isaaclab_pruning.task, .policies, and .baselines until then."
)


def require_isaaclab() -> None:
    try:
        import isaaclab  # noqa: F401
    except ImportError as error:
        raise RuntimeError(ISAAC_IMPORT_ERROR) from error


def make_pruning_env_cls():
    """Build the DirectRLEnv subclass only after Isaac Lab is importable."""
    require_isaaclab()

    from collections.abc import Sequence

    import torch

    from isaaclab.assets import Articulation
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
    from isaaclab.envs import DirectRLEnv
    from isaaclab.managers import SceneEntityCfg
    from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

    from isaaclab_pruning.geometry.cutter import cutter_boxes_from_spec
    from isaaclab_pruning.geometry.wood import nearby_wood_in_failure_zone
    from isaaclab_pruning.policies.observations import ObservationVariant, build_observation, proprioception
    from isaaclab_pruning.robot import load_ur5e_pruner_spec
    from isaaclab_pruning.sensors.tof_noise import ToFNoiseConfig, apply_tof_noise
    from isaaclab_pruning.task.reward import dense_pruning_reward
    from isaaclab_pruning.task.success import evaluate_cut_success

    from .pruning_env_cfg import PruningEnvCfg

    class PruningEnv(DirectRLEnv):
        cfg: PruningEnvCfg

        def __init__(self, cfg: PruningEnvCfg, render_mode: str | None = None, **kwargs):
            self.spec = load_ur5e_pruner_spec()
            self.variant = ObservationVariant(cfg.observation_variant)
            super().__init__(cfg, render_mode, **kwargs)
            self.robot_entity_cfg = SceneEntityCfg(
                name="robot",
                joint_names=["ur5e__.*"],
                body_names=[self.spec.eef_body],
            )
            self.robot_entity_cfg.resolve(self.scene)
            self.ik_controller.reset()

        def _setup_scene(self):
            spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
            self.scene.clone_environments(copy_from_source=False)
            self.robot = Articulation(cfg=self.cfg.robot_cfg)
            self.scene.articulations["robot"] = self.robot
            self.ik_controller = DifferentialIKController(
                cfg=DifferentialIKControllerCfg(
                    command_type="pose",
                    use_relative_mode=True,
                    ik_method=self.spec.ik_method,
                    ik_params={"lambda_val": self.spec.ik_lambda},
                ),
                num_envs=self.cfg.num_envs,
                device=self.device,
            )
            self.actions = torch.zeros((self.num_envs, self.spec.action_dim), device=self.device)

        def _pre_physics_step(self, actions: torch.Tensor) -> None:
            self.actions[:] = actions

        def _apply_action(self) -> None:
            self.ik_controller.set_command(self.actions)
            eef_idx = self.robot_entity_cfg.body_ids[0]
            jacobi_idx = eef_idx - 1 if self.robot.is_fixed_base else eef_idx
            jacobians = self.robot.root_physx_view.get_jacobians()[:, jacobi_idx, :, self.robot_entity_cfg.joint_ids]
            eef_pose_w = self.robot.data.body_pose_w[:, eef_idx]
            root_pose_w = self.robot.data.root_pose_w
            from isaaclab.utils.math import subtract_frame_transforms

            eef_pos_b, eef_quat_b = subtract_frame_transforms(
                root_pose_w[:, 0:3], root_pose_w[:, 3:7], eef_pose_w[:, 0:3], eef_pose_w[:, 3:7]
            )
            joint_pos_des = self.ik_controller.compute(
                eef_pos_b, eef_quat_b, jacobians, self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
            )
            self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)

        def _cut_success(self):
            eef_pose = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
            mouth, failure = cutter_boxes_from_spec(
                eef_pose_w=eef_pose,
                mouth_half_extents=self.spec.mouth_half_extents_m,
                failure_half_extents=self.spec.failure_half_extents_m,
                failure_offset_eef=self.spec.failure_offset_m,
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
                arm_collision=getattr(self, "arm_collision", None),
                other_wood_in_failure_zone=other_wood,
                perpendicularity_tolerance_deg=self.spec.perpendicularity_tolerance_deg,
            )

        def _get_rewards(self) -> torch.Tensor:
            eef = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0], 0:3]
            return dense_pruning_reward(eef, self.target.position_w, self._cut_success(), self.actions)

        def _get_dones(self):
            success = self._cut_success().success
            time_out = self.episode_length_buf >= self.max_episode_length
            return success, time_out

        def _get_observations(self) -> dict:
            eef = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
            proprio = proprioception(self.robot.data.joint_pos, self.robot.data.joint_vel, eef)
            observation = build_observation(
                self.variant,
                goal_w=self.target.position_w,
                proprio=proprio,
                flow_hw2=getattr(self, "flow", None),
                tof0=getattr(self, "tof0", None),
                tof1=getattr(self, "tof1", None),
                tof0_valid=getattr(self, "tof0_valid", None),
                tof1_valid=getattr(self, "tof1_valid", None),
                tof0_var=getattr(self, "tof0_var", None),
                metric_depth=getattr(self, "metric_student", None),
                metric_var=getattr(self, "metric_var", None),
            )
            return {"policy": observation}

        def _reset_idx(self, env_ids: Sequence[int] | None):
            if env_ids is None:
                env_ids = torch.arange(self.num_envs, device=self.device)
            super()._reset_idx(env_ids)

        def apply_tof_observation(
            self, ranges0, ranges1, radii0=None, radii1=None, config: ToFNoiseConfig | None = None
        ):
            noisy0 = apply_tof_noise(ranges0, hit_radii_m=radii0, config=config)
            noisy1 = apply_tof_noise(ranges1, hit_radii_m=radii1, config=config)
            self.tof0, self.tof1 = noisy0.range_m, noisy1.range_m
            self.tof0_valid, self.tof1_valid = noisy0.valid, noisy1.valid
            self.tof0_var, self.tof1_var = noisy0.variance_m2, noisy1.variance_m2

    return PruningEnv
