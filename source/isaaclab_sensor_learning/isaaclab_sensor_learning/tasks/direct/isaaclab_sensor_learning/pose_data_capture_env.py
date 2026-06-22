# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import torch
from collections.abc import Sequence


from isaaclab.assets import Articulation, AssetBase
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.envs import DirectRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import Camera, CameraCfg, MultiMeshRayCasterCamera, MultiMeshRayCasterCameraCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform, subtract_frame_transforms

import isaaclab_sensor_learning as pdc
import isaaclab_sensor_learning.sensor.yaml_to_cfg as yaml_to_cfg
import isaaclab_sensor_learning.sensor.rig_utils as rig_utils
from isaaclab_sensor_learning.sensor.generate_rig import RigGenerator
import isaaclab_sensor_learning.utils.usd_utils as usd_utils
import isaaclab_sensor_learning.utils.quaternion_utils as qutils

from .pose_data_capture_env_cfg import PoseDataCaptureEnvCfg

# from pose_data_capture.sensor.yaml_to_cfg import rig_yaml_to_sensor_cfgs


import os


class PoseDataCaptureEnv(DirectRLEnv):
    cfg: PoseDataCaptureEnvCfg

    def __init__(self, cfg: PoseDataCaptureEnvCfg, render_mode: str | None = None, **kwargs):
        self._curr_tree_mesh_prim = None
        if not cfg.rig_yaml_path:
            raise ValueError("cfg.rig_yaml_path must be set before instantiating PoseDataCaptureEnv.")

        self._sensor_cfgs = yaml_to_cfg.rig_yaml_to_sensor_cfgs(rig_yaml_path=cfg.rig_yaml_path) # TODO: change to dict later
        # self.rig_generator = RigGenerator(sensor_cfgs=self._sensor_cfgs)
        # self._rig_cfg = self.rig_generator.generate_rig()

        # self._sensor_cfgs =

        self.camera_poses: np.ndarray
        self.curr_pose_idx = 0

        self.sensors = {}
        self.sensor_groups = {
            "camera": [],
            "lidar": [],
            "tof": [],
        }

        self._new_action = False

        super().__init__(cfg, render_mode, **kwargs)

        self.robot_entity_cfg = SceneEntityCfg(name="robot", joint_names=[".*"], body_names=["wrist_3_link"])
        self.robot_entity_cfg.resolve(self.scene)

        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()
        return

    def _setup_scene(self):
        # add ground plane
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        # clone and replicate
        self.scene.clone_environments(copy_from_source=False)

        # trees
        self.prim = sim_utils.spawn_multi_usd_file(
            prim_path="/World/envs/env_.*/tree",
            cfg=self.cfg.trees_collection_cfg,
            translation=(0.0, 1.0, 0.0),
            orientation=qutils.wxyz_to_xyzw(np.array([0.0, 0.0, 0.0, 1.0])),
        )

        # robot
        self.cfg.robot_cfg.init_state.pos = (0.0, 0.0, 0.0)
        self.cfg.robot_cfg.init_state.rot = tuple(qutils.xyzw_to_wxyz(np.asarray([0.0, 0.0, 0.0, 1.0])))
        self.robot = Articulation(cfg=self.cfg.robot_cfg)
        self.scene.articulations["robot"] = self.robot

        # add sensors
        

        # for sensor_name, sensor_dict in self._sensor_cfgs.items():
        #     sensor_cfg = sensor_dict["cfg"]
        #     if isinstance(sensor_cfg, CameraCfg):
        #         self.sensors[sensor_name] = Camera(cfg=sensor_cfg)
        #     elif isinstance(sensor_cfg, MultiMeshRayCasterCameraCfg):
        #         self.sensors[sensor_name] = MultiMeshRayCasterCamera(cfg=sensor_cfg)
        #     else:
        #         raise ValueError(f"Unsupported sensor config type: {type(sensor_cfg)} for sensor '{sensor_name}'")
        #     self.sensor_groups[sensor_dict["type"]].append(sensor_name) # Need to move this to parsing function so that sensor can be grouped and correct offsets can be applied

        # controllers
        self.ik_controller = DifferentialIKController(
            cfg=DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls", ik_params={"lambda_val": 0.0001}
            ),
            num_envs=self.cfg.num_envs,
            device=self.device,
        )
        self.ik_controller.reset()

        # actions
        self.actions = torch.zeros(
            size=(self.num_envs, self.ik_controller.action_dim), dtype=torch.float32, device=self.device
        )  # (N, 7) target pose in world frame

        # markers
        frame_marker_cfg = FRAME_MARKER_CFG.copy()
        frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        self.eef_markers = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
        self.goal_markers = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))
        return

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions[:] = actions.clone().to(dtype=torch.float32)
        if not torch.allclose(actions, self.actions):
            self.actions = actions
            self._new_action = True
        return

    def _apply_action(self) -> None:
        if self._new_action:
            self.ik_controller.set_command(self.actions)  # (N, 7) target pose in world frame
            self._new_action = False
        # self.ik_controller.set_command(self.actions) # (N, 7) target pose in world frame

        if self.robot.is_fixed_base:
            eef_jacobi_idx = self.robot_entity_cfg.body_ids[0] - 1
        else:
            eef_jacobi_idx = self.robot_entity_cfg.body_ids[0]

        # print("body_names:", self.robot.data.body_names)
        # print("body_ids:", self.robot_entity_cfg.body_ids)
        # print("jacobians shape:", self.robot.root_physx_view.get_jacobians().shape)

        jacobians = self.robot.root_physx_view.get_jacobians()[:, eef_jacobi_idx, :, self.robot_entity_cfg.joint_ids]
        eef_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0]]
        root_pose_w = self.robot.data.root_pose_w
        joint_pos = self.robot.data.joint_pos[:, self.robot_entity_cfg.joint_ids]
        # compute frame in base frame
        eef_pos_b, eef_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3], root_pose_w[:, 3:7], eef_pose_w[:, 0:3], eef_pose_w[:, 3:7]
        )
        joint_pos_des = self.ik_controller.compute(eef_pos_b, eef_quat_b, jacobians, joint_pos)
        # apply actions
        self.robot.set_joint_position_target(joint_pos_des, joint_ids=self.robot_entity_cfg.joint_ids)

        # update marker poses
        eef_pose_w = self.robot.data.body_pose_w[:, self.robot_entity_cfg.body_ids[0], 0:7]
        self.eef_markers.visualize(eef_pose_w[:, 0:3], eef_pose_w[:, 3:7])
        self.goal_markers.visualize(self.actions[:, 0:3] + self.scene.env_origins, self.actions[:, 3:7])

        return

    def _get_observations(self) -> dict:
        sensor_data = {}
        for sensor_name, sensor in self.sensors.items():
            sensor_data[sensor_name] = sensor.data
        return sensor_data

    def _get_rewards(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, device=self.device)

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        )

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        super()._reset_idx(env_ids)
        return
