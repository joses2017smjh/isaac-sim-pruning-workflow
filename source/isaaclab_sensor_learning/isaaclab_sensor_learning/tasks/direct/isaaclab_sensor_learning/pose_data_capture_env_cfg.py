# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, RenderCfg, MultiUsdFileCfg, SpawnerCfg, RigidBodyPropertiesCfg
from isaaclab.utils import configclass

from isaaclab_assets.robots.universal_robots import UR10e_CFG


from isaaclab_sensor_learning import USD_DIR, TREES_DIR
import glob
import os
from pathlib import Path


@configclass
class PoseDataCaptureEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 60.0
    # - spaces definition
    action_space = 7
    observation_space = 4
    state_space = 0
    num_envs = 2

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120, render_interval=decimation, render=RenderCfg(antialiasing_mode="Off")
    )

    # scene
    scene = InteractiveSceneCfg(
        num_envs=num_envs,
        lazy_sensor_update=False,  # Change to false for evaluation
        replicate_physics=True,
        env_spacing=5.0,
    )

    # trees
    trees_collection_cfg = MultiUsdFileCfg(
        usd_path=glob.glob(os.path.join(TREES_DIR, "models", "*_uv.usda")),
        random_choice=False,
    )

    # robot
    robot_cfg: ArticulationCfg = UR10e_CFG.copy().replace(prim_path="/World/envs/env_.*/robot")

    rig_yaml_path: str = (
        Path(__file__).parent.parent.parent.parent / "config/rigs/test_rig0.yaml"
    )  # NOTE: Standin for now, will be replaced by arguments from evolutionary outputs
    # rig_yaml_path: str = "pose_data_capture/pose_data_capture/assets/rigs/rig0.yaml"
