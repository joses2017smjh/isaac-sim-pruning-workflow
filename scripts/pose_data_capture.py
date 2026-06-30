#!/usr/bin/env python3
from isaaclab.app import AppLauncher
import argparse

# add argparse arguments
parser = argparse.ArgumentParser(description="Pose data capture script")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default="Template-Pose-Data-Capture-Direct-v0", help="Name of the task.")
# parser.add_argument("--enable_cameras", action="store_true", default=True, help="Enable cameras in the environment.")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import torch
import numpy as np

from isaaclab_tasks.utils import parse_env_cfg

import isaaclab_sensor_learning.tasks  # noqa: F401
from isaaclab_sensor_learning.utils import quaternion_utils as qutils

import data_utils.data_logger as data_logger
import data_utils.pose_generator as pose_generator
import pprint as pp
import os
import re

import time
from isaacsim.core.utils import stage as stage_utils


def main():
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # print info (this is vectorized environment)
    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")
    # reset environment
    env.reset()

    robot = env.unwrapped.robot

    # Set up data logger
    # dlog = data_logger.DataLogger()

    # generate discrete poses
    x_range = (-1.0, 1.0)
    y_range = (0.0, 1.0)
    z_range = (0.5, 2.0)
    theta_range = (-np.pi / 4, np.pi / 4)
    phi_range = (-np.pi / 4, np.pi / 4)
    x_size = 5
    y_size = 5
    z_size = 5
    angles_size = 5
    start_orientation = np.array([0, 0, 1])
    discrete_poses = pose_generator.generate_discrete_poses(
        x_range=x_range,
        y_range=y_range,
        z_range=z_range,
        theta_range=theta_range,
        phi_range=phi_range,
        x_size=x_size,
        y_size=y_size,
        z_size=z_size,
        angles_size=angles_size,
        start_orientation=start_orientation,
    )

    # # a test pose for debugging
    # discrete_poses = np.array([[0.5, 0.5, 0.7, 0.707, 0, 0.707, 0]])
    # # trial metadata
    # trial_metadata = {
    #     "trial_name": dlog.trial_name,
    #     "num_envs": env.unwrapped.cfg.num_envs,
    #     "x_range": x_range,
    #     "y_range": y_range,
    #     "z_range": z_range,
    #     "theta_range": theta_range,
    #     "phi_range": phi_range,
    #     "x_size": x_size,
    #     "y_size": y_size,
    #     "z_size": z_size,
    #     "angles_size": angles_size,
    #     "poses": discrete_poses,
    # }
    # # tree metadata
    # # print(stage_utils.print_stage_prim_paths())
    # tree_metadata = {}
    # tree_prims = [prim for prim in stage_utils.get_current_stage().Traverse() if prim.GetName() == "tree"]
    # for tree_prim in tree_prims:
    #     # print(pp.pformat(dir(tree_prim)))
    #     prim_path = tree_prim.GetPath()
    #     match = re.search(r"env_(\d+)", str(prim_path))
    #     if match:
    #         env_idx = int(match.group(1))
    #     usd_path = tree_prim.GetMetadata("references").prependedItems[0].assetPath
    #     tree_str = os.path.basename(usd_path).split(".")[0].strip("_uv")
    #     tree_namespace, tree_type, tree_id = tree_str.split("_")
    #     tree_metadata[env_idx] = {
    #         "tree_usd_path": usd_path,
    #         "prim_path": str(prim_path),
    #         "env_idx": env_idx,
    #         "tree_namespace": tree_namespace,
    #         "tree_type": tree_type,
    #         "tree_id": tree_id,
    #         "pose": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    #     }

    # # sensors
    # sensor_metadata = {}
    # for sensor_name, sensor in env.unwrapped.sensors.items():
    #     # pp.pprint(sensor.cfg)
    #     # pp.pprint(sensor.cfg.asdict())
    #     sensor_metadata[sensor_name] = {
    #         "class_type": str(sensor.cfg.class_type),
    #         "width": sensor.cfg.width,
    #         "height": sensor.cfg.height,
    #         "data_rate_hz": 1 / sensor.cfg.update_period,
    #         "receiver_offset": np.concatenate(
    #             [sensor.cfg.offset.pos, qutils.wxyz_to_xyzw(np.asarray(sensor.cfg.offset.rot))]
    #         ),
    #         "depth_clipping_behavior": sensor.cfg.depth_clipping_behavior,
    #         "data_types": sensor.cfg.data_types,
    #         "z_near": sensor.cfg.spawn.clipping_range[0],
    #         "z_far": sensor.cfg.spawn.clipping_range[1],
    #         "focal_length": sensor.cfg.spawn.focal_length,
    #         "focus_distance": sensor.cfg.spawn.focus_distance,
    #         "f_stop": sensor.cfg.spawn.f_stop,
    #         "horizontal_aperture": sensor.cfg.spawn.horizontal_aperture,
    #         "vertical_aperture": sensor.cfg.spawn.vertical_aperture,
    #     }
    # dlog.save_trial_metadata(trial_metadata=trial_metadata)
    # dlog.save_tree_metadata(tree_metadata=tree_metadata)
    # dlog.save_sensor_metadata(
    #     sensor_metadata=sensor_metadata, n_poses=discrete_poses.shape[0], num_envs=env.unwrapped.cfg.num_envs
    # )

    # simulate environment
    pose_idx = 0
    while simulation_app.is_running():
        print(f"\rpose iter: {pose_idx+1}/{len(discrete_poses)}", end="")
        # run everything in inference mode
        with torch.inference_mode():
            # Actions
            actions = torch.tensor(
                discrete_poses[pose_idx][np.newaxis, :], dtype=torch.float32, device=env.unwrapped.device
            ).repeat(env.unwrapped.num_envs, 1)
            observations, rewards, terminated, truncated, info = env.step(actions)


            # # print(observations['tof0'].output['rgb'].shape)
            print(robot.data.body_pos_w[:, robot.find_bodies("wrist_3_link")[0]])
            print(torch.tensor([discrete_poses[pose_idx][0:3]], dtype=torch.float32, device=env.unwrapped.device))
            # # break # remove after testing
            if torch.allclose(
                robot.data.body_pos_w[:, robot.find_bodies("wrist_3_link")[0]],
                torch.tensor([discrete_poses[pose_idx][0:3]], dtype=torch.float32, device=env.unwrapped.device),
                atol=0.005,
            ) and torch.allclose(
                robot.data.body_quat_w[:, robot.find_bodies("wrist_3_link")[0]],
                torch.tensor([discrete_poses[pose_idx][3:7]], dtype=torch.float32, device=env.unwrapped.device),
                atol=0.005,
            ):

            #     pose_full = torch.cat(
            #         [
            #             robot.data.body_pos_w[:, robot.find_bodies("wrist_3_link")[0]],
            #             robot.data.body_quat_w[:, robot.find_bodies("wrist_3_link")[0]],
            #         ],
            #         dim=0,
            #     )
            #     if pose_idx >= len(discrete_poses):
            #         dlog.save_observations(observations=observations, pose=pose_full, last_obs=True)
            #         break
            #     else:
            #         dlog.save_observations(observations=observations, pose=pose_full)
            #     # break  # remove after testing
                pose_idx += 1

            # Remove after testing
            print(f"\nGoal pos: {discrete_poses[pose_idx][0:3]}, goal quat: {discrete_poses[pose_idx][3:7]}")
            print(f"Curr pos: {robot.data.body_pos_w[:, robot.find_bodies('wrist_3_link')[0]]}")
            print(f"Curr quat: {robot.data.body_quat_w[:, robot.find_bodies('wrist_3_link')[0]]}")

    # close the simulator
    env.close()
    # print("\n[INFO]: Simulation finished, data saved to: ", datafile_path)
    return


if __name__ == "__main__":
    import traceback

    try:
        main()
    except Exception as e:
        # print(f"[ERROR]: {e}")
        traceback.print_exc()
    finally:
        simulation_app.close()
