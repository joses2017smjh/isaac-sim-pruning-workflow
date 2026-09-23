"""Replay a recorded capture through the pruning controller as a ROS 2 graph.

    ros2 launch pruning_sil sil_replay.launch.py capture_dir:=/path/to/job_21328323

The player publishes recorded sensors and the servo node proposes commands from
them. Both are replay: no hardware is contacted, and no command is actuated.
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    capture_dir = LaunchConfiguration("capture_dir")
    rate_hz = LaunchConfiguration("rate_hz")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "capture_dir",
                description="Recorded capture directory holding report.json, frames.json and frames/",
            ),
            DeclareLaunchArgument(
                "rate_hz",
                default_value="10.0",
                description="Replay rate. The reference captures were recorded at 10 Hz.",
            ),
            Node(
                package="pruning_sil",
                executable="replay_player",
                name="pruning_replay_player",
                output="screen",
                parameters=[{"capture_dir": capture_dir, "rate_hz": rate_hz}],
            ),
            Node(
                package="pruning_sil",
                executable="servo_node",
                name="pruning_servo",
                output="screen",
                parameters=[{"capture_dir": capture_dir}],
            ),
        ]
    )
