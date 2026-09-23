"""Publish a recorded capture onto the SIL topics, one frame at a time.

This exists so the system can be run as an actual ROS 2 graph rather than as one
process calling a library: the player publishes recorded sensors, the servo node
subscribes and proposes commands. It plays recorded frames at the capture's own
cadence and stops at the end rather than looping, because a looped replay would
present the same branch twice as if it were two observations.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node

from . import topics as T
from .capture import Capture
from .export_rosbag2 import TOPIC_TYPES, build_messages
from .servo_node import COMMAND_QOS, SENSOR_QOS

_SENSOR_TOPICS = {
    T.TOPIC_WRIST_IMAGE,
    T.TOPIC_WRIST_DEPTH,
    T.TOPIC_WRIST_CAMERA_INFO,
    T.TOPIC_TOF_LEFT,
    T.TOPIC_TOF_RIGHT,
    T.TOPIC_TOF_LEFT_VALID,
    T.TOPIC_TOF_RIGHT_VALID,
}


def _message_class(type_name):
    package, kind, name = type_name.split("/")
    module = __import__(f"{package}.{kind}", fromlist=[name])
    return getattr(module, name)


class ReplayPlayer(Node):
    """Replays recorded sensor frames. It never synthesizes a frame."""

    def __init__(self, capture_dir=None, **kwargs):
        super().__init__("pruning_replay_player", **kwargs)
        self.declare_parameter("capture_dir", capture_dir or "")
        self.declare_parameter("rate_hz", T.CAPTURE_HZ)
        directory = self.get_parameter("capture_dir").get_parameter_value().string_value
        if not directory:
            raise ValueError("pruning_replay_player requires capture_dir")

        self.capture = Capture(directory)
        self.publishers_by_topic = {
            topic: self.create_publisher(
                _message_class(type_name), topic, SENSOR_QOS if topic in _SENSOR_TOPICS else COMMAND_QOS
            )
            for topic, type_name in TOPIC_TYPES.items()
        }
        self.index = 0
        rate = float(self.get_parameter("rate_hz").get_parameter_value().double_value) or T.CAPTURE_HZ
        self.timer = self.create_timer(1.0 / rate, self._tick)

    def _tick(self):
        if self.index >= len(self.capture):
            self.get_logger().info(f"Replay finished after {len(self.capture)} recorded frames")
            self.timer.cancel()
            return
        for topic, message in build_messages(self.capture, self.index):
            publisher = self.publishers_by_topic.get(topic)
            if publisher is not None:
                publisher.publish(message)
        self.index += 1


def main(argv=None):
    rclpy.init(args=argv)
    node = None
    try:
        node = ReplayPlayer()
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
