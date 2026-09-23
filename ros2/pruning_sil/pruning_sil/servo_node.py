"""A ROS 2 node that runs the existing pruning controller on replayed sensors.

The node owns no perception and no control law. It converts messages into arrays,
hands them to ``ControllerReplay`` (which in turn drives the repository's
``VisionPruningDemo``), and publishes what that proposes. Proposed is literal:
the command topic is advisory, and nothing in this package actuates anything.

Images and time-of-flight grids use sensor-data QoS, so a slow subscriber drops
samples instead of stalling the pipeline. Commands and decisions are reliable:
losing a hold decision is not an acceptable failure mode even in replay.
"""

from __future__ import annotations

import json

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import Float32MultiArray, String

from . import topics as T
from .capture import Capture
from .replay import ControllerReplay

SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=5,
    durability=QoSDurabilityPolicy.VOLATILE,
)

COMMAND_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    durability=QoSDurabilityPolicy.VOLATILE,
)


def image_to_array(message):
    """Decode the two encodings this replay publishes, and refuse anything else."""
    if message.encoding == "rgb8":
        array = np.frombuffer(message.data, dtype=np.uint8)
        return array.reshape(message.height, message.width, 3)
    if message.encoding == "32FC1":
        array = np.frombuffer(message.data, dtype=np.float32)
        return array.reshape(message.height, message.width)
    raise ValueError(f"Unsupported image encoding for this replay: {message.encoding!r}")


def array_to_image(array, encoding, frame_id, stamp):
    message = Image()
    message.header.frame_id = frame_id
    if stamp is not None:
        message.header.stamp = stamp
    message.height, message.width = int(array.shape[0]), int(array.shape[1])
    message.encoding = encoding
    message.is_bigendian = 0
    contiguous = np.ascontiguousarray(array)
    message.step = int(contiguous.strides[0])
    message.data = contiguous.tobytes()
    return message


class PruningServoNode(Node):
    """Publishes the command the existing controller proposes for each frame."""

    def __init__(self, capture_dir=None, **kwargs):
        super().__init__("pruning_servo", **kwargs)
        self.declare_parameter("capture_dir", capture_dir or "")
        self.declare_parameter("position_tolerance_m", 1e-9)

        directory = self.get_parameter("capture_dir").get_parameter_value().string_value
        if not directory:
            raise ValueError("pruning_servo requires capture_dir: the branch identity, axis and radius "
                             "come from scene metadata, not from the replayed image stream")

        # Static task context. On a real rig this is what a perception or planning
        # stage would supply; in this repository it is recorded scene metadata,
        # never a learned classifier.
        self.capture = Capture(directory)
        self.replay = ControllerReplay(self.capture)
        self.camera_matrix = self.capture.camera_matrix

        self.latest = {}
        self.frame_index = 0
        self.proposals = []

        self.command_publisher = self.create_publisher(PoseStamped, T.TOPIC_PROPOSED_COMMAND, COMMAND_QOS)
        self.decision_publisher = self.create_publisher(String, T.TOPIC_DECISION, COMMAND_QOS)

        self.create_subscription(Image, T.TOPIC_WRIST_IMAGE, self._on_rgb, SENSOR_QOS)
        self.create_subscription(Image, T.TOPIC_WRIST_DEPTH, self._on_depth, SENSOR_QOS)
        self.create_subscription(CameraInfo, T.TOPIC_WRIST_CAMERA_INFO, self._on_camera_info, SENSOR_QOS)
        self.create_subscription(Float32MultiArray, T.TOPIC_TOF_LEFT, self._on_tof_left, SENSOR_QOS)
        self.create_subscription(Float32MultiArray, T.TOPIC_TOF_RIGHT, self._on_tof_right, SENSOR_QOS)

    # Message handlers keep the newest sample; the frame step is driven explicitly
    # so a replay cannot silently skip or reorder a decision.
    def _on_rgb(self, message):
        self.latest["rgb"] = image_to_array(message)

    def _on_depth(self, message):
        self.latest["depth"] = image_to_array(message)

    def _on_camera_info(self, message):
        self.camera_matrix = np.asarray(message.k, dtype=float).reshape(3, 3)

    def _on_tof_left(self, message):
        self.latest["tof_left"] = np.asarray(message.data, dtype=np.float32)

    def _on_tof_right(self, message):
        self.latest["tof_right"] = np.asarray(message.data, dtype=np.float32)

    def step_frame(self, index):
        """Advance the controller one recorded frame and publish what it proposes."""
        proposal = self.replay.step(index)
        stamp = self.get_clock().now().to_msg()
        if proposal is None:
            # Frame 0: no image had been observed. Say so rather than inventing a
            # command, and never publish a pose.
            decision = {"state": "hold", "reason": "no_camera_observation", "frame_index": int(index)}
            self._publish_decision(decision, stamp)
            self.proposals.append(None)
            return None

        pose = PoseStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = T.FRAME_BASE
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (
            float(value) for value in proposal["pose_wxyz"][:3]
        )
        if len(proposal["pose_wxyz"]) >= 7:
            w, x, y, z = (float(value) for value in proposal["pose_wxyz"][3:7])
            pose.pose.orientation.w, pose.pose.orientation.x = w, x
            pose.pose.orientation.y, pose.pose.orientation.z = y, z
        self.command_publisher.publish(pose)

        decision = dict(proposal["decision"])
        decision["phase"] = proposal["phase"]
        decision["frame_index"] = int(index)
        self._publish_decision(decision, stamp)
        self.proposals.append(proposal)
        return proposal

    def _publish_decision(self, decision, stamp):
        message = String()
        message.data = json.dumps(decision, default=float, allow_nan=False, sort_keys=True)
        self.decision_publisher.publish(message)


def main(argv=None):
    rclpy.init(args=argv)
    node = None
    try:
        node = PruningServoNode()
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
