"""A ROS 2 node that runs the existing pruning controller on replayed sensors.

The node owns no perception and no control law. It converts messages into arrays,
hands them to ``ControllerReplay`` (which in turn drives the repository's
``VisionPruningDemo``), and publishes what that proposes. Proposed is literal:
the command topic is advisory, and nothing in this package actuates anything.

Images and time-of-flight grids use sensor-data QoS, so a slow subscriber drops
samples instead of stalling the pipeline. Commands and decisions are reliable:
losing a hold decision is not an acceptable failure mode even in replay.
RGB-D and tool-pose messages drive the advisory visual controller. ToF is
telemetry here; the Isaac environment's ToF and release gates are not ported.
"""

from __future__ import annotations

import json
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
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
    """Decode ROS row strides and byte order; reject malformed image payloads."""
    if message.encoding == "rgb8":
        dtype, channels = np.dtype(np.uint8), 3
    elif message.encoding == "32FC1":
        dtype, channels = np.dtype(">f4" if message.is_bigendian else "<f4"), 1
    else:
        raise ValueError(f"Unsupported image encoding for this replay: {message.encoding!r}")
    height, width, step = int(message.height), int(message.width), int(message.step)
    row_bytes = width * channels * dtype.itemsize
    if height <= 0 or width <= 0 or step < row_bytes or len(message.data) != height * step:
        raise ValueError("Malformed ROS image dimensions, row stride or payload length")
    shape = (height, width, channels) if channels == 3 else (height, width)
    strides = (step, channels * dtype.itemsize, dtype.itemsize) if channels == 3 else (step, dtype.itemsize)
    return np.array(np.ndarray(shape, dtype=dtype, buffer=message.data, strides=strides), copy=True).astype(
        np.uint8 if channels == 3 else np.float32, copy=False
    )


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
        self.declare_parameter("sensor_wait_s", 0.25)

        directory = self.get_parameter("capture_dir").get_parameter_value().string_value
        if not directory:
            raise ValueError(
                "pruning_servo requires capture_dir: the branch identity, axis and radius "
                "come from scene metadata, not from the replayed image stream"
            )

        # Static task context. On a real rig this is what a perception or planning
        # stage would supply; in this repository it is recorded scene metadata,
        # never a learned classifier.
        self.capture = Capture(directory)
        self.replay = ControllerReplay(self.capture)
        self.camera_matrix = self.capture.camera_matrix

        self.latest = {}
        self.frame_index = 0
        self.proposals = []
        self.sensor_wait_s = float(self.get_parameter("sensor_wait_s").value)
        if not np.isfinite(self.sensor_wait_s) or self.sensor_wait_s <= 0:
            raise ValueError("sensor_wait_s must be finite and positive")
        stamps = [self.capture.stamp_ns(i) for i in range(len(self.capture))]
        if any(b <= a for a, b in zip(stamps, stamps[1:])):
            raise ValueError("Recorded frame timestamps must be strictly increasing")
        self._index_by_stamp = {stamp: i for i, stamp in enumerate(stamps)}
        self._samples, self._tool_poses, self._pending_frames = {}, {}, {}
        self._next_stream_frame = 0
        self._stream_fault = None
        self.stream_decisions = []

        self.command_publisher = self.create_publisher(PoseStamped, T.TOPIC_PROPOSED_COMMAND, COMMAND_QOS)
        self.decision_publisher = self.create_publisher(String, T.TOPIC_DECISION, COMMAND_QOS)

        self.create_subscription(Image, T.TOPIC_WRIST_IMAGE, self._on_rgb, SENSOR_QOS)
        self.create_subscription(Image, T.TOPIC_WRIST_DEPTH, self._on_depth, SENSOR_QOS)
        self.create_subscription(CameraInfo, T.TOPIC_WRIST_CAMERA_INFO, self._on_camera_info, SENSOR_QOS)
        self.create_subscription(Float32MultiArray, T.TOPIC_TOF_LEFT, self._on_tof_left, SENSOR_QOS)
        self.create_subscription(Float32MultiArray, T.TOPIC_TOF_RIGHT, self._on_tof_right, SENSOR_QOS)
        self.create_subscription(PoseStamped, T.TOPIC_TOOL_POSE, self._on_tool_pose, COMMAND_QOS)
        self.create_timer(min(0.05, self.sensor_wait_s), self._process_stream)

    @staticmethod
    def _stamp_ns(message):
        return int(message.header.stamp.sec) * 1_000_000_000 + int(message.header.stamp.nanosec)

    def _store_sensor(self, key, message, decode):
        index = self._index_by_stamp.get(self._stamp_ns(message))
        # Never substitute a latest or nearest frame for the registered source.
        if index is None or index < max(0, self._next_stream_frame - 1):
            return
        if index >= self._next_stream_frame + 8:
            return
        sample = self._samples.setdefault(index, {})
        if key in sample:
            return
        try:
            if message.header.frame_id != T.FRAME_WRIST_OPTICAL:
                raise ValueError("camera_frame_mismatch")
            sample[key] = decode(message)
        except (TypeError, ValueError, BufferError) as error:
            sample["error"] = f"invalid_{key}: {error}"
        self._process_stream()

    def _on_rgb(self, message):
        self._store_sensor("rgb", message, image_to_array)

    def _on_depth(self, message):
        self._store_sensor("depth", message, image_to_array)

    def _on_camera_info(self, message):
        def decode(info):
            matrix = np.asarray(info.k, dtype=float).reshape(3, 3)
            width, height = self.capture.wrist_resolution
            if (int(info.width), int(info.height)) != (width, height):
                raise ValueError("camera_resolution_mismatch")
            if (
                not np.isfinite(matrix).all()
                or matrix[0, 0] <= 0
                or matrix[1, 1] <= 0
                or not np.allclose(matrix[2], [0, 0, 1], atol=1e-9)
            ):
                raise ValueError("invalid_camera_intrinsics")
            return matrix

        self._store_sensor("camera_matrix", message, decode)

    def _on_tof_left(self, message):
        self.latest["tof_left"] = np.asarray(message.data, dtype=np.float32)

    def _on_tof_right(self, message):
        self.latest["tof_right"] = np.asarray(message.data, dtype=np.float32)

    def _on_tool_pose(self, message):
        index = self._index_by_stamp.get(self._stamp_ns(message))
        if index is None:
            self._stream_fault = "tool_timestamp_not_recorded"
            self._publish_stream_hold(-1, message.header.stamp, self._stream_fault)
            return
        if index < self._next_stream_frame or index in self._pending_frames:
            return
        if index >= self._next_stream_frame + 8:
            self._stream_fault = "tool_frame_sequence_gap"
            self._publish_stream_hold(index, message.header.stamp, self._stream_fault)
            return
        pose = np.array(
            [
                message.pose.position.x,
                message.pose.position.y,
                message.pose.position.z,
                message.pose.orientation.w,
                message.pose.orientation.x,
                message.pose.orientation.y,
                message.pose.orientation.z,
            ],
            dtype=float,
        )
        if (
            message.header.frame_id != T.FRAME_BASE
            or not np.isfinite(pose).all()
            or abs(np.linalg.norm(pose[3:]) - 1) > 1e-3
        ):
            self._stream_fault = "invalid_tool_pose"
        self._tool_poses[index] = pose
        self._pending_frames[index] = (message.header.stamp, time.monotonic())
        self._process_stream()

    def _publish_stream_hold(self, index, stamp, reason):
        decision = {
            "state": "hold",
            "reason": reason,
            "frame_index": int(index),
            "input_mode": "ros_messages",
        }
        self._publish_decision(decision, stamp)
        self.stream_decisions.append(decision)

    def _process_stream(self):
        """Consume timestamp-matched messages in the recorded causal order.

        Camera transforms, target identity and contact context are recorded SIL
        metadata. RGB, optical-Z depth, intrinsics and tool poses come from ROS.
        A dropped source frame latches a hold; it cannot resume from file data.
        """
        while self._pending_frames:
            index = self._next_stream_frame
            entry = self._pending_frames.get(index)
            if entry is None:
                # An absent tool frame cannot silently change the controller's sequence.
                later_index = min(self._pending_frames)
                later_stamp, arrived = self._pending_frames[later_index]
                if time.monotonic() - arrived < self.sensor_wait_s:
                    return
                self._stream_fault = self._stream_fault or "tool_frame_sequence_gap"
                self._next_stream_frame = later_index
                continue
            stamp, arrived = entry
            source = self.capture.source_frame_index(index)
            reason = self._stream_fault
            if source == -1 and index == 0 and reason is None:
                self._publish_stream_hold(index, stamp, "no_camera_observation")
            else:
                if reason is None and source != index - 1:
                    reason = "unsupported_source_frame_schedule"
                sample = self._samples.get(source, {})
                if reason is None:
                    reason = sample.get("error")
                if reason is None and not {"rgb", "depth", "camera_matrix"} <= sample.keys():
                    if time.monotonic() - arrived < self.sensor_wait_s:
                        return
                    reason = "synchronized_camera_observation_missing"
                if reason is None:
                    width, height = self.capture.wrist_resolution
                    if sample["rgb"].shape != (height, width, 3) or sample["depth"].shape != (height, width):
                        reason = "camera_image_shape_mismatch"
                if reason is None and source not in self._tool_poses:
                    reason = "source_tool_pose_missing"
                if reason is None and not self.replay.initialized:
                    initial = self._samples.get(0, {})
                    if not {
                        "rgb",
                        "depth",
                    } <= initial.keys() or not self.replay.initialize(rgb=initial["rgb"], depth=initial["depth"]):
                        reason = "initial_target_not_visible"
                if reason is not None:
                    self._stream_fault = reason
                    self._publish_stream_hold(index, stamp, reason)
                else:
                    self.replay.observe(
                        source,
                        rgb=sample["rgb"],
                        depth=sample["depth"],
                        camera_matrix=sample["camera_matrix"],
                        tool_pose=self._tool_poses[source],
                    )
                    proposal = self.replay.command(index, tool_pose=self._tool_poses[index])
                    self.replay.controller.command_applied(proposal["phase"], proposal["decision"])
                    self._publish_proposal(proposal, index, stamp, "ros_messages")
                    self.stream_decisions.append(
                        dict(
                            proposal["decision"],
                            frame_index=index,
                            input_mode="ros_messages",
                        )
                    )
            del self._pending_frames[index]
            self._next_stream_frame += 1
            for cache in (self._samples, self._tool_poses):
                for old in list(cache):
                    if old < self._next_stream_frame - 1:
                        del cache[old]

    def step_frame(self, index):
        """Explicit offline comparison from files; the launched graph uses callbacks."""
        proposal = self.replay.step(index)
        from .export_rosbag2 import _stamp

        stamp = _stamp(self.capture.stamp_ns(index))
        if proposal is None:
            # Frame 0: no image had been observed. Say so rather than inventing a
            # command, and never publish a pose.
            decision = {
                "state": "hold",
                "reason": "no_camera_observation",
                "frame_index": int(index),
            }
            self._publish_decision(decision, stamp)
            self.proposals.append(None)
            return None

        self._publish_proposal(proposal, index, stamp, "capture_files")
        self.proposals.append(proposal)
        return proposal

    def _publish_proposal(self, proposal, index, stamp, input_mode):
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
        decision["input_mode"] = input_mode
        self._publish_decision(decision, stamp)

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
