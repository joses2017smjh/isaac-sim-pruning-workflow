"""Exercise received ROS messages, causality and fault holds on a real ROS graph."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
rclpy = pytest.importorskip("rclpy")

from fixture import write_capture  # noqa: E402
from pruning_sil import topics as T  # noqa: E402
from pruning_sil.capture import Capture  # noqa: E402
from pruning_sil.export_rosbag2 import _stamp, build_messages  # noqa: E402
from pruning_sil.replay import ControllerReplay, invalid_frame_is_held  # noqa: E402
from pruning_sil.servo_node import (
    COMMAND_QOS,
    SENSOR_QOS,
    PruningServoNode,
    array_to_image,
    image_to_array,
)  # noqa: E402


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


class Graph:
    def __init__(self, capture_dir):
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from sensor_msgs.msg import CameraInfo, Image
        from std_msgs.msg import String

        self.servo = PruningServoNode(capture_dir=str(capture_dir))
        self.transport = Node("stream_test_transport")
        self.decisions, self.poses = [], []
        self.transport.create_subscription(
            String,
            T.TOPIC_DECISION,
            lambda m: self.decisions.append(json.loads(m.data)),
            COMMAND_QOS,
        )
        self.transport.create_subscription(PoseStamped, T.TOPIC_PROPOSED_COMMAND, self.poses.append, COMMAND_QOS)
        self.publishers = {
            T.TOPIC_WRIST_IMAGE: self.transport.create_publisher(Image, T.TOPIC_WRIST_IMAGE, SENSOR_QOS),
            T.TOPIC_WRIST_DEPTH: self.transport.create_publisher(Image, T.TOPIC_WRIST_DEPTH, SENSOR_QOS),
            T.TOPIC_WRIST_CAMERA_INFO: self.transport.create_publisher(
                CameraInfo, T.TOPIC_WRIST_CAMERA_INFO, SENSOR_QOS
            ),
            T.TOPIC_TOOL_POSE: self.transport.create_publisher(PoseStamped, T.TOPIC_TOOL_POSE, COMMAND_QOS),
        }
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if (
                all(p.get_subscription_count() for p in self.publishers.values())
                and self.servo.decision_publisher.get_subscription_count()
            ):
                break
            self.spin(0.02)
        assert all(p.get_subscription_count() for p in self.publishers.values())

    def spin(self, duration=0.08):
        deadline = time.monotonic() + duration
        while time.monotonic() < deadline:
            rclpy.spin_once(self.servo, timeout_sec=0.005)
            rclpy.spin_once(self.transport, timeout_sec=0.005)

    def publish(self, capture, index, *, omit=(), change=None, reverse=False):
        messages = [
            (topic, message)
            for topic, message in build_messages(capture, index)
            if topic in self.publishers and topic not in omit
        ]
        if reverse:
            messages.reverse()
        for topic, message in messages:
            if change:
                message = change(topic, message)
            self.publishers[topic].publish(message)
        self.spin()

    def close(self):
        self.transport.destroy_node()
        self.servo.destroy_node()


def test_graph_runs_without_manual_step_and_matches_library_on_identical_messages(
    tmp_path,
):
    capture = Capture(write_capture(tmp_path / "capture", frames=6))
    replay = ControllerReplay(capture)
    expected = [replay.step(i) for i in range(6)]
    graph = Graph(capture.directory)
    try:
        for i in range(6):
            graph.publish(capture, i, reverse=True)
        graph.spin()
        assert [d["frame_index"] for d in graph.decisions] == list(range(6))
        assert all(d["input_mode"] == "ros_messages" for d in graph.decisions)
        assert graph.decisions[0]["reason"] == "no_camera_observation"
        assert len(graph.poses) == 5
        for i, decision in enumerate(graph.decisions[1:], 1):
            assert decision["state"] == expected[i]["decision"]["state"]
            assert decision.get("reason") == expected[i]["decision"].get("reason")
            if "delta_world_m" in decision:
                assert np.allclose(
                    decision["delta_world_m"],
                    expected[i]["decision"]["delta_world_m"],
                    atol=1e-12,
                )
        assert [p.header.stamp.sec * 10**9 + p.header.stamp.nanosec for p in graph.poses] == [
            capture.stamp_ns(i) for i in range(1, 6)
        ]
    finally:
        graph.close()


def test_graph_uses_received_pixels_without_loading_capture_images(tmp_path, monkeypatch):
    capture = Capture(write_capture(tmp_path / "capture", frames=4))
    bundles = [build_messages(capture, i) for i in range(4)]
    graph = Graph(capture.directory)
    try:
        # The publisher already loaded its messages. Any file fallback in the node now fails.
        def refuse(*args):
            raise AssertionError("ROS path must never load an image from disk")

        monkeypatch.setattr(graph.servo.capture, "rgb", refuse)
        monkeypatch.setattr(graph.servo.capture, "depth", refuse)
        for bundle in bundles:
            for topic, message in bundle:
                if topic in graph.publishers:
                    graph.publishers[topic].publish(message)
            graph.spin()
        assert [d["frame_index"] for d in graph.decisions] == list(range(4))
        assert len(graph.poses) == 3
    finally:
        graph.close()


@pytest.mark.parametrize("fault", ["blackout", "dropout", "negative_depth"])
def test_corrupted_received_images_hold_with_zero_command_delta(tmp_path, fault):
    capture = Capture(write_capture(tmp_path / "capture", frames=6))
    graph = Graph(capture.directory)
    try:
        graph.publish(capture, 0)
        graph.publish(capture, 1)

        def corrupt(topic, message):
            if fault == "blackout" and topic == T.TOPIC_WRIST_IMAGE:
                return array_to_image(
                    np.zeros_like(image_to_array(message)),
                    "rgb8",
                    message.header.frame_id,
                    message.header.stamp,
                )
            if fault != "blackout" and topic == T.TOPIC_WRIST_DEPTH:
                value = np.nan if fault == "dropout" else -1.0
                return array_to_image(
                    np.full_like(image_to_array(message), value),
                    "32FC1",
                    message.header.frame_id,
                    message.header.stamp,
                )
            return message

        for i in range(2, 6):
            graph.publish(capture, i, change=corrupt)
        affected = [d for d in graph.decisions if d["frame_index"] >= 3]
        assert len(affected) == 3
        assert all(invalid_frame_is_held(d) for d in affected)
        for pose in graph.poses[2:]:
            recorded = capture.frames[3]["tool_pose_wxyz"]
            assert [
                pose.pose.position.x,
                pose.pose.position.y,
                pose.pose.position.z,
            ] == pytest.approx(recorded[:3])
    finally:
        graph.close()


@pytest.mark.parametrize(
    "fault",
    [
        "missing_depth",
        "wrong_stamp",
        "wrong_camera_frame",
        "malformed_depth",
        "bad_intrinsics",
    ],
)
def test_source_faults_latch_a_hold_and_never_reuse_latest_frame(tmp_path, fault):
    capture = Capture(write_capture(tmp_path / "capture", frames=5))
    graph = Graph(capture.directory)
    try:
        graph.publish(capture, 0)
        graph.publish(capture, 1)

        def corrupt(topic, message):
            if topic == T.TOPIC_WRIST_DEPTH:
                if fault == "wrong_stamp":
                    message.header.stamp = _stamp(capture.stamp_ns(0))
                elif fault == "wrong_camera_frame":
                    message.header.frame_id = "unregistered_camera"
                elif fault == "malformed_depth":
                    message.data = bytes(4)
            if fault == "bad_intrinsics" and topic == T.TOPIC_WRIST_CAMERA_INFO:
                message.k = [0.0] * 9
            return message

        graph.publish(
            capture,
            2,
            omit=(T.TOPIC_WRIST_DEPTH,) if fault == "missing_depth" else (),
            change=corrupt,
        )
        graph.publish(capture, 3)
        graph.spin(0.35)
        graph.publish(capture, 4)
        affected = [d for d in graph.decisions if d["frame_index"] >= 3]
        assert len(affected) == 2
        assert all(d["state"] == "hold" for d in affected)
        assert all(d["reason"] != "no_camera_observation" for d in affected)
        assert len(graph.poses) == 2
    finally:
        graph.close()


def test_duplicate_messages_do_not_duplicate_commands(tmp_path):
    capture = Capture(write_capture(tmp_path / "capture", frames=4))
    graph = Graph(capture.directory)
    try:
        for i in range(4):
            graph.publish(capture, i)
            graph.publish(capture, i)
        assert len(graph.decisions) == 4
        assert len(graph.poses) == 3
    finally:
        graph.close()


def test_image_decoder_handles_padding_and_big_endian_depth():
    from sensor_msgs.msg import Image

    rgb = Image(
        height=2,
        width=2,
        encoding="rgb8",
        step=8,
        data=bytes([1, 2, 3, 4, 5, 6, 99, 99, 7, 8, 9, 10, 11, 12, 99, 99]),
    )
    assert image_to_array(rgb).tolist() == [
        [[1, 2, 3], [4, 5, 6]],
        [[7, 8, 9], [10, 11, 12]],
    ]
    depth = np.array([[0.25, 0.75], [1.25, np.inf]], dtype=">f4")
    message = Image(
        height=2,
        width=2,
        encoding="32FC1",
        is_bigendian=1,
        step=8,
        data=depth.tobytes(),
    )
    assert np.array_equal(image_to_array(message), depth.astype(np.float32))


@pytest.mark.parametrize("mutation", ["short_payload", "short_step", "zero_height"])
def test_image_decoder_rejects_malformed_payload(mutation):
    message = array_to_image(np.ones((2, 3), dtype=np.float32), "32FC1", T.FRAME_WRIST_OPTICAL, None)
    if mutation == "short_payload":
        message.data = bytes(1)
    elif mutation == "short_step":
        message.step = 1
    else:
        message.height = 0
    with pytest.raises(ValueError, match="Malformed"):
        image_to_array(message)
