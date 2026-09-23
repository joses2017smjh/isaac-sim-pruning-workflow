"""The node itself, driven over real ROS messages, must reproduce the recording.

These tests build the actual ``PruningServoNode``, publish recorded frames into
it through a real ROS 2 graph, and read what it published back. That is a
stronger statement than calling the controller as a library, because it also
covers the message encoding, the QoS settings and the frame bookkeeping.

The procedural-fixture tests run anywhere. The one test that needs the recorded
capture skips when the capture is absent, so a clean CI container without
licensed assets still runs the rest.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

rclpy = pytest.importorskip("rclpy")

from fixture import write_capture  # noqa: E402
from pruning_sil import topics as T  # noqa: E402
from pruning_sil.capture import Capture  # noqa: E402
from pruning_sil.replay import invalid_frame_is_held  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
RECORDED = REPO / "artifacts/isaac_render/job_21328323"


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


class _Sink:
    """Collects everything the node publishes on its two output topics."""

    def __init__(self, node_name="parity_sink"):
        from geometry_msgs.msg import PoseStamped
        from rclpy.node import Node
        from std_msgs.msg import String

        from pruning_sil.servo_node import COMMAND_QOS

        self.node = Node(node_name)
        self.poses, self.decisions = [], []
        self.node.create_subscription(PoseStamped, T.TOPIC_PROPOSED_COMMAND, self.poses.append, COMMAND_QOS)
        self.node.create_subscription(
            String, T.TOPIC_DECISION, lambda m: self.decisions.append(json.loads(m.data)), COMMAND_QOS
        )

    def drain(self, other, rounds=8):
        for _ in range(rounds):
            rclpy.spin_once(other, timeout_sec=0.05)
            rclpy.spin_once(self.node, timeout_sec=0.05)

    def destroy(self):
        self.node.destroy_node()


def _node(capture_dir):
    from pruning_sil.servo_node import PruningServoNode

    return PruningServoNode(capture_dir=str(capture_dir))


def test_node_publishes_a_hold_and_no_pose_for_the_first_frame(tmp_path):
    node = _node(write_capture(tmp_path / "capture", frames=4))
    sink = _Sink("first_frame_sink")
    try:
        node.step_frame(0)
        sink.drain(node)
        assert sink.decisions, "the node must say something about frame 0"
        assert sink.decisions[-1]["state"] == "hold"
        assert sink.decisions[-1]["reason"] == "no_camera_observation"
        # No image had been observed, so proposing a pose would be invention.
        assert sink.poses == []
    finally:
        sink.destroy()
        node.destroy_node()


def test_node_publishes_a_pose_and_decision_once_an_image_exists(tmp_path):
    node = _node(write_capture(tmp_path / "capture", frames=4))
    sink = _Sink("stepped_sink")
    try:
        for index in range(3):
            node.step_frame(index)
        sink.drain(node)
        assert sink.poses, "the node must propose a command after observing an image"
        assert all("frame_index" in decision for decision in sink.decisions)
        assert [d["frame_index"] for d in sink.decisions] == sorted(d["frame_index"] for d in sink.decisions)
    finally:
        sink.destroy()
        node.destroy_node()


def test_node_refuses_to_run_without_the_scene_metadata_it_needs():
    from pruning_sil.servo_node import PruningServoNode

    with pytest.raises(ValueError, match="capture_dir"):
        PruningServoNode(capture_dir="")


def test_decisions_serialize_without_nan(tmp_path):
    node = _node(write_capture(tmp_path / "capture", frames=4))
    sink = _Sink("json_sink")
    try:
        for index in range(3):
            node.step_frame(index)
        sink.drain(node)
        for decision in sink.decisions:
            # allow_nan=False on the publish side means a NaN would have raised.
            assert json.dumps(decision, allow_nan=False)
    finally:
        sink.destroy()
        node.destroy_node()


def test_image_encoding_round_trips_and_rejects_the_unknown(tmp_path):
    from pruning_sil.servo_node import array_to_image, image_to_array

    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    restored = image_to_array(array_to_image(rgb, "rgb8", "f", None))
    assert np.array_equal(restored, rgb)

    depth = np.array([[0.5, np.inf], [1.5, 2.0]], dtype=np.float32)
    back = image_to_array(array_to_image(depth, "32FC1", "f", None))
    # Non-hits must survive the encoding as non-finite, not become a range.
    assert np.array_equal(np.isfinite(back), np.isfinite(depth))
    assert back[0, 0] == pytest.approx(0.5)

    bad = array_to_image(rgb, "rgb8", "f", None)
    bad.encoding = "bayer_rggb8"
    with pytest.raises(ValueError, match="Unsupported image encoding"):
        image_to_array(bad)


def test_invalid_observations_are_never_treated_as_authorization():
    assert invalid_frame_is_held({"state": "hold"})
    assert invalid_frame_is_held({"state": "stale"})
    assert invalid_frame_is_held({})
    assert not invalid_frame_is_held({"state": "tracking"})
    assert not invalid_frame_is_held({"state": "release"})


@pytest.mark.skipif(not RECORDED.is_dir(), reason="recorded capture 21328323 is not present")
def test_node_reproduces_the_recorded_decisions_of_21328323():
    """Every decision state the node proposes must match what was recorded."""
    capture = Capture(RECORDED)
    node = _node(RECORDED)
    try:
        compared = 0
        for index in range(1, 25):
            proposal = node.step_frame(index)
            if proposal is None:
                continue
            recorded = capture.recorded_decision(index)
            assert proposal["decision"].get("state") == recorded.get("state"), f"frame {index}"
            left = np.asarray(proposal["decision"].get("delta_world_m", [0, 0, 0]), dtype=float)
            right = np.asarray(recorded.get("delta_world_m", [0, 0, 0]), dtype=float)
            assert float(np.linalg.norm(left - right)) <= 2.0e-3, f"frame {index}"
            compared += 1
        assert compared >= 20
    finally:
        node.destroy_node()
