"""Export a recorded pruning capture to a rosbag2 bag.

MCAP is the default storage, which is why the container installs
``rosbag2-storage-mcap``; ``--storage sqlite3`` falls back to the stock plugin.

Every message is stamped with the capture's own recorded time, not wall-clock
time, so replaying the bag reproduces the recorded cadence rather than the speed
of whatever machine wrote it. Nothing is interpolated or resampled: a frame that
was not recorded does not appear.

Only recorded telemetry and the wrist camera stream are exported. No orchard
mesh, bark texture or mock-pruner CAD geometry is written into a bag, so a bag
produced here carries nothing that is restricted from redistribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import topics as T
from .capture import Capture

DEFAULT_STORAGE = "mcap"


def _stamp(nanoseconds):
    from builtin_interfaces.msg import Time

    stamp = Time()
    stamp.sec = int(nanoseconds // 1_000_000_000)
    stamp.nanosec = int(nanoseconds % 1_000_000_000)
    return stamp


def _multiarray(values, message_type, dtype):
    from std_msgs.msg import MultiArrayDimension

    message = message_type()
    for label, size, stride in T.tof_layout_dims():
        dimension = MultiArrayDimension()
        dimension.label, dimension.size, dimension.stride = label, int(size), int(stride)
        message.layout.dim.append(dimension)
    message.layout.data_offset = 0
    message.data = np.asarray(values, dtype=dtype).reshape(-1).tolist()
    return message


def build_messages(capture, index):
    """Every message for one recorded frame, as (topic, message) pairs."""
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from sensor_msgs.msg import CameraInfo, Image, JointState
    from std_msgs.msg import Float32MultiArray, String, UInt8MultiArray
    from tf2_msgs.msg import TFMessage

    from .servo_node import array_to_image

    frame = capture.frames[index]
    stamp = _stamp(capture.stamp_ns(index))
    out = []

    if capture.has_images():
        out.append((T.TOPIC_WRIST_IMAGE, array_to_image(capture.rgb(index), "rgb8", T.FRAME_WRIST_OPTICAL, stamp)))
        # Optical-Z metres. Non-hits stay non-finite rather than becoming a
        # plausible-looking range.
        out.append((T.TOPIC_WRIST_DEPTH, array_to_image(capture.depth(index), "32FC1", T.FRAME_WRIST_OPTICAL, stamp)))

    info = CameraInfo()
    info.header.stamp, info.header.frame_id = stamp, T.FRAME_WRIST_OPTICAL
    width, height = capture.wrist_resolution
    info.width, info.height = int(width), int(height)
    matrix = capture.camera_matrix
    info.k = [float(value) for value in matrix.reshape(-1)]
    info.p = [
        float(matrix[0, 0]), 0.0, float(matrix[0, 2]), 0.0,
        0.0, float(matrix[1, 1]), float(matrix[1, 2]), 0.0,
        0.0, 0.0, 1.0, 0.0,
    ]
    info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    info.distortion_model = "plumb_bob"
    info.d = [0.0, 0.0, 0.0, 0.0, 0.0]
    out.append((T.TOPIC_WRIST_CAMERA_INFO, info))

    for side, distance_topic, valid_topic, frame_id in (
        ("left", T.TOPIC_TOF_LEFT, T.TOPIC_TOF_LEFT_VALID, T.FRAME_TOF_LEFT),
        ("right", T.TOPIC_TOF_RIGHT, T.TOPIC_TOF_RIGHT_VALID, T.FRAME_TOF_RIGHT),
    ):
        metres, mask = capture.tof(index, side)
        # NaN cannot travel through Float32MultiArray without becoming a number
        # somewhere downstream, so invalid zones carry a sentinel and the
        # validity grid beside them is authoritative.
        safe = np.where(np.isfinite(metres), metres, -1.0).astype(np.float32)
        out.append((distance_topic, _multiarray(safe, Float32MultiArray, np.float32)))
        out.append((valid_topic, _multiarray(mask.astype(np.uint8), UInt8MultiArray, np.uint8)))
        del frame_id

    joints = JointState()
    joints.header.stamp = stamp
    joints.position = [float(value) for value in frame.get("joint_position_rad", [])]
    joints.velocity = [float(value) for value in frame.get("joint_velocity_rad_s", [])]
    joints.name = [f"joint_{number}" for number in range(len(joints.position))]
    out.append((T.TOPIC_JOINT_STATES, joints))

    pose = PoseStamped()
    pose.header.stamp, pose.header.frame_id = stamp, T.FRAME_BASE
    tool = frame.get("tool_pose_wxyz") or []
    if len(tool) >= 3:
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (float(v) for v in tool[:3])
    if len(tool) >= 7:
        pose.pose.orientation.w, pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z = (
            float(v) for v in tool[3:7]
        )
    out.append((T.TOPIC_TOOL_POSE, pose))

    transform = TransformStamped()
    transform.header.stamp, transform.header.frame_id = stamp, T.FRAME_BASE
    transform.child_frame_id = T.FRAME_TOOL
    if len(tool) >= 3:
        transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z = (
            float(v) for v in tool[:3]
        )
    if len(tool) >= 7:
        transform.transform.rotation.w, transform.transform.rotation.x = float(tool[3]), float(tool[4])
        transform.transform.rotation.y, transform.transform.rotation.z = float(tool[5]), float(tool[6])
    else:
        transform.transform.rotation.w = 1.0
    out.append(("/tf", TFMessage(transforms=[transform])))

    command = PoseStamped()
    command.header.stamp, command.header.frame_id = stamp, T.FRAME_BASE
    recorded = frame.get("command_pose_root_wxyz") or []
    if len(recorded) >= 3:
        command.pose.position.x, command.pose.position.y, command.pose.position.z = (
            float(v) for v in recorded[:3]
        )
    out.append((T.TOPIC_RECORDED_COMMAND, command))

    release = String()
    release.data = json.dumps(
        {
            "frame_index": int(frame["index"]),
            "phase": frame.get("phase"),
            "detachment_requested_after_capture": bool(frame.get("detachment_requested_after_capture")),
            "controller_source_frame_index": capture.source_frame_index(index),
            "sensor_stop_reason": frame.get("sensor_stop_reason"),
        },
        sort_keys=True,
    )
    out.append((T.TOPIC_RECORDED_RELEASE, release))
    return out


TOPIC_TYPES = {
    T.TOPIC_WRIST_IMAGE: "sensor_msgs/msg/Image",
    T.TOPIC_WRIST_DEPTH: "sensor_msgs/msg/Image",
    T.TOPIC_WRIST_CAMERA_INFO: "sensor_msgs/msg/CameraInfo",
    T.TOPIC_TOF_LEFT: "std_msgs/msg/Float32MultiArray",
    T.TOPIC_TOF_RIGHT: "std_msgs/msg/Float32MultiArray",
    T.TOPIC_TOF_LEFT_VALID: "std_msgs/msg/UInt8MultiArray",
    T.TOPIC_TOF_RIGHT_VALID: "std_msgs/msg/UInt8MultiArray",
    T.TOPIC_JOINT_STATES: "sensor_msgs/msg/JointState",
    T.TOPIC_TOOL_POSE: "geometry_msgs/msg/PoseStamped",
    "/tf": "tf2_msgs/msg/TFMessage",
    T.TOPIC_RECORDED_COMMAND: "geometry_msgs/msg/PoseStamped",
    T.TOPIC_RECORDED_RELEASE: "std_msgs/msg/String",
}


def export(capture_dir, output, storage_id=DEFAULT_STORAGE, limit=None):
    import rosbag2_py
    from rclpy.serialization import serialize_message

    capture = Capture(capture_dir)
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing bag: {output}")

    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(output), storage_id=storage_id),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    for topic, type_name in TOPIC_TYPES.items():
        writer.create_topic(
            rosbag2_py.TopicMetadata(name=topic, type=type_name, serialization_format="cdr")
        )

    count = len(capture) if limit is None else min(len(capture), int(limit))
    written = 0
    for index in range(count):
        nanoseconds = capture.stamp_ns(index)
        for topic, message in build_messages(capture, index):
            writer.write(topic, serialize_message(message), nanoseconds)
            written += 1
    del writer
    return {"bag": str(output), "frames": count, "messages": written, "storage_id": storage_id}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--storage", default=DEFAULT_STORAGE, choices=("mcap", "sqlite3"))
    parser.add_argument("--limit", type=int, help="Export only the first N frames")
    args = parser.parse_args(argv)
    print(json.dumps(export(args.capture_dir, args.output, args.storage, args.limit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
