"""The software-in-the-loop interface, and how it maps onto the real rig.

This is software-in-the-loop. Every message replays a recorded simulator capture.
Nothing here drives hardware, and none of it is hardware-in-the-loop.

Message choice is deliberate. The real rig's time-of-flight packets are
``vl53l8cx_msgs/Vl53l8cx8x8`` and ``Vl53l8cx8x8Filtered``, whose ``Config8x8``
carries a ``Row8x8`` and a ``Column8x8``, each with a ``label``, a ``size`` and a
``stride``. That is the shape of a ``std_msgs/MultiArrayDimension``, so the grids
replay as ``MultiArray`` messages carrying the same labels and sizes. The stride
convention differs and is documented at ``TOF_STRIDES_REAL``. The real packages
are pinned in ``third_party/sources.yaml`` as
``fetch_only`` under a ``NOASSERTION`` repository root, so they are mapped here
rather than vendored, and this package builds without them.
"""

from __future__ import annotations

# Frames. These names come from the pinned mock-pruner description, so a later
# hardware bring-up does not have to rename anything.
FRAME_BASE = "base_link"
FRAME_TOOL = "mock_pruner__tool0"
FRAME_WRIST_OPTICAL = "mock_pruner__wrist_camera_optical_frame"
FRAME_TOF_LEFT = "mock_pruner__tof0"
FRAME_TOF_RIGHT = "mock_pruner__tof1"

# Replayed sensor stream.
TOPIC_WRIST_IMAGE = "/pruning/wrist/image_raw"
TOPIC_WRIST_DEPTH = "/pruning/wrist/depth"
TOPIC_WRIST_CAMERA_INFO = "/pruning/wrist/camera_info"
TOPIC_TOF_LEFT = "/pruning/tof/left/distance"
TOPIC_TOF_RIGHT = "/pruning/tof/right/distance"
TOPIC_TOF_LEFT_VALID = "/pruning/tof/left/valid"
TOPIC_TOF_RIGHT_VALID = "/pruning/tof/right/valid"
TOPIC_JOINT_STATES = "/joint_states"
TOPIC_TOOL_POSE = "/pruning/tool_pose"

# What the recorded controller did, replayed alongside the sensors so a parity
# check has something to compare against.
TOPIC_RECORDED_COMMAND = "/pruning/recorded/command_pose"
TOPIC_RECORDED_RELEASE = "/pruning/recorded/release"

# What this node proposes. "Proposed" is the operative word: nothing here is
# actuated, and on a real rig these would be reviewed before reaching a driver.
TOPIC_PROPOSED_COMMAND = "/pruning/proposed/command_pose"
TOPIC_DECISION = "/pruning/proposed/decision"

# 8x8 grid layout, mirroring vl53l8cx_msgs/Config8x8.
TOF_ROWS = 8
TOF_COLS = 8
TOF_ROW_LABEL = "row"
TOF_COL_LABEL = "column"

#: Sensor cadence recorded for the VL53L8CX firmware: continuous 8x8 at 15 Hz,
#: published as raw millimetres. The replayed capture runs at its own 10 Hz
#: capture rate, which is not the sensor's rate and is not a latency claim.
REAL_TOF_HZ = 15.0
CAPTURE_HZ = 10.0

#: Mapping from this SIL interface to the pinned real-rig interface. This is
#: documentation of an interface, not evidence that the two have been compared
#: on hardware. Nothing in this repository has run on the physical rig.
REAL_RIG_MAPPING = {
    TOPIC_TOF_LEFT: {
        "real_topic": "/microROS/vl53l8cx/distance",
        "real_type": "vl53l8cx_msgs/Vl53l8cx8x8",
        "real_units": "int32 millimetres",
        "sil_type": "std_msgs/Float32MultiArray",
        "sil_units": "float32 metres",
        "note": (
            "The real packet is unstamped and carries both sensors on one topic; "
            "the filtered stream /vl53l8cx/distance_filtered is "
            "vl53l8cx_msgs/Vl53l8cx8x8Filtered with a header and float64 data. "
            "This replay splits left and right and stamps both with capture time."
        ),
    },
    TOPIC_TOF_RIGHT: {
        "real_topic": "/microROS/vl53l8cx/distance",
        "real_type": "vl53l8cx_msgs/Vl53l8cx8x8",
        "real_units": "int32 millimetres",
        "sil_type": "std_msgs/Float32MultiArray",
        "sil_units": "float32 metres",
        "note": "Second of the two mock-pruner sensors; see tof0_filtered / tof1_filtered upstream.",
    },
    TOPIC_JOINT_STATES: {
        "real_topic": "/joint_states",
        "real_type": "sensor_msgs/JointState",
        "sil_type": "sensor_msgs/JointState",
        "note": "Same topic and type as the real rig.",
    },
    TOPIC_PROPOSED_COMMAND: {
        "real_topic": "/servo_node/delta_twist_cmds",
        "real_type": "geometry_msgs/TwistStamped",
        "sil_type": "geometry_msgs/PoseStamped",
        "note": (
            "The real rig is driven through MoveIt Servo, which takes a velocity "
            "twist. This node proposes a bounded Cartesian position delta per "
            "capture frame, so a hardware bring-up would divide the delta by the "
            "control period to produce a twist. At the recorded 10 Hz capture "
            "rate that divisor is 0.1 s. The conversion is not exercised here."
        ),
    },
    TOPIC_WRIST_IMAGE: {
        "real_topic": "(rig-specific wrist camera driver)",
        "real_type": "sensor_msgs/Image",
        "sil_type": "sensor_msgs/Image",
        "note": (
            "The simulated wrist camera is a simulation-defined mount with no "
            "hardware calibration. The pinned branch_detection_system does not "
            "establish the current camera model, so no real topic is claimed."
        ),
    },
}

#: What a reader must not conclude from this package.
NOT_CLAIMED = [
    "Not hardware-in-the-loop: every input is a recorded simulator capture.",
    "No physical VL53L8CX, camera or UR5e has been driven by this node.",
    "Proposed commands are published, never actuated.",
    "Depth is RTX optical-Z ground truth, not a real depth sensor.",
    "Timings are replay timings and are not a latency or real-time claim.",
]


#: The one place this interface deliberately does NOT copy the real message.
#: ``vl53l8cx_msgs`` declares ``Row8x8`` with ``stride 8`` and ``Column8x8`` with
#: ``stride 1``: their stride is the step between consecutive elements of that
#: dimension. ``std_msgs/MultiArrayDimension`` defines stride as the number of
#: elements the dimension spans, which for an 8x8 grid is 64 and 8. Labels and
#: sizes are identical; both strides differ. Copying their values into a
#: MultiArray would produce a message every standard consumer reads wrongly, so
#: a bridge to the real packet must translate rather than pass through.
TOF_STRIDES_REAL = (8, 1)
TOF_STRIDES_ROS = (TOF_ROWS * TOF_COLS, TOF_COLS)


def tof_layout_dims():
    """``(label, size, stride)`` per dimension for a MultiArray holding one grid.

    Labels and sizes match ``vl53l8cx_msgs`` Row8x8 and Column8x8. Strides follow
    the ROS MultiArrayDimension convention; see ``TOF_STRIDES_REAL``.
    """
    return (
        (TOF_ROW_LABEL, TOF_ROWS, TOF_STRIDES_ROS[0]),
        (TOF_COL_LABEL, TOF_COLS, TOF_STRIDES_ROS[1]),
    )
