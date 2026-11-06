"""Isaac Sim side of the live ROS 2 loop (PRUNING_ROS2_LOOP=1).

The recorder keeps rendering, stepping physics and writing evidence exactly as it
does in-process. What changes is who decides: ``RosControllerProxy`` stands in for
``VisionPruningDemo``. Each ``observe`` publishes the frame's sensor bundle over
ROS 2 and blocks until the C++ safety gate's command for that bundle arrives; the
next ``command`` returns that gated command, never the controller's proposal.

The simulator pauses while it waits (lockstep), so ROS 2 transport time never
leaks into simulated time. Message delay is instead modelled in simulated time:
``delay_steps`` control steps of the next frame still apply the previous command.

Only standard messages cross into this process. Isaac Sim's bundled rclpy is the
Humble build for its own Python, which cannot load ``pruning_sil_msgs``; the gate
mirrors its decision as JSON for that reason. Import this module only after the
bundled ROS 2 libraries are on the path (hpc/inner/render_pruning_workflow.sh).
"""

from __future__ import annotations

import json
import math
import time
from types import SimpleNamespace

import numpy as np

FRAME_WORLD = "world"
FRAME_OPTICAL = "mock_pruner__wrist_camera_optical_frame"
TOF_FRAMES = ("mock_pruner__tof0", "mock_pruner__tof1")
UR_JOINTS = (
    "ur5e__shoulder_pan_joint",
    "ur5e__shoulder_lift_joint",
    "ur5e__elbow_joint",
    "ur5e__wrist_1_joint",
    "ur5e__wrist_2_joint",
    "ur5e__wrist_3_joint",
)


class RosLoopTimeout(RuntimeError):
    """The ROS 2 graph did not answer a cycle in time."""


def _time(ns):
    from builtin_interfaces.msg import Time

    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def _ns(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _quaternion_wxyz(rotation):
    """Rotation matrix to a unit quaternion (w, x, y, z); for display topics only."""
    m = np.asarray(rotation, dtype=float)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = 2.0 * math.sqrt(trace + 1.0)
        q = (0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s)
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
        q = ((m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s)
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
        q = ((m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s)
    else:
        s = 2.0 * math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
        q = ((m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s)
    norm = math.sqrt(sum(v * v for v in q))
    return tuple(v / norm for v in q)


class RosLoopBridge:
    """Publishes sensor bundles and collects the gated answers, keyed by capture stamp."""

    def __init__(self, *, wait_s=120.0):
        import rclpy
        from geometry_msgs.msg import PoseStamped, WrenchStamped
        from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
        from sensor_msgs.msg import CameraInfo, Image, JointState
        from std_msgs.msg import String

        self.rclpy = rclpy
        self.msg = SimpleNamespace(
            PoseStamped=PoseStamped,
            WrenchStamped=WrenchStamped,
            CameraInfo=CameraInfo,
            Image=Image,
            JointState=JointState,
            String=String,
        )
        reliable = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=20,
            durability=QoSDurabilityPolicy.VOLATILE,
        )
        latched = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1,
            durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
        )
        rclpy.init()
        self.node = rclpy.create_node("isaac_pruning_sim")
        node = self.node
        self.pub = {
            "rgb": node.create_publisher(Image, "/pruning/wrist/image_raw", reliable),
            "depth": node.create_publisher(Image, "/pruning/wrist/depth", reliable),
            "info": node.create_publisher(CameraInfo, "/pruning/wrist/camera_info", reliable),
            "camera_pose": node.create_publisher(PoseStamped, "/pruning/wrist/camera_pose", reliable),
            "tool_pose": node.create_publisher(PoseStamped, "/pruning/tool_pose", reliable),
            "tof_left": node.create_publisher(Image, "/pruning/tof/left/range_image", reliable),
            "tof_right": node.create_publisher(Image, "/pruning/tof/right/range_image", reliable),
            "contact": node.create_publisher(WrenchStamped, "/pruning/contact/wrench", reliable),
            "joints": node.create_publisher(JointState, "/joint_states", reliable),
            "cycle": node.create_publisher(String, "/pruning/sim/cycle", reliable),
        }
        self.config_publisher = node.create_publisher(String, "/pruning/controller/config", latched)
        self.commands, self.jaws, self.evidence, self.decisions, self.arrivals = {}, {}, {}, {}, {}
        node.create_subscription(PoseStamped, "/pruning/gated/command_pose", self._command, reliable)
        node.create_subscription(JointState, "/pruning/gated/jaw", self._jaw, reliable)
        node.create_subscription(String, "/pruning/controller/evidence", self._evidence, reliable)
        node.create_subscription(String, "/pruning/safety/decision_json", self._decision, reliable)
        self.wait_s = float(wait_s)
        self.published_wall_ns = {}

    # --------------------------------------------------------------- callbacks

    def _arrived(self, stamp):
        self.arrivals.setdefault(stamp, time.time_ns())

    def _command(self, message):
        stamp = _ns(message.header.stamp)
        p, q = message.pose.position, message.pose.orientation
        self.commands[stamp] = [p.x, p.y, p.z, q.w, q.x, q.y, q.z]
        self._arrived(stamp)

    def _jaw(self, message):
        self.jaws[_ns(message.header.stamp)] = float(message.position[0]) if message.position else math.nan

    def _evidence(self, message):
        record = json.loads(message.data)
        self.evidence[int(record["stamp_ns"])] = record

    def _decision(self, message):
        record = json.loads(message.data)
        self.decisions[int(record["stamp_ns"])] = record

    # --------------------------------------------------------------- graph

    def publish_config(self, config):
        message = self.msg.String()
        message.data = json.dumps(config, sort_keys=True)
        self.config_publisher.publish(message)

    def wait_for_graph(self, timeout_s):
        """Block until the controller and the safety gate are connected, or raise."""
        node, deadline = self.node, time.monotonic() + float(timeout_s)
        while time.monotonic() < deadline:
            gate = node.count_publishers("/pruning/gated/command_pose") >= 1
            controller = node.count_publishers("/pruning/controller/evidence") >= 1
            # Both the controller and the gate read the tool pose.
            readers = self.pub["tool_pose"].get_subscription_count() >= 2
            if gate and controller and readers and self.pub["tof_left"].get_subscription_count() >= 1:
                return time.monotonic()
            self.rclpy.spin_once(node, timeout_sec=0.2)
        raise RosLoopTimeout("ROS 2 controller and safety gate did not connect")

    def publish_cycle(self, stamp, marker, *, rgb, depth, camera_matrix, world_from_optical, tool_pose_wxyz,
                      tof, contact_force_w, joints):  # fmt: skip
        m, header_stamp = self.msg, _time(stamp)

        def image(array, encoding, frame):
            message = m.Image()
            message.header.stamp, message.header.frame_id = header_stamp, frame
            array = np.ascontiguousarray(array)
            message.height, message.width = int(array.shape[0]), int(array.shape[1])
            message.encoding, message.is_bigendian = encoding, 0
            message.step = int(array.strides[0])
            message.data = array.tobytes()
            return message

        self.pub["rgb"].publish(image(np.asarray(rgb, dtype=np.uint8), "rgb8", FRAME_OPTICAL))
        self.pub["depth"].publish(image(np.asarray(depth, dtype="<f4"), "32FC1", FRAME_OPTICAL))
        info = m.CameraInfo()
        info.header.stamp, info.header.frame_id = header_stamp, FRAME_OPTICAL
        info.height, info.width = int(np.shape(depth)[0]), int(np.shape(depth)[1])
        k = np.asarray(camera_matrix, dtype=float)
        info.k = [float(v) for v in k.reshape(-1)]
        info.p = [
            float(k[0, 0]),
            0.0,
            float(k[0, 2]),
            0.0,
            0.0,
            float(k[1, 1]),
            float(k[1, 2]),
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ]
        info.r = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        info.distortion_model, info.d = "plumb_bob", [0.0] * 5
        self.pub["info"].publish(info)

        transform = np.asarray(world_from_optical, dtype=float)
        camera = m.PoseStamped()
        camera.header.stamp, camera.header.frame_id = header_stamp, FRAME_WORLD
        camera.pose.position.x, camera.pose.position.y, camera.pose.position.z = (float(v) for v in transform[:3, 3])
        w, x, y, z = _quaternion_wxyz(transform[:3, :3])
        camera.pose.orientation.w, camera.pose.orientation.x = w, x
        camera.pose.orientation.y, camera.pose.orientation.z = y, z
        self.pub["camera_pose"].publish(camera)

        pose = m.PoseStamped()
        pose.header.stamp, pose.header.frame_id = header_stamp, FRAME_WORLD
        values = [float(v) for v in tool_pose_wxyz]
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = values[:3]
        pose.pose.orientation.w, pose.pose.orientation.x = values[3], values[4]
        pose.pose.orientation.y, pose.pose.orientation.z = values[5], values[6]
        self.pub["tool_pose"].publish(pose)

        for side, frame in zip(("left", "right"), TOF_FRAMES):
            ranges = np.asarray(tof[side]["ranges"], dtype=np.float32).reshape(8, 8).copy()
            valid = np.asarray(tof[side]["valid"], dtype=bool).reshape(8, 8)
            ranges[~valid | ~np.isfinite(ranges)] = np.nan  # no usable zone, never a plausible range
            self.pub[f"tof_{side}"].publish(image(ranges.astype("<f4"), "32FC1", frame))

        wrench = m.WrenchStamped()
        wrench.header.stamp, wrench.header.frame_id = header_stamp, FRAME_WORLD
        fx, fy, fz = (float(v) for v in contact_force_w)
        wrench.wrench.force.x, wrench.wrench.force.y, wrench.wrench.force.z = fx, fy, fz
        self.pub["contact"].publish(wrench)

        state = m.JointState()
        state.header.stamp = header_stamp
        state.name = list(UR_JOINTS)
        state.position = [float(v) for v in joints["position"]]
        state.velocity = [float(v) for v in joints["velocity"]]
        self.pub["joints"].publish(state)

        cycle = m.String()
        cycle.data = json.dumps(dict(marker, stamp_ns=int(stamp)), sort_keys=True)
        self.published_wall_ns[stamp] = time.time_ns()
        self.pub["cycle"].publish(cycle)  # last: the bundle is complete

    def wait(self, stamp):
        """The gate's command, jaw and decision and the controller's evidence for one bundle."""
        deadline = time.monotonic() + self.wait_s
        while time.monotonic() < deadline:
            if all(stamp in store for store in (self.commands, self.jaws, self.evidence, self.decisions)):
                published = self.published_wall_ns.get(stamp)
                arrived = self.arrivals.get(stamp)
                return {
                    "command": self.commands.pop(stamp),
                    "jaw": self.jaws.pop(stamp),
                    "evidence": self.evidence.pop(stamp),
                    "decision": self.decisions.pop(stamp),
                    "publish_wall_ns": published,
                    "command_arrival_wall_ns": arrived,
                    "round_trip_s": None if published is None or arrived is None else (arrived - published) * 1e-9,
                }
            self.rclpy.spin_once(self.node, timeout_sec=0.002)
        missing = [name for name, store in (("command", self.commands), ("jaw", self.jaws),
                   ("evidence", self.evidence), ("decision", self.decisions)) if stamp not in store]  # fmt: skip
        raise RosLoopTimeout(f"No {', '.join(missing)} for stamp {stamp} within {self.wait_s} s")

    def end(self):
        message = self.msg.String()
        message.data = json.dumps({"end": True})
        self.pub["cycle"].publish(message)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
        self.node.destroy_node()
        self.rclpy.try_shutdown()


class RosControllerProxy:
    """Stands in for VisionPruningDemo inside the recorder; the controller itself runs in ROS 2.

    ``next_sensors`` must be set before each ``observe``: the ToF grids, the largest
    arm contact force vector and the joint state measured with that frame.
    """

    def __init__(self, bridge, local_demo):
        self.bridge = bridge
        self.local = local_demo  # built for configuration reporting only; never observes
        self.cut_step = None
        self.external_stop_reason = None
        self.vision_command_count = 0
        self.initialization = None
        self.next_sensors = None
        self.last_live_vision = None
        self.results = {}
        self.cycle_log = []
        self.timed_out = None
        self.last_result = None
        self._seed = None
        self._pending_stop = None
        self._index = -1
        self._detached = False

    @property
    def mouth_offset(self):
        return self.local.mouth_offset

    def stop(self, reason):
        if self.external_stop_reason is None:
            self.external_stop_reason = reason
            self._pending_stop = reason

    def initialize(self, rgb, depth, seed_pixel, camera_matrix=None, world_from_optical=None, tool_pose_wxyz=None):
        del rgb, depth, camera_matrix, world_from_optical, tool_pose_wxyz
        self._seed = [float(v) for v in seed_pixel]
        return {"state": "deferred_to_ros2_controller"}

    def observe(self, rgb, depth, camera_matrix, world_from_optical, time_s, tool_pose_wxyz, hazard_contact=False):
        del hazard_contact  # the gate and controller read it from the contact wrench
        stamp = int(round(float(time_s) * 1e9))
        marker = {
            "time_s": float(time_s),
            "index": self._index,
            "world_from_optical": np.asarray(world_from_optical).tolist(),
        }
        if self._index == -1:
            marker.update(
                initialize=True,
                seed_pixel=self._seed,
                visible=self._seed is not None and self._pending_stop != "initial_target_not_visible",
            )
        if self._pending_stop:
            marker["external_stop"], self._pending_stop = self._pending_stop, None
        self._index += 1
        if self.timed_out:
            return self._held_without_graph(stamp)
        sensors = self.next_sensors
        self.next_sensors = None
        if sensors is None:
            raise RuntimeError("next_sensors must be set before observe in the ROS 2 loop")
        self.bridge.publish_cycle(
            stamp,
            marker,
            rgb=rgb,
            depth=depth,
            camera_matrix=camera_matrix,
            world_from_optical=world_from_optical,
            tool_pose_wxyz=tool_pose_wxyz,
            tof=sensors["tof"],
            contact_force_w=sensors["contact_force_w"],
            joints=sensors["joints"],
        )
        try:
            result = self.bridge.wait(stamp)
        except RosLoopTimeout as error:
            self.timed_out = str(error)
            return self._held_without_graph(stamp)
        self.results[stamp] = result
        self.last_result = result
        evidence, decision = result["evidence"], result["decision"]
        live = evidence.get("live_vision")
        cut = (live or {}).get("cut") or {}
        jaw = float(result["jaw"])
        detach_event = jaw >= 1.0 and not self._detached
        self._detached = self._detached or detach_event
        held = not decision.get("passed", False)
        self.cut_step = SimpleNamespace(
            phase="stopped" if held else cut.get("phase"),
            closure_progress=jaw,
            detach_event=detach_event,
            detached=self._detached,
            stopped_reason=(decision.get("reason") or decision.get("latched_reason"))
            if held
            else cut.get("stopped_reason"),
        )
        self.vision_command_count = int(evidence.get("vision_command_count") or 0)
        if "initialization" in evidence:
            self.initialization = evidence["initialization"]
        self.last_live_vision = live
        self.cycle_log.append(
            {
                "stamp_ns": stamp,
                "round_trip_s": result["round_trip_s"],
                "passed": decision.get("passed"),
                "reason": decision.get("reason"),
                "evaluation_latency_s": decision.get("evaluation_latency_s"),
            }
        )
        return live

    def _held_without_graph(self, stamp):
        self.cut_step = SimpleNamespace(
            phase="stopped",
            closure_progress=self.cut_step.closure_progress if self.cut_step else 0.0,
            detach_event=False,
            detached=self._detached,
            stopped_reason="ros2_loop_timeout",
        )
        return self.last_live_vision

    def command(self, tool_pose_wxyz, now_s):
        """The gated command for the bundle stamped ``now_s``; a hold when there is none."""
        stamp = int(round(float(now_s) * 1e9))
        result = self.results.pop(stamp, None)
        pose = np.asarray(tool_pose_wxyz, dtype=float).copy()
        if result is None:
            return pose, "stopped_failure", {"state": "hold", "reason": self.timed_out and "ros2_loop_timeout"}
        decision, evidence = result["decision"], result["evidence"]
        command = np.asarray(result["command"], dtype=float)
        if not decision.get("passed", False):
            reason = decision.get("reason") or decision.get("latched_reason")
            return command, "stopped_failure", {"state": "hold", "reason": reason}
        return command, evidence.get("phase"), evidence.get("decision")

    def command_applied(self, phase, decision):
        """The ROS 2 controller acknowledges from the gate's verdict; nothing to do here."""
        del phase, decision

    def evidence(self):
        if self.last_live_vision is not None:
            return self.last_live_vision
        return self.local.evidence()
