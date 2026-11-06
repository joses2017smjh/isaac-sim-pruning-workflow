"""The pruning controller as a ROS 2 node in a live loop with Isaac Sim.

Isaac Sim publishes one sensor bundle per control cycle (wrist RGB, optical-Z
depth, intrinsics, tool pose, contact wrench and a ``/pruning/sim/cycle`` marker
carrying the exact capture time and camera pose). This node runs the repository's
unchanged ``VisionPruningDemo`` on that bundle and publishes, for the same stamp:

* ``/pruning/perception/target``   the cut observation the controller evaluated;
* ``/pruning/proposed/proposal``   the command it proposes for the next cycle;
* ``/pruning/controller/evidence`` its per-cycle evidence (JSON), which the
  simulator records in place of the in-process controller's.

It does not move anything. The C++ safety gate decides whether the proposal
reaches the simulated arm; this node learns that verdict from
``/pruning/safety/decision`` before it observes the next cycle, and acknowledges
or stops exactly as the in-process recorder did (``command_applied`` /``stop``).

The order of calls mirrors ``hpc/inner/render_pruning_workflow.py``: the tracker
is seeded once from the supplied pixel, the preview is observed at t = 0 without
contact, and every later cycle is observed after the previous verdict is known.
"""

from __future__ import annotations

import json
import math
import time
import traceback

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, WrenchStamped
from pruning_sil_msgs.msg import ControllerProposal, SafetyDecision, TargetEstimate
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from std_msgs.msg import String

from . import topics as T
from .servo_node import image_to_array

RELIABLE = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=20,
    durability=QoSDurabilityPolicy.VOLATILE,
)
LATCHED = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)
HAZARD_CONTACT_FORCE_N = 5.0


def to_time(ns):
    from builtin_interfaces.msg import Time

    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def ns_of(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def build_controller(config):
    """Construct VisionPruningDemo exactly as the Isaac recorder does, from its published config."""
    from isaaclab_pruning.sim.vision_demo_controller import ApproachStrategy, VisionPruningDemo

    approach = config["approach"]
    strategy = ApproachStrategy(
        mode=approach["mode"],
        standoff_m=float(approach["standoff_m"]),
        max_step_m=float(approach["max_step_m"]),
        max_rotation_deg=float(approach["max_rotation_deg"]),
        planned_tool_quat_wxyz=None
        if approach.get("planned_tool_quat_wxyz") is None
        else tuple(float(v) for v in approach["planned_tool_quat_wxyz"]),
    )
    return VisionPruningDemo(
        config["target_id"],
        np.asarray(config["target_axis_w"], dtype=float),
        float(config["target_radius_m"]),
        np.asarray(config["home_pose_wxyz"], dtype=float),
        closing_axis_tool=tuple(float(v) for v in config["closing_axis_tool"]),
        photometric_normalization=config["photometric_normalization"],
        approach=strategy,
        motion_model=config["motion_model"],
        jaw_self_mask=bool(config["jaw_self_mask"]),
        closure_hold=bool(config["closure_hold"]),
        depth_appearance=bool(config["depth_appearance"]),
        depth_appearance_arm=config["depth_appearance_arm"],
    )


def target_estimate(observation, cycle_ns, tracker_state):
    message = TargetEstimate()
    message.header.stamp = to_time(int(round(observation.vision_timestamp_s * 1e9)))
    message.header.frame_id = T.FRAME_WORLD
    message.observation_stamp = to_time(cycle_ns)
    message.target_id = observation.target_id
    message.target_semantic = observation.target_semantic
    message.vision_valid = bool(observation.vision_valid)
    message.vision_source = observation.vision_source
    if observation.held_target_timestamp_s is not None:
        message.has_held_target_stamp = True
        message.held_target_stamp = to_time(int(round(observation.held_target_timestamp_s * 1e9)))
    message.position.x, message.position.y, message.position.z = (float(v) for v in observation.target_position_w)
    message.axis.x, message.axis.y, message.axis.z = (float(v) for v in observation.target_axis_w)
    message.radius_m = float(observation.target_radius_m)
    message.tracker_state = str(tracker_state or "")
    return message


class LiveController(Node):
    def __init__(self, **kwargs):
        super().__init__("pruning_live_controller", **kwargs)
        self.config = None
        self.demo = None
        self.pending = {}
        self.decisions = {}
        self.previous = None  # (stamp, phase, decision) of the proposal the gate is judging
        self.initialization = None
        self.cycles = 0
        self.errors = []
        self.finished = False
        self.timings = []

        self.create_subscription(String, T.TOPIC_CONTROLLER_CONFIG, self._on_config, LATCHED)
        self.create_subscription(String, T.TOPIC_SIM_CYCLE, self._on_cycle, RELIABLE)
        self.create_subscription(Image, T.TOPIC_WRIST_IMAGE, lambda m: self._store("rgb", m), RELIABLE)
        self.create_subscription(Image, T.TOPIC_WRIST_DEPTH, lambda m: self._store("depth", m), RELIABLE)
        self.create_subscription(CameraInfo, T.TOPIC_WRIST_CAMERA_INFO, lambda m: self._store("info", m), RELIABLE)
        self.create_subscription(PoseStamped, T.TOPIC_TOOL_POSE, lambda m: self._store("pose", m), RELIABLE)
        self.create_subscription(WrenchStamped, T.TOPIC_CONTACT_WRENCH, lambda m: self._store("contact", m), RELIABLE)
        self.create_subscription(SafetyDecision, T.TOPIC_SAFETY_DECISION, self._on_decision, RELIABLE)

        self.target_publisher = self.create_publisher(TargetEstimate, T.TOPIC_TARGET_ESTIMATE, RELIABLE)
        self.proposal_publisher = self.create_publisher(ControllerProposal, T.TOPIC_PROPOSAL, RELIABLE)
        self.evidence_publisher = self.create_publisher(String, T.TOPIC_CONTROLLER_EVIDENCE, RELIABLE)
        self.command_publisher = self.create_publisher(PoseStamped, T.TOPIC_PROPOSED_COMMAND, RELIABLE)
        self.decision_publisher = self.create_publisher(String, T.TOPIC_DECISION, RELIABLE)
        self.create_timer(0.01, self._process)

    # ------------------------------------------------------------------ inputs

    def _on_config(self, message):
        if self.config is not None:
            return
        self.config = json.loads(message.data)
        self.demo = build_controller(self.config)
        self.get_logger().info(f"Controller built for {self.config['target_id']}")

    def _on_cycle(self, message):
        marker = json.loads(message.data)
        if marker.get("end"):
            self.finished = True
            return
        self.pending.setdefault(int(marker["stamp_ns"]), {})["marker"] = marker
        self._process()

    def _store(self, key, message):
        self.pending.setdefault(ns_of(message.header.stamp), {})[key] = message
        self._process()

    def _on_decision(self, message):
        self.decisions[ns_of(message.header.stamp)] = message
        self._process()

    # ------------------------------------------------------------------ cycle

    def _ready(self, stamp, bundle):
        if self.demo is None or not {"marker", "rgb", "depth", "info", "pose", "contact"} <= bundle.keys():
            return False
        # The verdict on the previous proposal must be known before observing again.
        return self.previous is None or self.previous[0] in self.decisions

    def _process(self):
        while self.pending:
            stamp = min(self.pending)
            if not self._ready(stamp, self.pending[stamp]):
                return
            bundle = self.pending.pop(stamp)
            started = time.monotonic()
            try:
                self._cycle(stamp, bundle)
            except Exception as error:  # noqa: BLE001 - a crash must become a published hold
                self.errors.append({"stamp_ns": stamp, "error": repr(error), "traceback": traceback.format_exc()})
                self.get_logger().error(f"Controller failed on cycle {stamp}: {error!r}")
                self._publish_failure(stamp, bundle, repr(error))
            self.timings.append(time.monotonic() - started)
            self.cycles += 1

    def _apply_previous_verdict(self):
        """Acknowledge or stop on the gate's verdict, as the recorder does after physics."""
        if self.previous is None:
            return None
        stamp, phase, decision = self.previous
        verdict = self.decisions.pop(stamp)
        for old in [key for key in self.decisions if key < stamp]:
            del self.decisions[old]
        if verdict.passed:
            self.demo.command_applied(phase, decision)
        else:
            reason = verdict.reason or verdict.latched_reason or "safety_gate_hold"
            self.demo.stop(reason)
            self.demo.command_applied("stopped_failure", {"state": "hold", "reason": reason})
        return verdict

    def _cycle(self, stamp, bundle):
        demo, marker = self.demo, bundle["marker"]
        verdict = self._apply_previous_verdict()
        if marker.get("external_stop"):
            demo.stop(str(marker["external_stop"]))

        rgb = image_to_array(bundle["rgb"])
        depth = image_to_array(bundle["depth"])
        camera_matrix = np.asarray(bundle["info"].k, dtype=float).reshape(3, 3)
        world_from_optical = np.asarray(marker["world_from_optical"], dtype=float)
        p, q = bundle["pose"].pose.position, bundle["pose"].pose.orientation
        tool = np.asarray([p.x, p.y, p.z, q.w, q.x, q.y, q.z], dtype=float)
        force = bundle["contact"].wrench.force
        hazard = math.sqrt(force.x**2 + force.y**2 + force.z**2) > HAZARD_CONTACT_FORCE_N
        time_s = float(marker["time_s"])

        if marker.get("initialize"):
            if marker.get("visible", True):
                self.initialization = demo.initialize(
                    rgb,
                    depth,
                    tuple(float(v) for v in marker["seed_pixel"]),
                    camera_matrix=camera_matrix,
                    world_from_optical=world_from_optical,
                    tool_pose_wxyz=tool,
                )
            else:
                self.initialization = {"state": "initialization_rejected", "reason": marker.get("visibility_reason")}
            # The recorder observes the preview without a contact measurement.
            hazard = False
        evidence = demo.observe(rgb, depth, camera_matrix, world_from_optical, time_s, tool, hazard_contact=hazard)
        observation = demo.last_cut_observation
        self.target_publisher.publish(target_estimate(observation, stamp, (demo.measurement or {}).get("state")))
        pose, phase, decision = demo.command(tool, time_s)
        decision = decision or {}
        self._publish_proposal(stamp, marker, pose, phase, decision)
        self.previous = (stamp, phase, decision)
        record = {
            "stamp_ns": stamp,
            "index": marker.get("index"),
            "live_vision": evidence,
            "phase": phase,
            "decision": decision,
            "previous_verdict": None
            if verdict is None
            else {"passed": verdict.passed, "reason": verdict.reason, "latched_reason": verdict.latched_reason},
            "vision_command_count": demo.vision_command_count,
        }
        if marker.get("initialize"):
            record["initialization"] = self.initialization
        message = String()
        message.data = json.dumps(record, default=_json_default, allow_nan=True)
        self.evidence_publisher.publish(message)

    def _publish_proposal(self, stamp, marker, pose, phase, decision):
        cut = self.demo.cut_step
        proposal = ControllerProposal()
        proposal.header.stamp, proposal.header.frame_id = to_time(stamp), T.FRAME_WORLD
        proposal.frame_index = max(0, int(marker.get("index", 0)))
        proposal.has_pose = True
        position, orientation = proposal.pose.position, proposal.pose.orientation
        position.x, position.y, position.z = (float(v) for v in pose[:3])
        orientation.w, orientation.x, orientation.y, orientation.z = (float(v) for v in pose[3:7])
        proposal.state = str(decision.get("state") or "")
        proposal.reason = str(decision.get("reason") or "")
        proposal.motion_phase = str(phase)
        proposal.cut_phase = str(cut.phase)
        proposal.cut_detached = bool(cut.detached)
        proposal.closure_progress = float(cut.closure_progress)
        self.proposal_publisher.publish(proposal)

        command = PoseStamped()
        command.header, command.pose = proposal.header, proposal.pose
        self.command_publisher.publish(command)
        advisory = String()
        advisory.data = json.dumps(dict(decision, phase=phase, input_mode="live_isaac"), default=_json_default)
        self.decision_publisher.publish(advisory)

    def _publish_failure(self, stamp, bundle, error):
        """A controller crash proposes no motion and says why; the gate then holds the arm."""
        proposal = ControllerProposal()
        proposal.header.stamp, proposal.header.frame_id = to_time(stamp), T.FRAME_WORLD
        proposal.has_pose = False
        proposal.state, proposal.reason = "hold", "controller_exception"
        proposal.cut_phase = "stopped"
        self.proposal_publisher.publish(proposal)
        estimate = TargetEstimate()
        estimate.header.stamp, estimate.header.frame_id = to_time(stamp), T.FRAME_WORLD
        estimate.observation_stamp = to_time(stamp)
        estimate.target_id, estimate.target_semantic = "", "unknown"
        estimate.vision_source = "live"
        estimate.position.x = estimate.position.y = estimate.position.z = math.nan
        self.target_publisher.publish(estimate)
        message = String()
        message.data = json.dumps({"stamp_ns": stamp, "error": error, "live_vision": None})
        self.evidence_publisher.publish(message)
        self.previous = (stamp, "stopped_failure", {"state": "hold", "reason": "controller_exception"})


def _json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    raise TypeError(f"Not JSON serializable: {type(value).__name__}")


def main(argv=None):
    rclpy.init(args=argv)
    node = LiveController()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.05)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
