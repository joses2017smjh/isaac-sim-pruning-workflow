"""Drive the C++ safety-gate node with recorded cycles and injected faults; compare with Python.

Inputs come from ``tools/export_gate_parity.py``: every control cycle rebuilt from
recorded Isaac captures, with the Python checks' verdict. For each capture this
starts ``safety_gate_node`` (the installed C++ executable), publishes each cycle's
six input messages over ROS 2, waits for the node's decision, and compares:

* the decision with Python: pass/hold, reason, latched reason, the ToF result and
  counter, the cut certificate (flags, reasons, distance, alignment, age);
* the forwarded command: a passing cycle forwards the proposal unchanged, a held
  cycle commands the measured tool pose, and the jaw never advances on a hold.

The fault scenarios replay one capture with a fault injected for ten cycles. Faults
the Python checks can see (ToF values, vision, contact) are also run through
``SafetyGateReference`` so the C++ node must agree with it; message-level faults
(malformed, missing, wrong frame) exist only in the node and are checked against
their stated rule. Every stopping fault must hold the arm from its first expected
cycle to the end of the run.

The proposal pose in this harness is synthetic (the measured pose moved 1 mm in x),
so a forwarded command is distinguishable from a hold. Run inside the SIL container
with the workspace sourced. CPU only.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import os
import platform
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import rclpy
from builtin_interfaces.msg import Time
from geometry_msgs.msg import PoseStamped, WrenchStamped
from pruning_sil_msgs.msg import ControllerProposal, OperatorHold, SafetyDecision, TargetEstimate
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Image, JointState

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "source/isaaclab_pruning"))

from isaaclab_pruning.task.safety_gate import GateCycle, SafetyGateReference  # noqa: E402
from isaaclab_pruning.task.simulated_cut import CutConfig, CutObservation, tool_mouth_geometry  # noqa: E402

RELIABLE = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=50,
    durability=QoSDurabilityPolicy.VOLATILE,
)
LATCHED = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)
WORLD = "world"
TOF_FRAMES = ("mock_pruner__tof0", "mock_pruner__tof1")
TOLERANCE = 1e-9
FAULT_START = 60
FAULT_CYCLES = 10
FAULT_TAIL = 6
PROPOSAL_SHIFT_M = 0.001
SETTLE_S = 1.5


def to_time(ns):
    return Time(sec=int(ns // 1_000_000_000), nanosec=int(ns % 1_000_000_000))


def ns_of(stamp):
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def stamp_seconds(ns):
    """Seconds exactly as the C++ node computes them from a stamp: sec + nanosec * 1e-9."""
    return float(ns // 1_000_000_000) + float(ns % 1_000_000_000) * 1e-9


def demo_config(target_id):
    # Must equal isaaclab_pruning.sim.vision_demo_controller.demo_cut_config (checked in tests).
    return CutConfig(
        selected_target_id=target_id,
        max_branch_radius_m=0.012,
        mouth_position_tolerance_m=0.008,
        stable_frames=4,
        closure_duration_s=0.6,
    )


# ----------------------------------------------------------------------------- messages


def range_image(values, frame, stamp_ns, *, encoding="32FC1", size=8):
    message = Image()
    message.header.stamp, message.header.frame_id = to_time(stamp_ns), frame
    grid = np.asarray([np.nan if v is None else v for v in values], dtype=np.float32)
    if size != 8:
        grid = grid[: size * size]
    message.height = message.width = size
    message.encoding = encoding
    message.is_bigendian = 0
    message.step = size * 4
    message.data = grid.astype("<f4").tobytes()
    return message


def pose_message(pose_wxyz, stamp_ns, frame=WORLD):
    message = PoseStamped()
    message.header.stamp, message.header.frame_id = to_time(stamp_ns), frame
    p, q = message.pose.position, message.pose.orientation
    p.x, p.y, p.z = (float(v) for v in pose_wxyz[:3])
    q.w, q.x, q.y, q.z = (float(v) for v in pose_wxyz[3:7])
    return message


def cycle_messages(record):
    """The six input messages of one cycle, as a sensor bridge and controller would publish them."""
    stamp = record["stamp_ns"]
    target = record["target"]
    out = {
        "tof_left": range_image(record["tof_left"], TOF_FRAMES[0], stamp),
        "tof_right": range_image(record["tof_right"], TOF_FRAMES[1], stamp),
        "tool_pose": pose_message(record["tool_pose_wxyz"], stamp),
    }
    wrench = WrenchStamped()
    wrench.header.stamp, wrench.header.frame_id = to_time(stamp), WORLD
    wrench.wrench.force.x = float(record["contact_force_n"])
    out["contact"] = wrench

    estimate = TargetEstimate()
    estimate.header.stamp = to_time(int(round(target["vision_time_s"] * 1e9)))
    estimate.header.frame_id = WORLD
    estimate.observation_stamp = to_time(stamp)
    estimate.target_id, estimate.target_semantic = target["id"], target["semantic"]
    estimate.vision_valid, estimate.vision_source = bool(target["valid"]), target["source"]
    if target["held_time_s"] is not None:
        estimate.has_held_target_stamp = True
        estimate.held_target_stamp = to_time(int(round(target["held_time_s"] * 1e9)))
    position = [math.nan if v is None else float(v) for v in target["position"]]
    estimate.position.x, estimate.position.y, estimate.position.z = position
    estimate.axis.x, estimate.axis.y, estimate.axis.z = (float(v) for v in target["axis"])
    estimate.radius_m = float(target["radius_m"])
    estimate.tracker_state = str(target.get("tracker_state") or "")
    out["target"] = estimate

    proposal = ControllerProposal()
    proposal.header.stamp, proposal.header.frame_id = to_time(stamp), WORLD
    proposal.frame_index = int(record["index"])
    proposal.has_pose = True
    shifted = list(record["tool_pose_wxyz"])
    shifted[0] += PROPOSAL_SHIFT_M
    proposal.pose = pose_message(shifted, stamp).pose
    proposal.state = str(record["proposal"].get("state") or "")
    proposal.cut_phase = str(record["proposal"]["cut_phase"] or "")
    proposal.cut_detached = bool(record["proposal"]["cut_detached"])
    proposal.closure_progress = float(record["proposal"]["closure_progress"])
    out["proposal"] = proposal
    return out


# ----------------------------------------------------------------------------- graph


class Harness(Node):
    def __init__(self):
        super().__init__("safety_gate_harness")
        self.pubs = {
            "tof_left": self.create_publisher(Image, "/pruning/tof/left/range_image", RELIABLE),
            "tof_right": self.create_publisher(Image, "/pruning/tof/right/range_image", RELIABLE),
            "tool_pose": self.create_publisher(PoseStamped, "/pruning/tool_pose", RELIABLE),
            "contact": self.create_publisher(WrenchStamped, "/pruning/contact/wrench", RELIABLE),
            "target": self.create_publisher(TargetEstimate, "/pruning/perception/target", RELIABLE),
            "proposal": self.create_publisher(ControllerProposal, "/pruning/proposed/proposal", RELIABLE),
        }
        self.decisions, self.commands, self.jaws = {}, {}, {}
        self.subs = [
            self.create_subscription(SafetyDecision, "/pruning/safety/decision", self._decision, RELIABLE),
            self.create_subscription(PoseStamped, "/pruning/gated/command_pose", self._command, RELIABLE),
            self.create_subscription(JointState, "/pruning/gated/jaw", self._jaw, RELIABLE),
        ]

    def _decision(self, message):
        self.decisions[ns_of(message.header.stamp)] = message

    def _command(self, message):
        self.commands[ns_of(message.header.stamp)] = message

    def _jaw(self, message):
        self.jaws[ns_of(message.header.stamp)] = message

    def reset(self):
        self.decisions.clear()
        self.commands.clear()
        self.jaws.clear()

    def connected(self):
        # Humble's Subscription has no publisher count; ask the graph by topic instead.
        return all(p.get_subscription_count() >= 1 for p in self.pubs.values()) and all(
            self.count_publishers(s.topic_name) >= 1 for s in self.subs
        )

    def spin_until(self, predicate, timeout_s):
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if predicate():
                return True
            rclpy.spin_once(self, timeout_sec=0.005)
        return predicate()


def start_gate(executable, record, *, input_timeout_s, initial_missing):
    target = record["target"]
    offset, closing = record["mouth_offset_tool_m"], record["closing_axis_tool"]

    def array(values):
        return "[" + ", ".join(repr(float(v)) for v in values) + "]"

    command = [
        executable,
        "--ros-args",
        "-p", f"selected_target_id:={target['id']}",
        "-p", f"mouth_offset_tool:={array(offset)}",
        "-p", f"closing_axis_tool:={array(closing)}",
        "-p", f"initial_cut_phase:={record['prior_cut_phase']}",
        "-p", f"initial_detached:={'true' if record['prior_detached'] else 'false'}",
        "-p", f"initial_missing:={int(initial_missing)}",
        "-p", f"input_timeout_s:={input_timeout_s}",
    ]  # fmt: skip
    return subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def stop_gate(process):
    if process.poll() is None:
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
    output = process.stdout.read() if process.stdout else ""
    return process.returncode, output[-4000:]


def seeded_missing(first):
    python = first["python"]
    return python["tof_missing_frames"] - 1 if python["tof_min_range_m"] is None else 0


def run_cycles(harness, executable, records, messages, *, input_timeout_s=5.0, before_cycle=None, wait_s=8.0):
    """Publish each cycle in order and collect what the node decided. Returns (rows, gate log)."""
    harness.reset()
    process = start_gate(
        executable, records[0], input_timeout_s=input_timeout_s, initial_missing=seeded_missing(records[0])
    )
    try:
        if not harness.spin_until(harness.connected, 30.0):
            raise RuntimeError("safety_gate_node did not connect within 30 s")
        # Matching is reported per side; give the node's readers time to match every writer, or the
        # first reliable samples can be published before the node is listening.
        harness.spin_until(lambda: False, SETTLE_S)
        rows = []
        for record, cycle in zip(records, messages):
            if before_cycle is not None:
                before_cycle(record)
            stamp = record["stamp_ns"]
            started = time.monotonic()
            for name, message in cycle.items():
                if message is not None:
                    harness.pubs[name].publish(message)
            got = harness.spin_until(lambda: stamp in harness.decisions and stamp in harness.jaws, wait_s)
            rows.append(
                {
                    "record": record,
                    "sent": cycle,
                    "decision": harness.decisions.get(stamp),
                    "command": harness.commands.get(stamp),
                    "jaw": harness.jaws.get(stamp),
                    "received": got,
                    "round_trip_s": time.monotonic() - started,
                }
            )
    finally:
        code, log = stop_gate(process)
    return rows, {"returncode": code, "log_tail": log}


# ----------------------------------------------------------------------------- comparison


def _same(left, right):
    left = None if left is None or (isinstance(left, float) and math.isnan(left)) else left
    right = None if right is None or (isinstance(right, float) and math.isnan(right)) else right
    if left is None or right is None:
        return left is None and right is None
    return abs(float(left) - float(right)) <= TOLERANCE * max(1.0, abs(float(right)))


def compare_decision(decision, python):
    """Fields where the C++ decision differs from the Python verdict."""
    if decision is None:
        return ["no_decision"]
    certificate = python["certificate"]
    differ = []
    checks = {
        "passed": decision.passed == python["passed"],
        "reason": decision.reason == (python["reason"] or ""),
        "latched_reason": decision.latched_reason == (python["latched_reason"] or ""),
        "tof_reason": decision.tof_reason == (python["tof_reason"] or ""),
        "tof_missing_frames": decision.tof_missing_frames == python["tof_missing_frames"],
        "tof_min_range_m": _same(decision.tof_min_range_m, python["tof_min_range_m"]),
        "cut_safe_to_approach": decision.cut_safe_to_approach == certificate["safe_to_approach"],
        "cut_ready_to_close": decision.cut_ready_to_close == certificate["ready_to_close"],
        "cut_reasons": list(decision.cut_reasons) == list(certificate["reasons"]),
        "mouth_distance_m": _same(decision.mouth_distance_m, certificate["mouth_distance_m"]),
        "perpendicularity_error_deg": _same(
            decision.perpendicularity_error_deg, certificate["perpendicularity_error_deg"]
        ),
        "vision_age_s": _same(decision.vision_age_s, certificate["vision_age_s"]),
        "vision_source": decision.vision_source == certificate["vision_source"],
        "waived_checks": list(decision.waived_checks) == list(certificate["waived_checks"]),
        "closure_authorized": decision.closure_authorized == python["closure_authorized"],
    }
    differ = [key for key, ok in checks.items() if not ok]
    return differ


def pose_tuple(message):
    p, q = message.pose.position, message.pose.orientation
    return (p.x, p.y, p.z, q.w, q.x, q.y, q.z)


def usable_pose(message):
    """The node's pose check: world frame, finite, unit quaternion within 1e-3."""
    if message is None or message.header.frame_id != WORLD:
        return False
    values = pose_tuple(message)
    if not all(math.isfinite(v) for v in values):
        return False
    return abs(math.sqrt(sum(v * v for v in values[3:])) - 1.0) <= 1e-3


def check_outputs(rows):
    """Forwarding rule: pass -> the proposal; hold -> the last usable measured pose; no jaw advance on a hold.

    Returns problem counts and the number of held cycles whose command was not the hold pose.
    """
    problems = Counter()
    gated_jaw = 0.0
    hold_pose = None
    moved_while_held = 0
    for row in rows:
        decision, command, jaw, sent = row["decision"], row["command"], row["jaw"], row["sent"]
        if usable_pose(sent.get("tool_pose")):
            hold_pose = pose_tuple(sent["tool_pose"])
        if decision is None or jaw is None:
            problems["missing_output"] += 1
            continue
        if decision.passed:
            proposal = sent["proposal"]
            expected = pose_tuple(proposal) if proposal is not None else hold_pose
        else:
            expected = hold_pose
        if command is None:
            if expected is not None:
                problems["missing_command"] += 1
        elif pose_tuple(command) != expected:
            problems["command_not_forwarding_rule"] += 1
            moved_while_held += int(not decision.passed)
        proposal = sent.get("proposal")
        proposed = float(proposal.closure_progress) if proposal is not None else gated_jaw
        expected_jaw = proposed if decision.passed else gated_jaw
        if jaw.position[0] != expected_jaw:
            problems["jaw_not_forwarding_rule"] += 1
        gated_jaw = jaw.position[0]
    return problems, moved_while_held


# ----------------------------------------------------------------------------- Python reference for faults


def python_reference(records):
    """Run SafetyGateReference on (possibly faulted) records; returns the verdict dicts."""
    first = records[0]
    reference = SafetyGateReference(demo_config(first["target"]["id"]))
    reference.missing = seeded_missing(first)
    out = []
    for record in records:
        target = record["target"]
        mouth, closing = tool_mouth_geometry(
            tuple(record["tool_pose_wxyz"]), tuple(record["mouth_offset_tool_m"]), tuple(record["closing_axis_tool"])
        )
        observation = CutObservation(
            time_s=stamp_seconds(record["stamp_ns"]),
            vision_timestamp_s=stamp_seconds(int(round(target["vision_time_s"] * 1e9))),
            target_id=target["id"],
            target_semantic=target["semantic"],
            vision_valid=bool(target["valid"]),
            target_position_w=tuple(math.nan if v is None else v for v in target["position"]),
            target_axis_w=tuple(target["axis"]),
            target_radius_m=float(target["radius_m"]),
            mouth_position_w=mouth,
            cutter_closing_axis_w=closing,
            hazard_contact=float(record["contact_force_n"]) > 5.0,
            vision_source=target["source"],
            held_target_timestamp_s=None
            if target["held_time_s"] is None
            else stamp_seconds(int(round(target["held_time_s"] * 1e9))),
        )
        ranges = np.asarray(
            [math.nan if v is None else v for v in record["tof_left"] + record["tof_right"]], dtype=float
        )
        verdict = reference.step(
            GateCycle(
                tof_ranges=ranges,
                tof_valid=np.isfinite(ranges),
                observation=observation,
                prior_cut_phase=record["prior_cut_phase"],
                prior_detached=record["prior_detached"],
            )
        )
        certificate = verdict.certificate
        out.append(
            {
                "passed": verdict.passed,
                "reason": verdict.reason,
                "latched_reason": verdict.latched_reason,
                "tof_reason": verdict.tof_reason,
                "tof_missing_frames": verdict.tof_missing_frames,
                "tof_min_range_m": verdict.tof_min_range_m,
                "closure_authorized": verdict.closure_authorized,
                "certificate": {
                    "safe_to_approach": certificate.safe_to_approach,
                    "ready_to_close": certificate.ready_to_close,
                    "reasons": list(certificate.reasons),
                    "mouth_distance_m": certificate.mouth_distance_m,
                    "perpendicularity_error_deg": certificate.perpendicularity_error_deg,
                    "vision_age_s": certificate.vision_age_s,
                    "vision_source": certificate.vision_source,
                    "waived_checks": list(certificate.waived_checks),
                },
            }
        )
    return out


def _with_target(record, **changes):
    return {**record, "target": {**record["target"], **changes}}


def _tof(record, left=None, right=None):
    out = dict(record)
    if left is not None:
        out["tof_left"] = left
    if right is not None:
        out["tof_right"] = right
    return out


GARBAGE = [math.nan, math.inf, -0.5, 0.0] * 16


def _shift_vision(seconds):
    return lambda r: _with_target(r, vision_time_s=r["target"]["vision_time_s"] + seconds)


def _set_frame(frame):
    return lambda m: setattr(m.header, "frame_id", frame)

# name -> (record mutation or None, message mutation or None, expected reason, first-hold offset, Python-visible)
FAULTS = {
    "tof_both_sensors_dead": (lambda r: _tof(r, [None] * 64, [None] * 64), None, "both_tof_missing_4_frames", 3, True),
    "tof_garbage_ranges": (lambda r: _tof(r, GARBAGE, GARBAGE), None, "both_tof_missing_4_frames", 3, True),
    "tof_obstacle_3cm": (lambda r: _tof(r, [0.03] + list(r["tof_left"][1:])), None, "tof_minimum_clearance", 0, True),
    "vision_stale_300ms": (_shift_vision(-0.30), None, "vision_stale_or_future", 0, True),
    "vision_from_future_50ms": (_shift_vision(0.05), None, "vision_stale_or_future", 0, True),
    "vision_lost": (lambda r: _with_target(r, valid=False), None, "vision_invalid", 0, True),
    "wrong_target_id": (lambda r: _with_target(r, id="tree1_SPUR_component_19444"), None, "wrong_target_id", 0, True),
    "target_not_a_branch": (lambda r: _with_target(r, semantic="trunk"), None, "target_not_branch", 0, True),
    "branch_too_thick_20mm": (lambda r: _with_target(r, radius_m=0.02), None, "branch_exceeds_demo_cut_limit", 0, True),
    "target_position_nan": (lambda r: _with_target(r, position=[None, None, None]), None, "invalid_geometry", 0, True),
    "hazard_contact_12N": (lambda r: {**r, "contact_force_n": 12.0}, None, "hazard_contact", 0, True),
    "tof_wrong_encoding": (None, ("tof_left", lambda m: setattr(m, "encoding", "mono8")), "invalid_tof_left", 0, False),
    "tof_wrong_size": (None, ("tof_right", "resize"), "invalid_tof_right", 0, False),
    "tof_wrong_frame": (None, ("tof_left", _set_frame("base_link")), "invalid_tof_left", 0, False),
    "tof_message_dropped": (None, ("tof_left", "drop"), "input_missing:tof_left", 0, False),
    "tool_pose_zero_quaternion": (None, ("tool_pose", "zero_quaternion"), "invalid_tool_pose", 0, False),
    "tool_pose_wrong_frame": (None, ("tool_pose", _set_frame("base_link")), "invalid_tool_pose", 0, False),
    "contact_nan": (None, ("contact", lambda m: setattr(m.wrench.force, "x", math.nan)), "invalid_contact", 0, False),
    "proposal_dropped": (None, ("proposal", "drop"), "input_missing:proposal", 0, False),
    "target_dropped": (None, ("target", "drop"), "input_missing:target", 0, False),
    "target_wrong_frame": (None, ("target", _set_frame("base_link")), "target_frame_mismatch", 0, False),
    "closure_without_ready_gate": (None, ("proposal", "closure"), "closure_without_ready_gate", 0, False),
    "operator_hold": (None, ("operator", None), "operator_hold", 0, False),
}  # fmt: skip
#: Not stops by design, reported so a reader sees what the gate does *not* catch.
NON_STOPPING = {
    "tof_one_sensor_dead": (lambda r: _tof(r, left=[None] * 64), None, None, None, True),
}


def mutate_message(messages, spec, record):
    name, action = spec
    if name == "operator":
        return messages
    message = messages.get(name)
    if action == "drop":
        messages[name] = None
    elif action == "resize":
        messages[name] = range_image(record["tof_right"], TOF_FRAMES[1], record["stamp_ns"], size=4)
    elif action == "zero_quaternion":
        q = message.pose.orientation
        q.w = q.x = q.y = q.z = 0.0
    elif action == "closure":
        message.closure_progress = 0.5
    else:
        action(message)
    return messages


def run_fault(harness, executable, records, name, spec):
    record_mutation, message_spec, reason, offset, python_visible = spec
    window = range(FAULT_START, FAULT_START + FAULT_CYCLES)
    selected = records[: FAULT_START + FAULT_CYCLES + FAULT_TAIL]
    faulted = [record_mutation(r) if (record_mutation and r["index"] in window) else r for r in selected]
    messages = []
    for record in faulted:
        cycle = cycle_messages(record)
        if message_spec and record["index"] in window:
            cycle = mutate_message(cycle, message_spec, record)
        messages.append(cycle)

    operator = message_spec is not None and message_spec[0] == "operator"
    # Created per scenario and destroyed after: a latched hold must not leak into the next run.
    hold_publisher = harness.create_publisher(OperatorHold, "/pruning/operator/hold", LATCHED) if operator else None

    def before(record):
        if operator and record["index"] == FAULT_START:
            hold = OperatorHold()
            hold.header.stamp = harness.get_clock().now().to_msg()
            hold.hold, hold.source = True, "validate_safety_gate"
            hold_publisher.publish(hold)
            harness.spin_until(lambda: False, 0.3)  # let the hold arrive before cycle FAULT_START

    try:
        rows, log = run_cycles(harness, executable, faulted, messages, input_timeout_s=0.5, before_cycle=before)
    finally:
        if hold_publisher is not None:
            harness.destroy_publisher(hold_publisher)
    forwarding, moved_while_held = check_outputs(rows)
    held = [row for row in rows if row["decision"] is not None and not row["decision"].passed]
    first = held[0] if held else None
    affected = [row for row in rows if row["record"]["index"] in window]
    first_expected = FAULT_START + (offset or 0)
    outcome = {
        "expected_reason": reason,
        "expected_first_hold_index": first_expected if reason else None,
        "first_hold_index": first["record"]["index"] if first else None,
        "first_hold_reason": first["decision"].reason if first else None,
        "affected_cycles": len(affected),
        "affected_cycles_held": sum(
            1 for row in affected if row["decision"] is not None and not row["decision"].passed
        ),
        "held_through_end": bool(first)
        and all(
            row["decision"] is not None and not row["decision"].passed
            for row in rows
            if row["record"]["index"] >= first["record"]["index"]
        ),
        "forwarding_problems": dict(forwarding),
        "held_cycles_commanding_motion": moved_while_held,
        "gate_returncode": log["returncode"],
    }
    if reason is None:
        outcome["ok"] = not held and not outcome["forwarding_problems"]
    else:
        outcome["ok"] = (
            outcome["first_hold_index"] == first_expected
            and outcome["first_hold_reason"] == reason
            and outcome["held_through_end"]
            and not outcome["forwarding_problems"]
        )
    outcome["ok"] = outcome["ok"] and moved_while_held == 0
    if python_visible:
        expected = python_reference(faulted)
        differ = Counter()
        for row, verdict in zip(rows, expected):
            for key in compare_decision(row["decision"], verdict):
                differ[key] += 1
        outcome["python_reference_cycles"] = len(expected)
        outcome["python_reference_mismatches"] = dict(differ)
        outcome["ok"] = outcome["ok"] and not differ
    return outcome


# ----------------------------------------------------------------------------- main


def load_captures(path, only=None):
    captures = defaultdict(list)
    digest = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        for line in stream:
            digest.update(line.rstrip("\n").encode())
            record = json.loads(line)
            if only and record["capture"] not in only:
                continue
            captures[record["capture"]].append(record)
    return captures, digest.hexdigest()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cycles", required=True, help="cycles_all.jsonl.gz from export_gate_parity.py")
    parser.add_argument("--gate", required=True, help="installed safety_gate_node executable")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fault-capture", default="artifacts/isaac_render/job_21328323")
    parser.add_argument("--only", nargs="*", help="limit the parity sweep to these captures")
    parser.add_argument("--skip-parity", action="store_true")
    args = parser.parse_args(argv)

    captures, digest = load_captures(args.cycles, set(args.only) if args.only else None)
    if args.fault_capture not in captures:
        fault_only, _ = load_captures(args.cycles, {args.fault_capture})
        captures.update(fault_only)
    rclpy.init()
    harness = Harness()
    evidence = {
        "schema_version": 1,
        "tool": "tools/validate_safety_gate.py",
        "cycles_sha256": digest,
        "gate_executable": args.gate,
        "host": platform.node(),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID"),
        "tolerance_continuous": TOLERANCE,
        "proposal_pose": f"synthetic: measured pose + {PROPOSAL_SHIFT_M} m in x",
    }
    totals = Counter()
    mismatch_examples = []
    per_capture = {}
    started = time.monotonic()
    try:
        if not args.skip_parity:
            for label, records in sorted(captures.items()):
                if args.only and label not in args.only:
                    continue
                messages = [cycle_messages(record) for record in records]
                rows, log = run_cycles(harness, args.gate, records, messages)
                counts = Counter()
                for row in rows:
                    counts["cycles"] += 1
                    differ = compare_decision(row["decision"], row["record"]["python"])
                    if differ:
                        counts["mismatched_cycles"] += 1
                        for key in differ:
                            counts[f"differs_{key}"] += 1
                        if len(mismatch_examples) < 50:
                            mismatch_examples.append(
                                {"capture": label, "index": row["record"]["index"], "fields": differ}
                            )
                    else:
                        counts["matched_cycles"] += 1
                    counts["python_holds"] += int(not row["record"]["python"]["passed"])
                    counts["cpp_holds"] += int(row["decision"] is not None and not row["decision"].passed)
                    counts["round_trip_ms_max"] = max(counts["round_trip_ms_max"], int(row["round_trip_s"] * 1000))
                forwarding, _ = check_outputs(rows)
                counts.update({f"forwarding_{k}": v for k, v in forwarding.items()})
                counts["gate_returncode_nonzero"] = int(log["returncode"] not in (0, -2, None))
                per_capture[label] = dict(counts)
                for key, value in counts.items():
                    if key == "round_trip_ms_max":
                        totals[key] = max(totals[key], value)
                    else:
                        totals[key] += value
                totals["captures"] += 1
                print(f"{label}: {counts['matched_cycles']}/{counts['cycles']} matched", flush=True)
        evidence["parity"] = {
            "totals": dict(totals),
            "per_capture": per_capture,
            "mismatch_examples": mismatch_examples,
        }

        fault_records = captures[args.fault_capture]
        faults = {}
        for name, spec in {**FAULTS, **NON_STOPPING}.items():
            faults[name] = run_fault(harness, args.gate, fault_records, name, spec)
            print(
                f"fault {name}: ok={faults[name]['ok']} first={faults[name]['first_hold_index']} "
                f"reason={faults[name]['first_hold_reason']}",
                flush=True,
            )
        stopping = [name for name in FAULTS]
        evidence["faults"] = {
            "capture": args.fault_capture,
            "window": {"first_index": FAULT_START, "cycles": FAULT_CYCLES, "clean_tail_cycles": FAULT_TAIL},
            "scenarios": faults,
            "stopping_faults": len(stopping),
            "stopping_faults_held": sum(1 for name in stopping if faults[name]["ok"]),
            "non_stopping_documented": list(NON_STOPPING),
        }
    finally:
        harness.destroy_node()
        rclpy.try_shutdown()
    evidence["elapsed_s"] = round(time.monotonic() - started, 1)
    parity_ok = args.skip_parity or (
        totals["mismatched_cycles"] == 0
        and not any(key.startswith("forwarding_") and value for key, value in totals.items())
        and totals["cycles"] > 0
    )
    faults_ok = all(item["ok"] for item in evidence["faults"]["scenarios"].values())
    evidence["passed"] = bool(parity_ok and faults_ok)
    Path(args.output).write_text(json.dumps(evidence, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "passed": evidence["passed"],
                "totals": dict(totals),
                "faults_held": evidence["faults"]["stopping_faults_held"],
                "faults": evidence["faults"]["stopping_faults"],
            },
            indent=2,
        )
    )
    return 0 if evidence["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
