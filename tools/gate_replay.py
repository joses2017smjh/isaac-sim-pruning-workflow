#!/usr/bin/env python3
"""The Python reference gate replayed over rebuilt cycle records, with the node's own jaw rule. No ROS.

``tools/export_gate_parity.py`` rebuilds one record per recorded control cycle. ``replay`` runs
``SafetyGateReference`` over such records exactly as ``tools/validate_safety_gate.py`` feeds the C++ node, so the
harness (inside the ROS container) and the scorers (outside it) share one definition of what the gate should say.

``node_rules=True`` adds the one rule the node applies on top of the reference: the jaw output may advance only
on a passing cycle that authorizes closure; otherwise the node latches ``closure_without_ready_gate``. A node
latch is written into the reference, as the node's ``gate_->latch`` writes it into the C++ gate.

Jaw presence: a record may carry ``jaw_beams``, nine characters, one per beam of
``isaaclab_pruning.sensors.jaw_beam_grid`` ('.' clear, anything else blocked). ``jaw_min_beams`` above 0 switches
the reference's jaw-presence rule on and feeds it the count of blocked beams.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO / "source/isaaclab_pruning") not in sys.path:
    sys.path.insert(0, str(REPO / "source/isaaclab_pruning"))

from isaaclab_pruning.task.safety_gate import GateCycle, SafetyGateReference  # noqa: E402
from isaaclab_pruning.task.simulated_cut import CutConfig, CutObservation, tool_mouth_geometry  # noqa: E402

CLOSURE_WITHOUT_READY_GATE = "closure_without_ready_gate"
CLEAR = "."


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


def seeded_missing(first):
    python = first["python"]
    return python["tof_missing_frames"] - 1 if python["tof_min_range_m"] is None else 0


def blocked_count(codes):
    """Blocked beams in a reading; None when the record has no reading."""
    return None if codes is None else sum(1 for code in codes if code != CLEAR)


def cycle_of(record, jaw_key=None):
    """One record as the gate sees it."""
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
    ranges = np.asarray([math.nan if v is None else v for v in record["tof_left"] + record["tof_right"]], dtype=float)
    return GateCycle(
        tof_ranges=ranges,
        tof_valid=np.isfinite(ranges),
        observation=observation,
        prior_cut_phase=record["prior_cut_phase"],
        prior_detached=record["prior_detached"],
        jaw_beams_blocked=blocked_count(record.get(jaw_key)) if jaw_key else None,
    )


def replay(records, *, jaw_min_beams=0, jaw_key=None, node_rules=False):
    """Verdict dicts, one per record. With ``node_rules`` each also carries the node's ``gated_jaw``."""
    first = records[0]
    reference = SafetyGateReference(demo_config(first["target"]["id"]), jaw_presence_min_beams=int(jaw_min_beams))
    reference.missing = seeded_missing(first)
    gated_jaw = 0.0
    out = []
    for record in records:
        verdict = reference.step(cycle_of(record, jaw_key))
        passed, reason, latched = verdict.passed, verdict.reason, verdict.latched_reason
        authorized = verdict.closure_authorized
        if node_rules and passed:
            proposed = float(record["proposal"]["closure_progress"])
            if proposed > gated_jaw and not authorized:
                reference.latched_reason = CLOSURE_WITHOUT_READY_GATE
                passed, reason, latched, authorized = (
                    False,
                    CLOSURE_WITHOUT_READY_GATE,
                    CLOSURE_WITHOUT_READY_GATE,
                    False,
                )
            else:
                gated_jaw = proposed
        certificate = verdict.certificate
        row = {
            "passed": passed,
            "reason": reason,
            "latched_reason": latched,
            "tof_reason": verdict.tof_reason,
            "tof_missing_frames": verdict.tof_missing_frames,
            "tof_min_range_m": verdict.tof_min_range_m,
            "closure_authorized": authorized,
            "jaw_presence_ok": verdict.jaw_presence_ok,
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
        if node_rules:
            row["gated_jaw"] = gated_jaw
        out.append(row)
    return out


def first_hold(verdicts, records):
    """``(record index, reason)`` of the first held cycle, or ``(None, None)``."""
    for record, verdict in zip(records, verdicts):
        if not verdict["passed"]:
            return record["index"], verdict["reason"]
    return None, None


def jaw_advances(records):
    """Indices of the cycles in which the recorded controller advanced the jaw."""
    high, out = 0.0, []
    for record in records:
        progress = float(record["proposal"]["closure_progress"])
        if progress > high:
            out.append(record["index"])
            high = progress
    return out
