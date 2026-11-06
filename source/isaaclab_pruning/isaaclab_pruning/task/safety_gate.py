"""The two stop-or-go checks, and the rules that combine them into one gate call.

``sensor_guard`` is the time-of-flight check the Isaac recorder runs on every fresh
sample (moved here unchanged from ``hpc/inner/render_pruning_workflow.py``).
``evaluate_cut_gate`` lives in ``simulated_cut``. ``SafetyGateReference`` combines
them the way the recorder and ``SimulatedCutController`` do, so a separate
implementation (the C++ ROS 2 node in ``ros2/pruning_sil_cpp``) can be checked
against Python call for call.

The reference covers only the stops these two checks cause. The cutter's own
temporal stops (instability or a reused frame during closure, timestamp
regression, closure-hold source checks) stay in the Python controller; the gate
never turns a controller hold into motion, and it does not need to repeat them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .simulated_cut import CutCertificate, CutConfig, CutObservation, evaluate_cut_gate

#: Below this measured ToF range the arm stops. A demo limit, not a hardware rating.
MINIMUM_CLEARANCE_M = 0.06
#: Consecutive cycles with no usable zone on either sensor before the arm stops.
MISSING_FRAMES_LIMIT = 4
TOF_REASONS = ("tof_minimum_clearance", "both_tof_missing_4_frames")
OPERATOR_HOLD = "operator_hold"
GATE_LOST_DURING_CLOSURE = "gate_lost_during_closure"


def sensor_guard(ranges, valid, consecutive_missing, *, minimum_clearance_m=MINIMUM_CLEARANCE_M):
    """Safety gate on measured live ToF; the scripted goal is still geometry-derived.

    Returns ``(reason or None, consecutive_missing)``. A zone counts only when it is
    flagged valid, finite and positive. Both sensors' zones are pooled.
    """
    ranges, valid = np.asarray(ranges, dtype=float), np.asarray(valid, dtype=bool)
    samples = ranges[valid & np.isfinite(ranges) & (ranges > 0.0)]
    missing = consecutive_missing + 1 if not samples.size else 0
    if samples.size and samples.min() < minimum_clearance_m:
        return "tof_minimum_clearance", missing
    if missing >= MISSING_FRAMES_LIMIT:
        return "both_tof_missing_4_frames", missing
    return None, missing


@dataclass(frozen=True)
class GateCycle:
    """Everything the gate sees in one control cycle.

    ``prior_cut_phase`` and ``prior_detached`` are the cutter's state *before* this
    cycle's observation, as the controller reported it on the previous cycle.
    """

    tof_ranges: np.ndarray
    tof_valid: np.ndarray
    observation: CutObservation
    prior_cut_phase: str = "approach"
    prior_detached: bool = False
    operator_hold: bool = False


@dataclass(frozen=True)
class GateVerdict:
    passed: bool
    reason: str | None
    latched_reason: str | None
    tof_reason: str | None
    tof_missing_frames: int
    tof_min_range_m: float | None
    certificate: CutCertificate
    closure_authorized: bool


class SafetyGateReference:
    """Python reference for the gate call. Latches the first stop, as the recorder does."""

    def __init__(self, config: CutConfig, *, minimum_clearance_m: float = MINIMUM_CLEARANCE_M):
        self.config = config
        self.minimum_clearance_m = float(minimum_clearance_m)
        self.missing = 0
        self.latched_reason: str | None = None

    def step(self, cycle: GateCycle) -> GateVerdict:
        tof_reason, self.missing = sensor_guard(
            cycle.tof_ranges, cycle.tof_valid, self.missing, minimum_clearance_m=self.minimum_clearance_m
        )
        ranges = np.asarray(cycle.tof_ranges, dtype=float)
        usable = ranges[np.asarray(cycle.tof_valid, dtype=bool) & np.isfinite(ranges) & (ranges > 0.0)]
        certificate = evaluate_cut_gate(cycle.observation, self.config)

        # Precedence follows the recorder: the operator, then ToF (checked before the
        # observation), then the cut gate inside SimulatedCutController.update.
        reason = None
        if cycle.operator_hold:
            reason = OPERATOR_HOLD
        elif tof_reason is not None:
            reason = tof_reason
        elif cycle.prior_detached:
            # After detachment the target may leave view; only contact still stops.
            if cycle.observation.hazard_contact:
                reason = "hazard_contact"
        elif not certificate.safe_to_approach:
            reason = certificate.reasons[0]
        elif cycle.prior_cut_phase == "closing" and not certificate.ready_to_close:
            reason = GATE_LOST_DURING_CLOSURE
        if reason is not None and self.latched_reason is None:
            self.latched_reason = reason
        passed = self.latched_reason is None
        return GateVerdict(
            passed=passed,
            reason=None if passed else (reason or self.latched_reason),
            latched_reason=self.latched_reason,
            tof_reason=tof_reason,
            tof_missing_frames=self.missing,
            tof_min_range_m=float(usable.min()) if usable.size else None,
            certificate=certificate,
            closure_authorized=passed and (cycle.prior_detached or certificate.ready_to_close),
        )
