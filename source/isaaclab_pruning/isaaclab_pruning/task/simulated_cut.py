"""Evidence-gated visual jaw closure and branch detachment, not wood fracture.

The caller supplies *live* perception measurements in one world frame and applies
the one-shot ``detach_event`` to a visual/rigid branch proxy. This module does not
model blades, cutting force, material failure, or hardware cutting capacity. It
also does not move the robot: ``phase`` tells its motion controller when to hold
for closure, retreat, or stop. Semantic identity is only as reliable as the
caller's perception source; a label supplied here is not independent validation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

MODEL_LABEL = "simulated_detachment_not_fracture"


def _vector(value: tuple[float, float, float], *, unit: bool = False) -> np.ndarray | None:
    vector = np.asarray(value, dtype=float)
    if vector.shape != (3,) or not np.isfinite(vector).all():
        return None
    if unit:
        norm = float(np.linalg.norm(vector))
        if norm <= 1e-12:
            return None
        vector = vector / norm
    return vector


@dataclass(frozen=True)
class CutConfig:
    """Demo geometry limits; ``max_branch_radius_m`` is not a hardware rating."""

    selected_target_id: str
    max_branch_radius_m: float = 0.007
    mouth_position_tolerance_m: float = 0.012
    alignment_tolerance_deg: float = 15.0
    stable_frames: int = 3
    closure_duration_s: float = 0.5
    vision_max_age_s: float = 0.25
    max_stable_speed_m_s: float = 0.025
    max_target_shift_m: float = 0.01
    max_stable_radius_change_m: float = 0.001
    allowed_semantics: tuple[str, ...] = ("branch", "spur")

    def __post_init__(self) -> None:
        if not self.selected_target_id.strip():
            raise ValueError("selected_target_id must name the selected branch.")
        positive_fields = (
            self.max_branch_radius_m,
            self.mouth_position_tolerance_m,
            self.closure_duration_s,
            self.vision_max_age_s,
            self.max_stable_speed_m_s,
            self.max_target_shift_m,
            self.max_stable_radius_change_m,
        )
        if any(not math.isfinite(value) or value <= 0 for value in positive_fields):
            raise ValueError("Geometry, time, and speed limits must be finite and positive.")
        if not math.isfinite(self.alignment_tolerance_deg) or not 0 <= self.alignment_tolerance_deg < 90:
            raise ValueError("alignment_tolerance_deg must be in [0, 90).")
        if isinstance(self.stable_frames, bool) or not isinstance(self.stable_frames, int) or self.stable_frames < 2:
            raise ValueError("stable_frames must be an integer >= 2.")
        if not self.allowed_semantics or any(value not in ("branch", "spur") for value in self.allowed_semantics):
            raise ValueError("Only explicitly identified branches/spurs can be cut.")


@dataclass(frozen=True)
class CutObservation:
    """Measured tool geometry and selected live vision target in world meters.

    The closing axis is the jaw's motion direction, so a branch must be
    *perpendicular* to it. The mouth position is not necessarily the tool origin.
    ``hazard_contact`` must exclude only intentionally permitted target contact;
    post/trunk, arm, and non-target wood contacts are hazards.
    """

    time_s: float
    vision_timestamp_s: float
    target_id: str
    target_semantic: str
    vision_valid: bool
    target_position_w: tuple[float, float, float]
    target_axis_w: tuple[float, float, float]
    target_radius_m: float
    mouth_position_w: tuple[float, float, float]
    cutter_closing_axis_w: tuple[float, float, float]
    hazard_contact: bool = False


@dataclass(frozen=True)
class CutCertificate:
    """Geometric gate only; the controller separately verifies temporal stability."""

    safe_to_approach: bool
    ready_to_close: bool
    reasons: tuple[str, ...]
    mouth_distance_m: float | None
    perpendicularity_error_deg: float | None
    vision_age_s: float | None
    model: str = MODEL_LABEL


def evaluate_cut_gate(observation: CutObservation, config: CutConfig) -> CutCertificate:
    """Fail closed on identity, age, contact, radius, or malformed geometry."""
    reasons: list[str] = []
    if not observation.vision_valid:
        reasons.append("vision_invalid")
    if observation.target_id != config.selected_target_id:
        reasons.append("wrong_target_id")
    if observation.target_semantic not in config.allowed_semantics:
        reasons.append("target_not_branch")
    if observation.hazard_contact:
        reasons.append("hazard_contact")
    age = observation.time_s - observation.vision_timestamp_s
    if not math.isfinite(age) or not 0 <= age <= config.vision_max_age_s:
        reasons.append("vision_stale_or_future")
    if not math.isfinite(observation.target_radius_m) or observation.target_radius_m <= 0:
        reasons.append("invalid_branch_radius")
    elif observation.target_radius_m > config.max_branch_radius_m:
        reasons.append("branch_exceeds_demo_cut_limit")

    target = _vector(observation.target_position_w)
    mouth = _vector(observation.mouth_position_w)
    branch_axis = _vector(observation.target_axis_w, unit=True)
    closing_axis = _vector(observation.cutter_closing_axis_w, unit=True)
    geometry_valid = all(value is not None for value in (target, mouth, branch_axis, closing_axis))
    distance = None
    alignment = None
    if not geometry_valid:
        reasons.append("invalid_geometry")
    else:
        distance = float(np.linalg.norm(target - mouth))
        alignment = math.degrees(math.asin(float(np.clip(abs(np.dot(branch_axis, closing_axis)), 0, 1))))
    safe = not reasons
    if distance is not None and distance > config.mouth_position_tolerance_m:
        reasons.append("outside_mouth_tolerance")
    if alignment is not None and alignment > config.alignment_tolerance_deg:
        reasons.append("closing_axis_not_perpendicular")
    return CutCertificate(
        safe_to_approach=safe,
        ready_to_close=not reasons,
        reasons=tuple(reasons),
        mouth_distance_m=distance,
        perpendicularity_error_deg=alignment,
        vision_age_s=age if math.isfinite(age) else None,
    )


def tool_mouth_geometry(
    tool_pose_wxyz: tuple[float, float, float, float, float, float, float],
    mouth_offset_tool: tuple[float, float, float] = (0.0, 0.0, 0.0),
    closing_axis_tool: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Transform a demo's explicitly specified mouth offset and jaw axis to world.

    The mock-pruner's current CAD mouth has not been calibrated. These defaults
    describe a zero-offset visual proxy, not its legacy STL-derived cutter box.
    """
    pose = np.asarray(tool_pose_wxyz, dtype=float)
    offset = _vector(mouth_offset_tool)
    axis = _vector(closing_axis_tool, unit=True)
    if pose.shape != (7,) or not np.isfinite(pose).all() or offset is None or axis is None:
        raise ValueError("Pose, mouth offset, and nonzero closing axis must be finite.")
    quaternion = pose[3:]
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-12:
        raise ValueError("Tool orientation quaternion must be nonzero.")
    w, x, y, z = quaternion / norm
    rotation = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )
    return tuple(float(value) for value in pose[:3] + rotation @ offset), tuple(
        float(value) for value in rotation @ axis
    )


@dataclass(frozen=True)
class CutStep:
    """Serializable state. Only ``detach_event`` authorizes one proxy detachment."""

    phase: str
    closure_progress: float
    stable_count: int
    detach_event: bool
    detached: bool
    closure_started_s: float | None
    detached_at_s: float | None
    stopped_reason: str | None
    certificate: CutCertificate
    model: str = MODEL_LABEL


class SimulatedCutController:
    """One selected target per reset: approach → align → closing → retreat.

    Hazard, lost/stale vision, or target replacement latches a stop. Losing the
    geometric gate during closure also stops without a detach event. Consecutive
    *new* camera observations must be slow and aligned before closure starts.
    After detachment, the caller owns safe retreat; target visibility may naturally
    disappear as the detached piece falls, but hazard contact still latches a stop.
    """

    def __init__(self, config: CutConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.phase = "approach"
        self.stable_count = 0
        self.closure_progress = 0.0
        self.closure_started_s: float | None = None
        self.detached_at_s: float | None = None
        self.stopped_reason: str | None = None
        self._previous: CutObservation | None = None
        self._closure_reference: CutObservation | None = None

    def _snapshot(self, certificate: CutCertificate, *, detach_event: bool = False) -> CutStep:
        return CutStep(
            phase=self.phase,
            closure_progress=self.closure_progress,
            stable_count=self.stable_count,
            detach_event=detach_event,
            detached=self.detached_at_s is not None,
            closure_started_s=self.closure_started_s,
            detached_at_s=self.detached_at_s,
            stopped_reason=self.stopped_reason,
            certificate=certificate,
        )

    def _stop(self, reason: str) -> None:
        self.phase = "stopped"
        self.stopped_reason = reason
        self.stable_count = 0

    def request_stop(self, reason: str) -> None:
        """Latch an external safety gate without inventing a contact measurement."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("A stop must record a nonempty reason")
        if self.phase != "stopped":
            self._stop(reason)

    def _stable_observation(self, observation: CutObservation) -> bool:
        previous = self._previous
        if previous is None:
            return True
        if observation.vision_timestamp_s <= previous.vision_timestamp_s:
            return False
        elapsed = observation.time_s - previous.time_s
        speed = float(
            np.linalg.norm(np.asarray(observation.mouth_position_w) - np.asarray(previous.mouth_position_w)) / elapsed
        )
        target_shift = float(
            np.linalg.norm(np.asarray(observation.target_position_w) - np.asarray(previous.target_position_w))
        )
        radius_change = abs(observation.target_radius_m - previous.target_radius_m)
        return (
            speed <= self.config.max_stable_speed_m_s
            and target_shift <= self.config.max_target_shift_m
            and radius_change <= self.config.max_stable_radius_change_m
        )

    def update(self, observation: CutObservation) -> CutStep:
        if not math.isfinite(observation.time_s):
            raise ValueError("time_s must be finite and strictly increasing.")
        if self._previous is not None and observation.time_s <= self._previous.time_s:
            raise ValueError("time_s must be finite and strictly increasing.")
        certificate = evaluate_cut_gate(observation, self.config)
        if self.phase == "stopped":
            return self._snapshot(certificate)
        if self.detached_at_s is not None:
            if observation.hazard_contact:
                self._stop("hazard_contact")
            self._previous = observation
            return self._snapshot(certificate)
        if not certificate.safe_to_approach:
            self._stop(certificate.reasons[0])
        elif self._previous is not None and observation.vision_timestamp_s < self._previous.vision_timestamp_s:
            self._stop("vision_timestamp_regressed")
        elif self.phase == "closing":
            if not certificate.ready_to_close:
                self._stop("gate_lost_during_closure")
            elif observation.vision_timestamp_s == self._previous.vision_timestamp_s:
                self._stop("vision_frame_reused_during_closure")
            elif (
                not self._stable_observation(observation)
                or np.linalg.norm(
                    np.asarray(observation.target_position_w) - np.asarray(self._closure_reference.target_position_w)
                )
                > self.config.max_target_shift_m
                or abs(observation.target_radius_m - self._closure_reference.target_radius_m)
                > self.config.max_stable_radius_change_m
            ):
                self._stop("unstable_during_closure")
            else:
                elapsed = observation.time_s - self.closure_started_s
                self.closure_progress = min(1.0, elapsed / self.config.closure_duration_s)
                if self.closure_progress >= 1.0:
                    self.detached_at_s = observation.time_s
                    self.phase = "retreat"
                    self._previous = observation
                    return self._snapshot(certificate, detach_event=True)
        elif certificate.ready_to_close:
            self.phase = "align"
            self.stable_count = self.stable_count + 1 if self._stable_observation(observation) else 0
            if self.stable_count >= self.config.stable_frames:
                self.phase = "closing"
                self.closure_started_s = observation.time_s
                self._closure_reference = observation
        else:
            self.stable_count = 0
            self.phase = (
                "align" if certificate.mouth_distance_m <= self.config.mouth_position_tolerance_m else "approach"
            )
        self._previous = observation
        return self._snapshot(certificate)
