"""Causal RGB-D branch tracking and a labelled simulated-detachment demo.

Only the initial pixel and branch identity/axis/radius are supplied by the scene.
Subsequent approach positions come from live image tracking and rendered depth.
This is classical visual servoing, not learned branch recognition or fracture.

Three labelled, default-off arms change what the demo measures or certifies:
``jaw_self_mask`` (the tracker ignores the robot's own predicted jaw pixels),
``closure_hold`` (a cut-gate change: during closure only, a loss the jaw mask
explains is replaced by the closure-start target, recorded as such) and
``depth_appearance`` (D_strict + J: when the grey appearance check fails, a
static-world depth test and the jaw silhouette decide whether the 0.35 gate
passes). ``depth_appearance_arm`` picks that check's registered arm: ``strict``
(the default; the raw NCC stays in the confidence) or ``agreement`` (on an
accepted frame the depth-agreement fraction replaces it). With all off, every
call, input and output is what it was before they existed.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace

import numpy as np

from isaaclab_pruning.perception.depth_appearance import ARMS as DEPTH_APPEARANCE_ARMS
from isaaclab_pruning.perception.depth_appearance import DepthAppearanceTracker, frame_jaw_boxes
from isaaclab_pruning.perception.jaw_self_mask import (
    HOLD_MAX_ROTATION_DEG,
    HOLD_MAX_TRANSLATION_M,
    MIN_UNMASKED_PATCH_ELEMENTS,
    closure_hold_explanation,
    jaw_boxes,
    jaw_gap_m,
    jaw_mask,
    jaw_mask_summary,
    pose_change,
    proxy_roll_rad,
)
from isaaclab_pruning.perception.visual_servo import (
    VisualServoConfig,
    VisualServoTracker,
    bounded_visual_servo_translation,
)
from isaaclab_pruning.task.simulated_cut import (
    CutConfig,
    CutObservation,
    SimulatedCutController,
    tool_mouth_geometry,
)

APPROACH_MODES = ("straight", "tool_axis_standoff", "horizontal_standoff", "planned_pose_standoff")
STANDOFF_REACHED_M = 0.005
ORIENTATION_REACHED_DEG = 1.0


def quat_to_matrix(q_wxyz):
    w, x, y, z = np.asarray(q_wxyz, dtype=float) / np.linalg.norm(q_wxyz)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_angle_deg(q0, q1):
    """Rotation angle between two orientations, in degrees."""
    a, b = np.asarray(q0, dtype=float), np.asarray(q1, dtype=float)
    dot = abs(float(np.dot(a / np.linalg.norm(a), b / np.linalg.norm(b))))
    return float(np.degrees(2.0 * np.arccos(min(1.0, dot))))


def rotate_toward(q_from, q_to, max_deg):
    """Slerp from ``q_from`` toward ``q_to`` by at most ``max_deg``; unit wxyz out."""
    a = np.asarray(q_from, dtype=float) / np.linalg.norm(q_from)
    b = np.asarray(q_to, dtype=float) / np.linalg.norm(q_to)
    if np.dot(a, b) < 0:
        b = -b
    angle = quat_angle_deg(a, b)
    if angle <= max_deg or angle < 1e-9:
        return b
    t = max_deg / angle
    omega = np.arccos(min(1.0, float(np.dot(a, b))))
    out = (np.sin((1 - t) * omega) * a + np.sin(t * omega) * b) / np.sin(omega)
    return out / np.linalg.norm(out)


def closing_axis_tool_at(tool_quat_wxyz, branch_axis_w):
    """The jaw closing axis, in the tool frame, perpendicular to both the tool axis and the branch at a pose."""
    rotation = quat_to_matrix(tool_quat_wxyz)
    closing_w = np.cross(rotation[:, 2], np.asarray(branch_axis_w, dtype=float))
    norm = np.linalg.norm(closing_w)
    if norm < 1e-9:
        raise ValueError("The tool axis is parallel to the branch at this pose")
    return tuple(float(v) for v in rotation.T @ (closing_w / norm))


@dataclass(frozen=True)
class ApproachStrategy:
    """How the mouth travels to the tracked point. Never a gate, never a threshold.

    ``straight`` is the September 23 baseline: every step points from the mouth
    at the live target. The two standoff modes first bring the mouth to a point
    ``standoff_m`` short of the target along a fixed axis, then finish along
    that axis. The axis is frozen at the first ``tracking`` measurement and is
    never recomputed from scene metadata: ``tool_axis_standoff`` uses the tool's
    forward (+z, the mouth direction) at that moment, ``horizontal_standoff``
    the horizontal component of mouth-to-target. ``max_step_m`` is the per-step
    cap shared by approach and retreat.

    ``planned_pose_standoff`` is a known-map plan: it rotates the tool, at most
    ``max_rotation_deg`` per frame, toward ``planned_tool_quat_wxyz`` (the home
    orientation when None) while the mouth moves to a point ``standoff_m`` back
    along the planned tool axis from the target tracked at the first tracking
    frame, then finishes along that axis toward the live tracked target. The
    retreat reverses through the standoff point, then returns home.
    """

    mode: str = "straight"
    standoff_m: float = 0.0
    max_step_m: float = 0.004
    max_rotation_deg: float = 1.5
    planned_tool_quat_wxyz: tuple | None = None

    def __post_init__(self):
        if self.mode not in APPROACH_MODES:
            raise ValueError(f"approach mode must be one of {APPROACH_MODES}")
        if not np.isfinite(self.standoff_m) or self.standoff_m < 0:
            raise ValueError("standoff_m must be finite and nonnegative")
        if self.mode == "straight" and self.standoff_m != 0.0:
            raise ValueError("the straight approach has no standoff")
        if self.mode != "straight" and self.standoff_m <= 0.0:
            raise ValueError("a standoff approach needs a positive standoff")
        if not np.isfinite(self.max_step_m) or not 0 < self.max_step_m <= 0.01:
            raise ValueError("max_step_m must be positive and at most 10 mm")
        if not np.isfinite(self.max_rotation_deg) or not 0 < self.max_rotation_deg <= 5.0:
            raise ValueError("max_rotation_deg must be positive and at most 5 degrees")
        if self.planned_tool_quat_wxyz is not None:
            if self.mode != "planned_pose_standoff":
                raise ValueError("only the planned approach takes a planned tool orientation")
            quat = np.asarray(self.planned_tool_quat_wxyz, dtype=float)
            if quat.shape != (4,) or not np.isfinite(quat).all() or abs(np.linalg.norm(quat) - 1.0) > 1e-3:
                raise ValueError("planned_tool_quat_wxyz must be a finite unit quaternion")
            object.__setattr__(self, "planned_tool_quat_wxyz", tuple(float(v) for v in quat))


class VisionPruningDemo:
    """One selected branch, no automatic reacquisition or ground-truth fallback.

    ``jaw_self_mask``: each frame's tracker update receives the robot's own jaw
    silhouette (``perception.jaw_self_mask``) predicted from the rendered tool
    pose, the jaw's attachment roll, the closure progress that was rendered into
    the frame (the cut step *before* this observation), the branch radius and
    the rendering camera. ``closure_hold``: while the cutter is closing, the tool
    is still within ``HOLD_MAX_*`` of its closure-start pose and the tracker's
    loss is one the mask explains, the cut observation carries the closure
    reference's target labelled ``vision_source='closure_hold'``; a second,
    unchanged cutter receives the honest observation every frame and is recorded
    as ``cut_without_hold``. Both default to off.
    """

    def __init__(
        self,
        target_id,
        branch_axis_w,
        branch_radius_m,
        home_pose_wxyz,
        *,
        max_step_m=0.004,
        closing_axis_tool=(1.0, 0.0, 0.0),
        photometric_normalization="raw",
        approach=None,
        motion_model="translation",
        jaw_self_mask=False,
        closure_hold=False,
        depth_appearance=False,
        depth_appearance_arm="strict",
    ):
        if not all(isinstance(flag, bool) for flag in (jaw_self_mask, closure_hold, depth_appearance)):
            raise ValueError("jaw_self_mask, closure_hold and depth_appearance must be bool")
        if depth_appearance_arm not in DEPTH_APPEARANCE_ARMS:
            raise ValueError(f"depth_appearance_arm must be one of {DEPTH_APPEARANCE_ARMS}")
        if depth_appearance_arm != "strict" and not depth_appearance:
            raise ValueError("depth_appearance_arm other than strict needs depth_appearance")
        if depth_appearance and (jaw_self_mask or closure_hold):
            # The registered depth variant runs on the unmasked tracker; the arms are never combined.
            raise ValueError("depth_appearance is registered without the jaw self-mask and the closure hold")
        self.target_id = str(target_id)
        self.axis = np.asarray(branch_axis_w, dtype=float)
        self.radius = float(branch_radius_m)
        self.home = np.asarray(home_pose_wxyz, dtype=float)
        self.approach = approach if approach is not None else ApproachStrategy(max_step_m=max_step_m)
        self.max_step = float(self.approach.max_step_m)
        self.approach_axis_w = None
        self.approach_phase = "straight" if self.approach.mode == "straight" else "standoff"
        self.planned_quat = None
        self.standoff_point_w = None
        self.retreat_phase = None
        self.mouth_offset = (0.0, 0.0, 0.070)
        self.closing_axis_tool = tuple(closing_axis_tool)
        tool_mouth_geometry(self.home, self.mouth_offset, self.closing_axis_tool)
        # Retain more initial same-depth texture corners on the thin Blender
        # spur. All subsequent inlier, appearance, depth and stop gates remain
        # unchanged. Maintain local corners only after valid measurements;
        # new corners are validated next frame and never reinitialize a loss.
        tracker_config = VisualServoConfig(
            depth_radius_px=1,
            feature_quality_level=0.005,
            replenish_features=True,
            photometric_normalization=photometric_normalization,
            motion_model=motion_model,
            # The registered value; read by the tracker only when a self-mask is supplied.
            min_unmasked_patch_elements=MIN_UNMASKED_PATCH_ELEMENTS,
        )
        self.tracker = (
            DepthAppearanceTracker(tracker_config, arm=depth_appearance_arm)
            if depth_appearance
            else VisualServoTracker(tracker_config)
        )
        self.cutter = SimulatedCutController(
            CutConfig(
                selected_target_id=self.target_id,
                max_branch_radius_m=0.012,
                mouth_position_tolerance_m=0.008,
                stable_frames=4,
                closure_duration_s=0.6,
            )
        )
        self.measurement = None
        self.measurement_time = None
        self.cut_step = None
        self.observations = 0
        self.vision_command_count = 0
        self.vision_command_request_count = 0
        self.hazard_contact = False
        self.external_stop_reason = None
        # Labelled arms; nothing below is touched when both are off.
        self.jaw_self_mask = jaw_self_mask
        self.closure_hold = closure_hold
        self.depth_appearance = depth_appearance
        self.depth_appearance_arm = depth_appearance_arm if depth_appearance else None
        self.jaw_roll_rad = proxy_roll_rad(self.closing_axis_tool) if (jaw_self_mask or depth_appearance) else None
        self.jaw_record = None
        self.depth_record = None
        self.cutter_without_hold = SimulatedCutController(self.cutter.config) if closure_hold else None
        self.cut_without_hold_step = None
        self.closure_pose = None
        self.latched_by_jaw = False
        self.hold_record = None

    def stop(self, reason):
        self.cutter.request_stop(reason)
        if self.cutter_without_hold is not None:
            # The shadow verdict must see every external stop the real cutter sees.
            self.cutter_without_hold.request_stop(reason)
        if self.external_stop_reason is None:
            self.external_stop_reason = reason

    def initialize(self, rgb, depth, seed_pixel, camera_matrix=None, world_from_optical=None, tool_pose_wxyz=None):
        if not self.jaw_self_mask:
            return self.tracker.initialize(rgb, seed_pixel, depth)
        if camera_matrix is None or world_from_optical is None or tool_pose_wxyz is None:
            raise ValueError("jaw_self_mask needs the camera calibration, camera pose and tool pose to initialize")
        # The preview is rendered with the jaws open (closure progress 0).
        mask, record = self._jaw_self_mask(tool_pose_wxyz, 0.0, camera_matrix, world_from_optical, np.shape(rgb)[:2])
        result = self.tracker.initialize(rgb, seed_pixel, depth, exclusion_mask=mask)
        result["jaw_self_mask"] = record
        return result

    def _jaw_self_mask(self, tool_pose_wxyz, closure_progress, camera_matrix, world_from_optical, shape):
        """The robot's own jaw pixels in this frame, and their telemetry record."""
        boxes = jaw_boxes(tool_pose_wxyz, self.jaw_roll_rad, closure_progress, self.radius)
        mask = jaw_mask(boxes, camera_matrix, world_from_optical, shape)
        record = {
            "closure_progress_used": float(closure_progress),
            "gap_m": float(jaw_gap_m(closure_progress, self.radius)),
            "roll_rad": self.jaw_roll_rad,
            **jaw_mask_summary(boxes, mask, camera_matrix, world_from_optical),
        }
        return mask, record

    def observe(self, rgb, depth, camera_matrix, world_from_optical, time_s, tool_pose_wxyz, hazard_contact=False):
        # The closure progress the runner rendered into this frame is the cut step from the previous
        # observation; the cutter advances it only further down, so it must be read first.
        rendered_progress = self.cut_step.closure_progress if self.cut_step else 0.0
        if self.jaw_self_mask:
            mask, self.jaw_record = self._jaw_self_mask(
                tool_pose_wxyz, rendered_progress, camera_matrix, world_from_optical, np.shape(rgb)[:2]
            )
            self.measurement = self.tracker.update(rgb, depth, camera_matrix, world_from_optical, exclusion_mask=mask)
        elif self.depth_appearance:
            # J's jaw silhouette: the same self-model as the mask (rendered pose and progress, roll, radius).
            boxes = frame_jaw_boxes(tool_pose_wxyz, self.jaw_roll_rad, rendered_progress, self.radius)
            self.depth_record = {
                "closure_progress_used": float(rendered_progress),
                "jaw_boxes_known": boxes is not None,
            }
            self.measurement = self.tracker.update(rgb, depth, camera_matrix, world_from_optical, jaw_boxes=boxes)
        else:
            self.measurement = self.tracker.update(rgb, depth, camera_matrix, world_from_optical)
        self.measurement_time = float(time_s)
        self.observations += 1
        self.hazard_contact = bool(hazard_contact)
        mouth, closing_axis = tool_mouth_geometry(tool_pose_wxyz, self.mouth_offset, self.closing_axis_tool)
        valid = self.measurement.get("state") == "tracking"
        target = self.measurement.get("target_position_world_m")
        observation = CutObservation(
            time_s=float(time_s),
            vision_timestamp_s=float(time_s),
            target_id=self.target_id,
            target_semantic="branch",
            vision_valid=valid,
            target_position_w=tuple(target) if target is not None else (float("nan"),) * 3,
            target_axis_w=tuple(self.axis),
            target_radius_m=self.radius,
            mouth_position_w=mouth,
            cutter_closing_axis_w=closing_axis,
            hazard_contact=self.hazard_contact,
        )
        if not self.closure_hold:
            self.cut_step = self.cutter.update(observation)
            return self.evidence()
        honest = observation
        observation, self.hold_record = self._closure_hold(honest, tool_pose_wxyz, float(time_s))
        was_closing = self.cutter.phase == "closing"
        self.cut_step = self.cutter.update(observation)
        if self.cut_step.phase == "closing" and not was_closing:
            self.closure_pose = np.asarray(tool_pose_wxyz, dtype=float).copy()
        self.cut_without_hold_step = self.cutter_without_hold.update(honest)
        return self.evidence()

    def _closure_hold(self, observation, tool_pose_wxyz, time_s):
        """The observation the cutter receives, and the per-frame hold record.

        The hold applies only while the cutter is closing, the tool is still
        within ``HOLD_MAX_*`` of its closure-start pose, and the measurement is a
        loss the jaw mask explains (``closure_hold_explanation``). A held
        observation carries the closure reference's target and timestamp,
        ``vision_valid=True`` and this frame's time; the cut certificate then
        lists the checks that labelling waives.
        """
        eligible = self.cutter.phase == "closing"
        stationary, dt_mm, dr_deg = False, None, None
        if self.closure_pose is not None:
            translation_m, rotation_deg = pose_change(self.closure_pose, tool_pose_wxyz)
            dt_mm, dr_deg = translation_m * 1e3, rotation_deg
            stationary = translation_m < HOLD_MAX_TRANSLATION_M and rotation_deg < HOLD_MAX_ROTATION_DEG
        explained, explanation = closure_hold_explanation(self.measurement, self.latched_by_jaw)
        held = bool(eligible and stationary and explained)
        held_age = None
        if held:
            reference = self.cutter.closure_reference
            observation = replace(
                observation,
                target_position_w=reference.target_position_w,
                vision_valid=True,
                vision_timestamp_s=time_s,
                vision_source="closure_hold",
                held_target_timestamp_s=reference.vision_timestamp_s,
            )
            held_age = time_s - reference.vision_timestamp_s
            self.latched_by_jaw = True
        record = {
            "eligible": eligible,
            "stationary": stationary,
            "dt_mm": dt_mm,
            "dr_deg": dr_deg,
            "explained": explained,
            "explanation": explanation,
            "held": held,
            "held_target_age_s": held_age,
            "latched_by_jaw": self.latched_by_jaw,
            "measurement_state": self.measurement.get("state"),
        }
        return observation, record

    def command(self, tool_pose_wxyz, now_s):
        pose = np.asarray(tool_pose_wxyz, dtype=float)
        command = pose.copy()
        if self.external_stop_reason:
            return command, "stopped_failure", {"state": "hold", "reason": self.external_stop_reason}
        if self.cut_step is None:
            return command, "observe", {"state": "hold", "reason": "no_camera_observation"}
        if self.cut_step.phase == "stopped" or self.hazard_contact:
            return command, "stopped_failure", {"state": "hold", "reason": self.cut_step.stopped_reason}
        if (
            self.cut_step.detached
            and self.approach.mode == "planned_pose_standoff"
            and self.standoff_point_w is not None
        ):
            return self._planned_retreat(pose)
        if self.cut_step.detached:
            delta = self.home[:3] - pose[:3]
            norm = float(np.linalg.norm(delta))
            command[:3] += delta * min(1.0, self.max_step / max(norm, 1e-12))
            return command, "retreat" if norm > 0.003 else "complete", {"state": "retreat", "remaining_m": norm}
        if self.cut_step.phase == "closing" or self.cut_step.certificate.ready_to_close:
            return command, "simulated_closure" if self.cut_step.phase == "closing" else "align", {"state": "hold"}
        if self.approach.mode == "planned_pose_standoff":
            return self._planned_command(pose, now_s)
        mouth, _ = tool_mouth_geometry(pose, self.mouth_offset, self.closing_axis_tool)
        # Zero residual standoff is to the explicitly added demo mouth, 70 mm
        # in front of the mock tool origin, not into the fixed CAD body.
        axis, standoff = (0.0, 0.0, 1.0), 0.0
        if self.approach.mode != "straight":
            if self.approach_axis_w is None:
                self.approach_axis_w = self._freeze_approach_axis(pose, mouth)
            if self.approach_axis_w is not None:
                axis = tuple(self.approach_axis_w)
                standoff = self.approach.standoff_m if self.approach_phase == "standoff" else 0.0
        decision = bounded_visual_servo_translation(
            self.measurement,
            mouth,
            approach_axis_world=axis,
            standoff_m=standoff,
            now_s=float(now_s),
            measurement_time_s=self.measurement_time,
            max_step_m=self.max_step,
            hazard_contact=self.hazard_contact,
        )
        decision["approach_phase"] = self.approach_phase
        if (
            self.approach_phase == "standoff"
            and decision["state"] == "tracking"
            and decision["remaining_distance_m"] is not None
            and decision["remaining_distance_m"] <= STANDOFF_REACHED_M
        ):
            self.approach_phase = "final"
        command[:3] += np.asarray(decision["delta_world_m"])
        if decision["state"] == "tracking":
            self.vision_command_request_count += 1
        return command, "vision_approach" if decision["state"] == "tracking" else "vision_hold", decision

    def _mouth(self, position, quat):
        return np.asarray(position, dtype=float) + quat_to_matrix(quat) @ np.asarray(self.mouth_offset, dtype=float)

    def _pose_for_mouth(self, mouth, quat):
        """Tool pose that puts the mouth at ``mouth`` with orientation ``quat``."""
        position = np.asarray(mouth, dtype=float) - quat_to_matrix(quat) @ np.asarray(self.mouth_offset, dtype=float)
        return np.concatenate((position, np.asarray(quat, dtype=float)))

    def _planned_command(self, pose, now_s):
        mouth = self._mouth(pose[:3], pose[3:])
        if self.planned_quat is None:
            planned = self.approach.planned_tool_quat_wxyz
            self.planned_quat = np.asarray(planned if planned is not None else self.home[3:], dtype=float)
            self.planned_quat /= np.linalg.norm(self.planned_quat)
        axis = quat_to_matrix(self.planned_quat)[:, 2]
        # Validity and freshness exactly as the baseline: a non-tracking or stale measurement holds.
        check = bounded_visual_servo_translation(
            self.measurement,
            mouth,
            approach_axis_world=tuple(axis),
            standoff_m=0.0,
            now_s=float(now_s),
            measurement_time_s=self.measurement_time,
            max_step_m=self.max_step,
            hazard_contact=self.hazard_contact,
        )
        if check["state"] != "tracking":
            check["approach_phase"] = self.approach_phase
            return pose.copy(), "vision_hold", check
        target = np.asarray(check["target_position_world_m"], dtype=float)
        if self.standoff_point_w is None:
            # Frozen once from the live tracked target, never from scene metadata.
            self.standoff_point_w = (target - self.approach.standoff_m * axis).tolist()
            self.approach_axis_w = axis.tolist()
        orientation_error = quat_angle_deg(pose[3:], self.planned_quat)
        goal = np.asarray(self.standoff_point_w) if self.approach_phase == "standoff" else target
        error = goal - mouth
        distance = float(np.linalg.norm(error))
        delta = error * min(1.0, self.max_step / distance) if distance > 0 else np.zeros(3)
        quat = rotate_toward(pose[3:], self.planned_quat, self.approach.max_rotation_deg)
        command = self._pose_for_mouth(mouth + delta, quat)
        decision = {
            **check,
            "delta_world_m": (command[:3] - pose[:3]).tolist(),
            "mouth_delta_world_m": delta.tolist(),
            "remaining_distance_m": distance,
            "orientation_error_deg": orientation_error,
            "approach_phase": self.approach_phase,
        }
        if (
            self.approach_phase == "standoff"
            and distance <= STANDOFF_REACHED_M
            and orientation_error <= ORIENTATION_REACHED_DEG
        ):
            self.approach_phase = "final"
        self.vision_command_request_count += 1
        return command, "vision_approach", decision

    def _planned_retreat(self, pose):
        """Back out along the planned axis to the standoff point, then home, rotating back as it goes."""
        mouth = self._mouth(pose[:3], pose[3:])
        if self.retreat_phase is None:
            self.retreat_phase = "to_standoff"
        if self.retreat_phase == "to_standoff":
            error = np.asarray(self.standoff_point_w) - mouth
            distance = float(np.linalg.norm(error))
            if distance <= STANDOFF_REACHED_M:
                self.retreat_phase = "to_home"
            else:
                delta = error * min(1.0, self.max_step / distance)
                command = self._pose_for_mouth(mouth + delta, self.planned_quat)
                return command, "retreat", {"state": "retreat", "retreat_phase": "to_standoff", "remaining_m": distance}
        delta = self.home[:3] - pose[:3]
        norm = float(np.linalg.norm(delta))
        quat = rotate_toward(pose[3:], self.home[3:], self.approach.max_rotation_deg)
        command = pose.copy()
        command[:3] += delta * min(1.0, self.max_step / max(norm, 1e-12))
        command[3:] = quat
        done = norm <= 0.003 and quat_angle_deg(pose[3:], self.home[3:]) <= 0.5
        return (
            command,
            "complete" if done else "retreat",
            {"state": "retreat", "retreat_phase": "to_home", "remaining_m": norm},
        )

    def _freeze_approach_axis(self, pose, mouth):
        """The fixed approach axis, from the tool pose or the first tracked target; None until tracking."""
        if not isinstance(self.measurement, dict) or self.measurement.get("state") != "tracking":
            return None
        if self.approach.mode == "tool_axis_standoff":
            forward = np.asarray(mouth, dtype=float) - pose[:3]
        else:
            target = np.asarray(self.measurement.get("target_position_world_m"), dtype=float)
            forward = target - np.asarray(mouth, dtype=float)
            forward[2] = 0.0
        norm = float(np.linalg.norm(forward))
        if not np.isfinite(norm) or norm <= 1e-9:
            return None
        return (forward / norm).tolist()

    def command_applied(self, phase, decision):
        """Acknowledge only after external gates and a successful physical step."""
        if phase == "vision_approach" and decision and decision.get("state") == "tracking":
            self.vision_command_count += 1

    def evidence(self):
        evidence = self._base_evidence()
        if self.jaw_self_mask:
            evidence["jaw_self_mask"] = self.jaw_record
        if self.depth_appearance:
            evidence["depth_appearance"] = self.depth_record
        if self.closure_hold:
            evidence["closure_hold"] = self.hold_record
            step = self.cut_without_hold_step
            evidence["cut_without_hold"] = (
                None if step is None else {"phase": step.phase, "stopped_reason": step.stopped_reason}
            )
        return evidence

    def _base_evidence(self):
        return {
            "tracker_config": asdict(self.tracker.config),
            "measurement": self.measurement,
            "measurement_time_s": self.measurement_time,
            "cut": None if self.cut_step is None else asdict(self.cut_step),
            "observations": self.observations,
            "vision_command_count": self.vision_command_count,
            "vision_command_request_count": self.vision_command_request_count,
            "external_stop_reason": self.external_stop_reason,
            "approach_strategy": asdict(self.approach),
            "approach_axis_w": self.approach_axis_w,
            "approach_phase": self.approach_phase,
            "planned_tool_quat_wxyz": None if self.planned_quat is None else self.planned_quat.tolist(),
            "standoff_point_w": self.standoff_point_w,
            "retreat_phase": self.retreat_phase,
            "mouth_offset_tool_m": list(self.mouth_offset),
            "closing_axis_tool": list(self.closing_axis_tool),
            "target_identity_axis_radius_source": "Selected Blender branch metadata; not learned recognition",
            "depth_source": "Live RTX optical-Z ground truth, not learned depth",
            "cut_model": "Visual jaw surrogate and discrete detachment; not material fracture",
        }
