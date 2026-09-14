"""Causal RGB-D branch tracking and a labelled simulated-detachment demo.

Only the initial pixel and branch identity/axis/radius are supplied by the scene.
Subsequent approach positions come from live image tracking and rendered depth.
This is classical visual servoing, not learned branch recognition or fracture.
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

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


class VisionPruningDemo:
    """One selected branch, no automatic reacquisition or ground-truth fallback."""

    def __init__(
        self,
        target_id,
        branch_axis_w,
        branch_radius_m,
        home_pose_wxyz,
        *,
        max_step_m=0.004,
        closing_axis_tool=(1.0, 0.0, 0.0),
    ):
        self.target_id = str(target_id)
        self.axis = np.asarray(branch_axis_w, dtype=float)
        self.radius = float(branch_radius_m)
        self.home = np.asarray(home_pose_wxyz, dtype=float)
        self.max_step = float(max_step_m)
        self.mouth_offset = (0.0, 0.0, 0.070)
        self.closing_axis_tool = tuple(closing_axis_tool)
        tool_mouth_geometry(self.home, self.mouth_offset, self.closing_axis_tool)
        # Retain more initial same-depth texture corners on the thin Blender
        # spur. All subsequent inlier, appearance, depth and stop gates remain
        # unchanged. Maintain local corners only after valid measurements;
        # new corners are validated next frame and never reinitialize a loss.
        self.tracker = VisualServoTracker(
            VisualServoConfig(depth_radius_px=1, feature_quality_level=0.005, replenish_features=True)
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

    def stop(self, reason):
        self.cutter.request_stop(reason)
        if self.external_stop_reason is None:
            self.external_stop_reason = reason

    def initialize(self, rgb, depth, seed_pixel):
        return self.tracker.initialize(rgb, seed_pixel, depth)

    def observe(self, rgb, depth, camera_matrix, world_from_optical, time_s, tool_pose_wxyz, hazard_contact=False):
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
        self.cut_step = self.cutter.update(observation)
        return self.evidence()

    def command(self, tool_pose_wxyz, now_s):
        pose = np.asarray(tool_pose_wxyz, dtype=float)
        command = pose.copy()
        if self.external_stop_reason:
            return command, "stopped_failure", {"state": "hold", "reason": self.external_stop_reason}
        if self.cut_step is None:
            return command, "observe", {"state": "hold", "reason": "no_camera_observation"}
        if self.cut_step.phase == "stopped" or self.hazard_contact:
            return command, "stopped_failure", {"state": "hold", "reason": self.cut_step.stopped_reason}
        if self.cut_step.detached:
            delta = self.home[:3] - pose[:3]
            norm = float(np.linalg.norm(delta))
            command[:3] += delta * min(1.0, self.max_step / max(norm, 1e-12))
            return command, "retreat" if norm > 0.003 else "complete", {"state": "retreat", "remaining_m": norm}
        if self.cut_step.phase == "closing" or self.cut_step.certificate.ready_to_close:
            return command, "simulated_closure" if self.cut_step.phase == "closing" else "align", {"state": "hold"}
        mouth, _ = tool_mouth_geometry(pose, self.mouth_offset, self.closing_axis_tool)
        # Zero residual standoff is to the explicitly added demo mouth, 70 mm
        # in front of the mock tool origin, not into the fixed CAD body.
        decision = bounded_visual_servo_translation(
            self.measurement,
            mouth,
            approach_axis_world=(0.0, 0.0, 1.0),
            standoff_m=0.0,
            now_s=float(now_s),
            measurement_time_s=self.measurement_time,
            max_step_m=self.max_step,
            hazard_contact=self.hazard_contact,
        )
        command[:3] += np.asarray(decision["delta_world_m"])
        if decision["state"] == "tracking":
            self.vision_command_request_count += 1
        return command, "vision_approach" if decision["state"] == "tracking" else "vision_hold", decision

    def command_applied(self, phase, decision):
        """Acknowledge only after external gates and a successful physical step."""
        if phase == "vision_approach" and decision and decision.get("state") == "tracking":
            self.vision_command_count += 1

    def evidence(self):
        return {
            "measurement": self.measurement,
            "measurement_time_s": self.measurement_time,
            "cut": None if self.cut_step is None else asdict(self.cut_step),
            "observations": self.observations,
            "vision_command_count": self.vision_command_count,
            "vision_command_request_count": self.vision_command_request_count,
            "external_stop_reason": self.external_stop_reason,
            "mouth_offset_tool_m": list(self.mouth_offset),
            "closing_axis_tool": list(self.closing_axis_tool),
            "target_identity_axis_radius_source": "Selected Blender branch metadata; not learned recognition",
            "depth_source": "Live RTX optical-Z ground truth, not learned depth",
            "cut_model": "Visual jaw surrogate and discrete detachment; not material fracture",
        }
