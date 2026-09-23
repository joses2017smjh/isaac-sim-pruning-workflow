"""Drive the existing pruning controller from replayed sensor frames.

This adds no perception and no control law. It imports ``VisionPruningDemo``,
which already owns the tracker, the bounded Cartesian command and the closure and
geometry gates, and it feeds that object images in the recorded order. If a
number here disagrees with the recording, the disagreement is in the sequencing,
not in a second copy of the algorithm.

Causality is the whole point of the exercise. The recorded controller decided
frame N's command from the image captured at frame N-1, and every frame records
which image that was. The replay therefore observes frame N-1's image, then asks
for frame N's command, and refuses to compare frames where the recording says a
different image was used.
"""

from __future__ import annotations

import numpy as np

from .capture import NO_SOURCE_FRAME, CaptureError


class ControllerReplay:
    """Replays one capture through the real controller, one frame at a time."""

    def __init__(self, capture, *, max_step_m=None):
        self.capture = capture
        target = capture.target()
        if target["axis_w"] is None or target["radius_m"] is None:
            raise CaptureError("Capture target has no recorded axis or radius")

        # Imported, never reimplemented. This object is the tracker, the command
        # bound and the gates.
        from isaaclab_pruning.sim.vision_demo_controller import VisionPruningDemo

        kwargs = {"photometric_normalization": capture.photometric_normalization()}
        if max_step_m is not None:
            kwargs["max_step_m"] = float(max_step_m)
        report_closing = capture.report.get("blender_scene", {}).get("visual_proxy_closing_axis_tool")
        if report_closing is not None:
            kwargs["closing_axis_tool"] = tuple(float(value) for value in report_closing)

        self.controller = VisionPruningDemo(
            target["id"], target["axis_w"], target["radius_m"], capture.home_pose(), **kwargs
        )
        self.initialized = False
        self.observed_frames = []

    def initialize(self):
        """Seed the tracker from the one supplied pixel, as the recording did."""
        rgb = self.capture.rgb(0)
        depth = self.capture.depth(0)
        self.initialized = bool(self.controller.initialize(rgb, depth, self.capture.seed_pixel()))
        return self.initialized

    def observe(self, index):
        """Observe one recorded image. This is what a later command decides from."""
        frame = self.capture.frames[int(index)]
        self.controller.observe(
            self.capture.rgb(index),
            self.capture.depth(index),
            self.capture.camera_matrix,
            self.capture.world_from_optical(index),
            float(frame["time_s"]),
            np.asarray(frame["tool_pose_wxyz"], dtype=float),
            hazard_contact=bool(frame.get("contact_force_n", 0.0) > 0.0),
        )
        self.observed_frames.append(int(index))

    def command(self, index):
        """Ask the controller for the command it would propose at this frame."""
        frame = self.capture.frames[int(index)]
        pose, phase, decision = self.controller.command(
            np.asarray(frame["tool_pose_wxyz"], dtype=float), float(frame["time_s"])
        )
        return {"pose_wxyz": np.asarray(pose, dtype=float), "phase": phase, "decision": decision or {}}

    def step(self, index):
        """Observe the image the recording says this frame's decision came from.

        Returns the proposed command, or ``None`` when the recording says no image
        had been observed yet, which is the honest answer for frame 0.
        """
        source = self.capture.source_frame_index(index)
        if source == NO_SOURCE_FRAME:
            return None
        if source != index - 1:
            # The recording says a different image was used. Rather than quietly
            # comparing the wrong pair, say so.
            raise CaptureError(
                f"Frame {index} records controller_source_frame_index={source}, not {index - 1}; "
                "this replay only reproduces the recorded one-frame-behind schedule"
            )
        if not self.initialized:
            self.initialize()
        self.observe(source)
        proposal = self.command(index)
        self.controller.command_applied(proposal["phase"], proposal["decision"])
        return proposal


def compare_decision(proposed, recorded, *, position_tolerance_m):
    """Compare one proposed decision against the recorded one."""
    result = {
        "state_matches": proposed.get("state") == recorded.get("state"),
        "proposed_state": proposed.get("state"),
        "recorded_state": recorded.get("state"),
        "reason_matches": proposed.get("reason") == recorded.get("reason"),
    }
    left, right = proposed.get("delta_world_m"), recorded.get("delta_world_m")
    if left is None or right is None:
        result["delta_error_m"] = None
        result["delta_matches"] = left is None and right is None
    else:
        error = float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))
        result["delta_error_m"] = error
        result["delta_matches"] = error <= position_tolerance_m
    result["matches"] = bool(result["state_matches"] and result["delta_matches"])
    return result


def invalid_frame_is_held(decision):
    """An invalid observation must hold. It must never approach or release."""
    state = (decision or {}).get("state")
    return state not in ("tracking", "release", "closing")
