"""A beam grid across the jaw gap: the geometry of a presence sensor the mock pruner does not have yet.

Nine beams run between the inner faces of the two jaws, parallel to the closing axis: three positions along
the jaw at three heights along the tool axis, centred on the jaw mouth. A beam is *blocked* when anything
crosses it. The safety gate can require a blocked beam before the jaw may close
(``task.safety_gate.SafetyGateReference(jaw_presence_min_beams=...)``), which is evidence about the mouth that
does not come from the controller's tracker.

This module is the geometry only, with no simulator import. ``tools/cast_jaw_beams.py`` evaluates it offline
at recorded tool poses against the exported orchard meshes; nothing in Isaac Sim publishes it yet. The grid's
spacing was chosen in the exploratory study of November 25, 2026 (``tools/study_mouth_evidence.py``) and is a
demo choice, not a part's datasheet.
"""

from __future__ import annotations

import numpy as np

from isaaclab_pruning.perception.jaw_self_mask import JAW_HEIGHT_M, _quat_matrix, _rot_z, jaw_gap_m

#: Beam positions along the jaw (the jaw frame's y), metres from the tool axis.
BEAMS_ALONG_JAW_M = (-0.008, 0.0, 0.008)
#: Beam heights along the tool axis, metres from the jaw mouth.
BEAMS_ALONG_TOOL_M = (-0.010, 0.0, 0.010)
BEAM_COUNT = len(BEAMS_ALONG_JAW_M) * len(BEAMS_ALONG_TOOL_M)
#: Index of the beam through the mouth centre, in the order ``beam_origins`` returns them.
CENTRE_BEAM = BEAM_COUNT // 2
FRAME = "mock_pruner__tool0"


def jaw_rotation(tool_pose_wxyz, roll_rad):
    """World rotation of the jaw frame: x the closing axis, y along the jaw, z the tool axis."""
    pose = np.asarray(tool_pose_wxyz, dtype=float)
    if pose.shape != (7,) or not np.isfinite(pose).all():
        raise ValueError("tool pose must be seven finite values: x y z w qx qy qz")
    return _quat_matrix(pose[3:]) @ _rot_z(float(roll_rad))


def beam_origins(tool_position_w, jaw_rotation_w, gap_m, mouth_offset_m=JAW_HEIGHT_M):
    """Where each beam leaves the first jaw's inner face, heights major and jaw positions minor."""
    if not np.isfinite(gap_m) or gap_m <= 0.0:
        raise ValueError("the jaw gap must be finite and positive")
    position = np.asarray(tool_position_w, dtype=float)
    rotation = np.asarray(jaw_rotation_w, dtype=float)
    return np.array(
        [
            position + rotation @ np.array([-gap_m / 2.0, y, mouth_offset_m + z])
            for z in BEAMS_ALONG_TOOL_M
            for y in BEAMS_ALONG_JAW_M
        ]
    )


def beam_segments(tool_pose_wxyz, roll_rad, closure_progress, radius_m):
    """``(origins, direction, length)`` of the nine beams at one tool pose and jaw closure."""
    rotation = jaw_rotation(tool_pose_wxyz, roll_rad)
    gap = jaw_gap_m(float(closure_progress), float(radius_m))
    origins = beam_origins(np.asarray(tool_pose_wxyz, dtype=float)[:3], rotation, gap)
    return origins, rotation[:, 0].copy(), gap


def blocked_beams(first_hit_m, gap_m):
    """One flag per beam from the first-hit distance along it; a miss is ``inf`` or ``nan``."""
    hits = np.asarray(first_hit_m, dtype=float)
    if hits.shape != (BEAM_COUNT,):
        raise ValueError(f"expected {BEAM_COUNT} first-hit distances")
    return np.isfinite(hits) & (hits >= 0.0) & (hits < gap_m)
