#!/usr/bin/env python3
"""Score the jaw-in-view experiment (protocol of October 1) against P1-P8.

Outcomes come from the per-batch evidence that ``aggregate_eval.py`` wrote (the grader's classes, never
re-graded here); failed, incomplete and refused runs stay in every table. The evidence is checked against the
registered six runs before anything is scored. Each run's own records supply what a clause names: the recorded
mask count and depth (P1), the tracker's states and patch telemetry (P2, P3), the cut phases, certificates,
closure-hold records and the shadow cutter (P4, P5), and the piece and tool poses (P8). P7 replays each 530 run
that held with both flags off (``tools/replay_jaw_self_mask.py``): the baseline tracker on that run's own live
frames. The scored modules are imported from the batches' frozen snapshot (``<batch>/code``, the code the runs
executed, checked against each plan's sha256), never from the checkout. Each run is labelled with its GPU model
from the node in its report.

Frame numbers are 0-based ``frames.json`` indices. Closure start ``k`` is the first frame whose cut phase is
``closing``, the deadline is ``k + 6``, the detach frame is the frame whose cut carries ``detach_event``, and
a stop is decided on the first frame whose cut phase is ``stopped`` (the evidence's ``stop_frame`` is one
later). Where the protocol leaves a detail open it is fixed here and recorded in ``interpretations``. What was
seen of the live runs before this scorer was finished is disclosed in ``construction_exposure``. Every result
carries the label: known-map plan; jaw self-mask; closure hold with freshness and frame-reuse checks waived on
held frames.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import io
import json
import math
import subprocess
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(ROOT / "source/isaaclab_pruning"))
from score_perception_round import clause, prediction  # noqa: E402
from score_planned_approach import gpu_model  # noqa: E402

# The checkout's copies, so the module imports and its tests run anywhere; main() rebinds every name in
# FROZEN_NAMES to the batches' frozen snapshot before anything is scored.
from isaaclab_pruning.perception.jaw_self_mask import (  # noqa: E402
    MASK_MARGIN_PX,
    PIXEL_CENTRE,
    closure_hold_explanation,
    jaw_boxes,
    jaw_mask,
    jaw_mask_summary,
    jaw_projection,
    jaw_signed_distance_px,
    polygon_signed_distance,
    pose_change,
    registered_closure_hold,
    registered_jaw_self_mask,
    rotation_angle_deg,
)
from isaaclab_pruning.task.simulated_cut import CLOSURE_HOLD_WAIVED_CHECKS, tool_mouth_geometry  # noqa: E402

PROTOCOL = "docs/EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md"
EVIDENCE_DIR = "docs/evidence/jaw_in_view_2026-10-01"
BATCHES = ("jaw-hold-a-20261001", "jaw-hold-b-20261001", "jaw-hold-c-20261001")
STRATEGY = "planned_pose_jaw_hold"
LABEL = "known-map plan; jaw self-mask; closure hold with freshness and frame-reuse checks waived on held frames"
#: The registered runs, as (target tree, target) in run-index order per batch (protocol, Runs).
REGISTERED_RUNS = {
    "jaw-hold-a-20261001": ((0, 530), (1, 19444)),
    "jaw-hold-b-20261001": ((0, 530), (1, 14944)),
    "jaw-hold-c-20261001": ((0, 530), (1, 15004)),
}
#: Every planned run of the three batches: 530 three times, 19444 once, controls 14944 and 15004 once each.
EXPECTED_RUNS = {530: 3, 19444: 1, 14944: 1, 15004: 1}
CONTROLS = (14944, 15004)
#: "All runs use source light and 200 frames" (10 fps).
PLANNED_FRAMES = 200
PLANNED_FPS = 10
DAYLIGHT = "source"
#: The plan froze these sources. Every batch's snapshot must hold them unchanged; the scorer imports
#: jaw_self_mask, simulated_cut and (through the replay tool) visual_servo and the controller from the snapshot,
#: and never runs the runner, the scene or the grader.
SCORED_SOURCES = (
    "source/isaaclab_pruning/isaaclab_pruning/perception/jaw_self_mask.py",
    "source/isaaclab_pruning/isaaclab_pruning/perception/visual_servo.py",
    "source/isaaclab_pruning/isaaclab_pruning/sim/vision_demo_controller.py",
    "source/isaaclab_pruning/isaaclab_pruning/task/simulated_cut.py",
    "source/isaaclab_pruning/isaaclab_pruning/sim/blender_demo_scene.py",
    "hpc/inner/render_pruning_workflow.py",
    "tools/replay_jaw_self_mask.py",
    "tools/validate_vision_sequence.py",
)
#: The names this module takes from the scored code, by module; main() rebinds every one to the snapshot.
FROZEN_NAMES = {
    "isaaclab_pruning.perception.jaw_self_mask": (
        "MASK_MARGIN_PX",
        "PIXEL_CENTRE",
        "closure_hold_explanation",
        "jaw_boxes",
        "jaw_mask",
        "jaw_mask_summary",
        "jaw_projection",
        "jaw_signed_distance_px",
        "polygon_signed_distance",
        "pose_change",
        "registered_closure_hold",
        "registered_jaw_self_mask",
        "rotation_angle_deg",
    ),
    "isaaclab_pruning.task.simulated_cut": ("CLOSURE_HOLD_WAIVED_CHECKS", "tool_mouth_geometry"),
}
#: Imported from the snapshot; the replay tool brings the controller and the tracker with it.
FROZEN_MODULES = (*FROZEN_NAMES, "replay_jaw_self_mask")

#: What was seen of the live runs before this scorer was finished. Disclosed, never read by a clause.
CONSTRUCTION_EXPOSURE = [
    "The predictions scored here are P1-P8 of docs/EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md, registered before the "
    "six runs were submitted; this scorer implements them, and every detail it fixes is listed in 'interpretations'.",
    "While the draft of this scorer was being written, a mapping agent read jaw-hold-a-20261001 run_00 (target 530) "
    "mid-recording, at 81 frames, and printed frame 75's records: closure_hold eligible true, stationary true, "
    "explained false, held false, measurement_state tracking; cut_without_hold phase closing; jaw_self_mask "
    "closure_progress_used 0.0, mask_pixel_count 5793.",
    "The runner's progress log of that run showed 'phase=retreat' at frames 101-121.",
    "The main session saw Slurm states and elapsed times: first A run 0 COMPLETED, A run 1 FAILED and B run 0 "
    "COMPLETED; then, on October 2 and before this scorer was committed, the final states of all six tasks: A run 0 "
    "COMPLETED 36m28s; A run 1 FAILED exit 1 24m45s; B run 0 COMPLETED 33m18s; B run 1 COMPLETED 27m13s; C run 0 "
    "COMPLETED 33m23s; C run 1 COMPLETED 26m19s; all on cn-gpu5, an RTX 8000.",
    "Those Slurm states disclosed outcomes. Each task ran tools/run_vision_experiment.py, whose execute() exits 0 only "
    "when the capture completed, the unchanged grader passed all 17 checks (grade_sequence ok) and "
    "configuration_matches is true. So COMPLETED meant that A run 0, B runs 0 and 1 and C runs 0 and 1 each passed "
    "17/17 with configuration_matches true, and A run 1's exit 1 meant that 19444's capture, grade or configuration "
    "check failed. An earlier draft of this disclosure said 'a Slurm state is not a grade', which understated this. "
    "Outcomes here are still read only from the per-batch evidence written by tools/aggregate_eval.py (the grader's "
    "classes).",
    "The per-batch grade files are produced by a CPU job before this scorer is committed and are not opened until it "
    "is committed.",
    "Apart from these items, no jaw-hold run output was read by anyone while this scorer was written, reviewed and "
    "amended; of the jaw-hold batches, its final amendment read only the three plan.json files and the frozen code "
    "snapshot.",
    "What these items bear on. The Slurm states revealed, before this scorer was committed, the outcome P6 counts (all "
    "three 530 runs passed 17/17), P2's pass clause (both controls passed 17/17), that five runs have complete "
    "200-frame captures (P1 coverage) and that none of those five was refused (configuration_matches true). For 19444 "
    "(P3) they revealed only that the run did not pass: had it passed, P3's predicted stop could not have occurred; "
    "its exit 1 left P3's verdict open. The run 0 records and progress log bear on P4, P5, P6, P7 and P8 for that "
    "run: the retreat line means it detached, which puts it in P8's population and, if it held, in P7's.",
    "No clause reads or uses any of these items. The interpretations they could bear on (P4's premise rule, P5's "
    "population, P7's window, gap rule and anchor, P8's window) were fixed after them and are justified in "
    "'interpretations' from the protocol text alone. The final amendment, made from a blind checker's report before "
    "any grade file was opened, decides P1 on the registered literal reading (the 0.01 px edge band is reported as a "
    "sensitivity only) and lets a P7 run refute only when the baseline measured the target on every frame of the "
    "window.",
]
#: Per run, what of it was seen before scoring (construction_exposure); reported in the runs table only.
PASSED_EXIT = "exit 0: capture complete, grader ok (17/17) and configuration_matches true"
SEEN_BEFORE_SCORING = {
    ("jaw-hold-a-20261001", 0): "records read mid-recording at 81 frames (frame 75 printed); progress log showed "
    f"phase=retreat at frames 101-121; Slurm COMPLETED 36m28s ({PASSED_EXIT})",
    ("jaw-hold-a-20261001", 1): "Slurm FAILED 24m45s (exit 1: the capture, the grader's ok or configuration_matches "
    "was false)",
    ("jaw-hold-b-20261001", 0): f"Slurm COMPLETED 33m18s ({PASSED_EXIT})",
    ("jaw-hold-b-20261001", 1): f"Slurm COMPLETED 27m13s ({PASSED_EXIT})",
    ("jaw-hold-c-20261001", 0): f"Slurm COMPLETED 33m23s ({PASSED_EXIT})",
    ("jaw-hold-c-20261001", 1): f"Slurm COMPLETED 26m19s ({PASSED_EXIT})",
}
#: The one run whose own records were seen; its per-run views in P4-P8 point to construction_exposure.
RECORDS_SEEN = {("jaw-hold-a-20261001", 0)}

SCOPE = (
    "Scoring of the jaw-in-view experiment (P1-P8): six simulator runs of planned_pose_jaw_hold in three batches "
    "(530 three times, 19444 once, controls 14944 and 15004 once each). Every result is labelled '" + LABEL + "' "
    "and is reported apart from every unchanged-gate result, never pooled. Outcomes are the grader's, read from the "
    "per-batch evidence and never re-graded here. Simulator renders of a two-box jaw surrogate with exact poses and "
    "commanded closure; not a field result. What was seen of the live runs before this scorer was finished is "
    "listed in construction_exposure."
)
#: The unchanged grader's checks (tools/validate_vision_sequence.py:grade_sequence); never run here.
GRADER_CHECKS = (
    "capture_complete_and_nonblank",
    "ordered_contiguous_frame_indexes",
    "strictly_increasing_capture_times",
    "causal_previous_frame_commands",
    "one_matching_detach_event_and_request",
    "detach_has_completed_certified_closure",
    "detached_state_consistent_with_event",
    "selected_piece_pose_evidence_complete",
    "piece_stationary_before_release",
    "piece_dropped_after_release",
    "robot_position_evidence_complete",
    "post_detach_retreat_toward_home",
    "returned_home_and_final_phase_complete",
    "vision_commands_applied_and_counted",
    "captured_contact_within_limit",
    "no_recorded_stops",
    "capture_did_not_advance_physics",
)
GRADER_NOTE = (
    "The grader never reads vision_source: a certificate on a held frame has no reasons (the waived checks are "
    "listed, not appended), so a detach certified on a held frame passes detach_has_completed_certified_closure."
)

# Registered values (the protocol's); nothing here moves them.
DEPTH_TOLERANCE_M = 0.001  # P1: recorded depth more than 1 mm beyond the box
PATCH_SIDE = 13
PATCH_ELEMENTS = PATCH_SIDE * PATCH_SIDE  # P2, P3: the 13 x 13 appearance patch
EXPECTED_STOP_FRAMES_19444 = (67, 68)  # P3: an expectation, not a refutation condition
MAX_CLOSURE_START_MOUTH_M = 0.004  # P4: at most 4 mm, inclusive
FIRST_LOSS_OFFSET = 2  # P4: the first closure loss at closure start + 2
NO_TRACKING_OFFSET = 3  # P4: tracking must not continue at closure start + 3
CLOSURE_FRAMES = 6  # 0.6 s at 10 fps: the deadline, and detachment, at closure start + 6
TOOL_TRANSLATION_M, TOOL_ROTATION_DEG = 0.0005, 0.25  # P5: within 0.5 mm and 0.25 deg
HELD_MOUTH_TOLERANCE_M = 0.00005  # P5: within 0.05 mm
DRIFT_M = 0.003  # P7: at least 3 mm
PIECE_MOTION_M = DISTANCE_CHANGE_M = 0.0001  # P8: less than 0.1 mm
# Definitions fixed here (interpretations), not registered values.
EDGE_BAND_PX = 0.01  # P1 sensitivity only, never a verdict: hits within 0.01 px of the undilated silhouette edge
ROLL_TOLERANCE_RAD = 1e-12  # P1 cross-check: the controller's roll and the renderer's may differ by an ulp
#: The two explanations of a first closure loss that P4 calls mask-attributable (the latched branch cannot be first).
ATTRIBUTABLE = ("too_few_unmasked_patch_pixels", "flow_failed_only_after_mask_drop")
#: Stops the hold itself can cause; a held frame that triggers one stops.
HOLD_FAILURE_STOPS = ("closure_hold_outside_closing", "closure_hold_target_mismatch", "invalid_vision_source")
#: Tracker states that latch until explicit initialization; P7's baseline window ends at the first one.
LATCHED_STATES = ("tracking_lost", "jaw_mask_occluded")
#: States that do not latch (the tracker can track again on the next frame); P7 lists them as gaps.
GAP_STATES = ("invalid_depth", "invalid_calibration")
MEASUREMENT_KEYS = (
    "state",
    "reason",
    "pixel_xy",
    "target_position_world_m",
    "jaw_mask_patch_unmasked",
    "jaw_mask_patch_masked_prev",
    "jaw_mask_patch_masked_cur",
    "jaw_mask_dropped_valid",
    "jaw_mask_in_mask_any",
    "jaw_mask_unmasked_flow_would_pass",
    "jaw_mask_pixel_in_mask",
)

TEXT = {
    "P1": "(mask fidelity) On every frame of all six runs, the recorded mask pixel count equals an offline "
    "reconstruction from the recorded pose, progress and camera. No predicted jaw pixel shows recorded depth more "
    "than 1 mm beyond the analytic box depth. Refuted by any mismatch.",
    "P2": "(controls, no harm) 14944 and 15004 pass 17/17. The mask removes no patch element before detach, the "
    "tracker reports tracking on every frame before detach, and the hold never applies. Refuted if either fails a "
    "check, loses tracking before detach, masks a patch element before detach, or holds.",
    "P3": "(19444, explicit stop before the jaw is measured) 19444 stops on the final leg with a mask-attributable "
    "state (jaw_mask_occluded expected at frames 67-68). No frame reports tracking with the tracked pixel inside "
    "the jaw silhouette. Refuted if such a frame occurs, or if, after the jaw first touches the patch, it stops on "
    "a loss the mask does not explain or continues to align. A stop before the jaw reaches the patch leaves P3 "
    "untested.",
    "P4": "(530 mechanism) Predicted only for runs whose closure starts with a gate mouth distance of at most 4 mm "
    "(recorded 2.69-3.05 mm). The first closure loss comes at closure start + 2, the first frame rendered with "
    "the jaw closing, and is mask-attributable. The shadow controller (cut_without_hold) stops there with "
    "vision_invalid. Refuted if, in such a run, tracking continues at closure start + 3, or the first loss comes "
    "at another frame or is not mask-attributable. Runs that start closure above 4 mm are reported, not "
    "predicted.",
    "P5": "(530 hold to detachment) In every 530 run where P4's loss occurs, the hold covers every frame from it "
    "to the closure deadline. The tool stays within 0.5 mm and 0.25 deg. The gate's mouth distance on held frames "
    "equals its closure-start value within 0.05 mm, and detachment comes at closure start + 6. Refuted if any "
    "held frame stops or detachment comes at another frame.",
    "P6": "(530 outcome under the label) At least one of the three 530 runs passes all 17 checks of the unchanged "
    "grader. Refuted if none does. Every failed check is reported per run.",
    "P7": "(the hold covered a real drift) In each 530 run that holds, the baseline tracker replayed on that run's "
    "own live frames moves its measured target at least 3 mm from the closure-start target before the deadline. "
    "Refuted if it stays within 3 mm; the hold was then unnecessary in that run.",
    "P8": "(the static-branch premise, evaluation only) In every 530 run that detaches, the selected piece moves "
    "less than 0.1 mm, and the true mouth-to-spur distance changes by less than 0.1 mm, between closure start and "
    "detach. Refuted otherwise.",
}

INTERPRETATIONS = [
    "Outcomes are the runs[] rows of the three per-batch evidence files written by tools/aggregate_eval.py (the "
    "grader's classes); nothing is re-graded here. A pass is outcome 'pass' (graded, checks_passed == checks_total "
    "== 17). An incomplete, rejected or refused run is not a pass, and every planned run stays in every table.",
    "Run set: the scorer refuses to score unless the evidence holds exactly the registered six runs: one batch per "
    "file, its batch_dir named as the file, planned_runs_kept equal to the plan's runs and to the evidence rows, every "
    "evidence row matched to its plan row by run_directory with the same tree, target, light, photometric mode and "
    "strategy, batch A = 530 and 19444, B = 530 and 14944, C = 530 and 15004, and plans registered for this protocol "
    "with 200 frames at 10 fps, source light and both arms on. aggregate_eval never drops a planned run (a missing "
    "capture is 'incomplete'), so valid evidence always passes these checks.",
    "Scored code: jaw_self_mask and simulated_cut, and through tools/replay_jaw_self_mask the controller and the "
    "tracker, are imported from jaw-hold-a's snapshot <batch>/code, the code the runs executed (run_vision_experiment "
    "points PRUNING_ROOT at it and verify_snapshot checked it before each run). Every batch's snapshot must equal its "
    "plan for the eight scored sources, the three plans must freeze identical sources, and every module the import "
    "loads from the snapshot must hash to the plan; otherwise nothing is scored. The checkout is not imported: since "
    "the plan revision 3c6a211 one commit (2938ede, the default-off depth_appearance arm) changed the controller, the "
    "runner and the replay tool there, and the checkout's controller imports perception/depth_appearance.py, which no "
    "plan froze; scored_code.checkout reports both. The runner, the scene and the grader are never run here; the "
    "scorer's own helpers (score_perception_round, score_planned_approach) come from the checkout and equal the "
    "snapshot's copies.",
    "Refused captures: configuration_matches (experiment_result.json) true makes a run of the arm, and the scorer's "
    "own report checks, reported beside it, cannot override it; false on a complete capture (report frame_count == "
    "200) refuses it. When the flag is absent, or false on an incomplete capture (the frozen check requires the "
    "plan's 200 frames, so a capture that died mid-recording fails it for that alone), the report decides: it is "
    "refused when it shows an arm not enabled, constants, thresholds or waived checks other than the registered ones, "
    "a tracker minimum other than 140, or another target, light, photometric mode or approach than the plan's. A "
    "refused run is no pass (P2, P6), is excluded from every P1-P5, P7 and P8 judgment, and is listed in a clause "
    "that holds false and cannot refute. An incomplete run that ran the arm keeps its recorded frames as evidence.",
    "Coverage: a run is outside a prediction's condition (reported; such a clause holds and cannot refute) only when "
    "its recording settles it: all 200 frames recorded, or the cut already stopped or detached (nothing closes, holds "
    "or detaches after that). A refused or unrecorded run, and one whose recording ends with the cut still live, is "
    "listed in a clause that holds false and cannot refute, so a prediction over every such run cannot be supported "
    "while one of them might fall inside it.",
    "Verdicts: refuted when any clause meets its registered refutation; otherwise 'untested' when the condition a "
    "prediction needs never arose in a judgeable run (named in untested_reason; such a clause holds false, refutes "
    "false and carries observed.vacuous); otherwise supported when every clause holds and partly supported when one "
    "does not.",
    "Frame numbers are 0-based frames.json indices. Closure start k is the first frame whose cut phase is "
    "'closing'; the closure deadline is k + 6 (0.6 s at 10 fps); the detach frame d is the frame whose cut carries "
    "detach_event; the cut-stop decision frame s is the first frame whose cut phase is 'stopped' (the evidence's "
    "stop_frame, the runner's first 'stopped_failure' frame, is s + 1). P3-P5 frame numbers are cut-decision "
    "frames.",
    "P1 reconstructs each frame's mask with the frozen jaw_self_mask module (jaw_boxes, jaw_mask: 2.0 px margin, "
    "pixel centre 0.5) from the renderer-side inputs: report.blender_scene.visual_proxy_roll_rad and "
    "target.radius_m, the frame's visual_jaw_closure_progress, tool_pose_wxyz and wrist camera pose (world from "
    "optical), and report.camera's intrinsics and resolution. A reconstruction that fails counts as a mismatch. The "
    "controller's mask record must name the rendered jaw: its roll_rad must equal the renderer's within 1e-12 rad (the "
    "controller's roll is a twice-normalized copy of the scene's closing axis, so the two can differ by an ulp; the "
    "largest difference is reported); otherwise the mask was built for another jaw and P1 cannot be supported, though "
    "this cannot refute. Its closure_progress_used is a recording echo, equal by construction (both fields are "
    "demo.cut_step.closure_progress read before observe), and bbox and corners are reported; the rendered jaw itself "
    "is judged only by the depth clause.",
    "P1's predicted jaw pixels are the pixels whose centre ray (u + 0.5, v + 0.5) hits either box, found by a slab "
    "test in the optical frame, where the entry parameter is optical Z; a violation is recorded depth that is "
    "non-finite or strictly more than 1 mm beyond the box depth, and any violation refutes (the registered reading: "
    "'Refuted by any mismatch'). A sensitivity is reported beside it and decides nothing: the count that leaves out "
    "hits whose centre lies within 0.01 px of the undilated silhouette edge (signed distance to either box's float32 "
    "hull > -0.01 px at index coordinates), with each such pixel beyond 1 mm listed with its signed distance and "
    "depths. It is reported because the renderer poses the jaw in float32 against the float64 analytic box: on the "
    "twelve earlier planned-pose runs (not the scored runs) every violation was one pixel whose centre lay within "
    "3.4e-5 px of the silhouette edge (seven frames, all 19444, each seeing past the jaw). Deciding P1 on the "
    "sensitivity would narrow a registered refutation condition; that needs the user's sign-off and a separately "
    "labelled amendment, and none was given before scoring. The 1 mm tolerance is unchanged. Pixels with something "
    "nearer than the jaw, and hits outside the dilated mask, are reported, not judged. Every recorded frame counts, "
    "including frames after a stop, after detach and during retreat.",
    "P1 coverage: a run with fewer than 200 recorded frames, a frame without a mask record or a readable depth file, "
    "and a refused or unrecorded planned run are listed in a coverage clause that cannot refute but keeps P1 from "
    "being supported. The preview and initialization masks are not recorded frames and their camera pose is not "
    "recorded (it is rebuilt as the replay does); they are reported for information only.",
    "P2: 'before detach' is frames 0..d inclusive (every image captured before release); 0..d-1 is reported as a "
    "sensitivity; a run without a detach is judged on all its frames, and that window is complete only when all 200 "
    "frames are recorded. A masked patch element is jaw_mask_patch_unmasked < 169 (elements unmasked in both the "
    "previous and the current patch); feature drops by the mask (jaw_mask_dropped_valid, jaw_mask_in_mask_any) are "
    "reported, not judged. A hold is any frame with closure_hold.held true or a certificate vision_source of "
    "'closure_hold'. The pass clause is refuted only by an unrefused control the grader graded with a failed check "
    "(the registered refutation names a failed check); a control that was never graded failed no check, and a refused "
    "control is not a run of the arm, so either keeps the clause from holding without refuting it, and a refused "
    "control's frames are not judged.",
    "P3: pixel_xy is taken in pixel-index coordinates (as the tracker's rint and getRectSubPix use it); pixel_xy - "
    "0.5 is a reported sensitivity. Inside the silhouette means a signed distance <= 0 to either undilated jaw "
    "hull (jaw_signed_distance_px), judged on every frame whose state is tracking; on an incomplete recording that "
    "clause cannot hold, since it is about every frame.",
    "P3: the jaw first touches the patch on the first frame whose previous or current patch has a masked element "
    "(jaw_mask_patch_masked_prev + jaw_mask_patch_masked_cur > 0); where that telemetry is absent, the 13 x 13 "
    "getRectSubPix footprint of the reconstructed mask at the recorded pixel decides. From that frame on "
    "(inclusive) P3 is refuted by a stop whose measurement is a loss (not tracking) that "
    "closure_hold_explanation(measurement, latched_by_jaw=False) does not explain, or by a cut phase of align or "
    "closing; a stop while the tracker still reports tracking is not a loss.",
    "P3: the stop is mask-attributable when closure_hold_explanation(measurement, latched_by_jaw=False) explains "
    "it. It is on the final leg when the stop frame's visual_servo_decision.approach_phase is 'final', or, when that "
    "decision carries no approach_phase (the controller omits it while it holds for alignment or closure), when the "
    "controller's own live approach_phase of the previous frame is 'final' (the leg the robot was on when the command "
    "was issued); the source used is recorded. jaw_mask_occluded at frames 67-68 is an expectation clause, not a "
    "refutation condition. P3 is untested when the cut stops before the first touch, when the jaw never touches the "
    "patch, or when 19444 was refused or not recorded; when 19444 neither stops nor touches, the predicted stop did "
    "not occur but no registered refutation condition applies, and untested_reason says so.",
    "P4 is predicted for 530 runs whose gate mouth distance at closure start (cut.certificate.mouth_distance_m at "
    "k) is at most 4 mm, inclusive; runs above 4 mm, and runs whose settled recording has no closure start, are "
    "reported. The first closure loss L is the first frame in k+1..k+6 whose honest measurement "
    "(live_vision.measurement, never the held observation) is not tracking. A run with L = k+1 is judged (its first "
    "loss comes at another frame). Otherwise, when the cut phase recorded at k+1 is not 'closing' (the cut left "
    "closing at k+1 for a non-vision reason while the tracker still tracked, so k+2 was rendered with the jaw open), "
    "P4's premise, a first frame rendered closing at k+2, never arose: the run is listed as untested for P4 with the "
    "cut's stop frame and reason, in a clause that holds false and cannot refute. A run that tracks through k+3 is "
    "judged even when its recording ends later in the window (tracking then continues at k+3); a run whose recording "
    "ends before k+3 without a loss is not evaluable. Mask-attributable means closure_hold_explanation(measurement, "
    "latched_by_jaw=False) returns too_few_unmasked_patch_pixels or flow_failed_only_after_mask_drop. The shadow stop "
    "and the render check (k+1 rendered open, k+2 closing) are registered claims that cannot refute. The recorded "
    "closure_hold record at L is a recording-consistency echo (nothing is held before L, so it equals the recomputed "
    "attribution by construction) and is informational.",
    "P5's population is the 530 runs where P4's loss occurs: the first closure loss is at k+2 and mask-attributable, "
    "with the cut still closing after k+1 (so k+2 is the first frame rendered closing), whatever the closure-start "
    "distance. 'P4's loss' names the event P4 describes; P4's 4 mm bound limits where P4 predicts that event, not "
    "what the event is, so a run above 4 mm where it occurs is in the population. Every other 530 run is reported, "
    "not predicted, with its first loss, attribution, held frames, detach and cut stop. A broader reading (any "
    "mask-attributable first loss) would let runs outside P5's stated condition refute or support it. This reading "
    "was fixed after the items in construction_exposure, from the text alone. The hold must cover exactly k+2..k+6 "
    "(the deadline frame is held). The tool motion is recomputed with pose_change from the closure-start pose over "
    "k+1..k+6 with strict <; held mouth distances must stay within 0.05 mm (inclusive) of the value at k. A held "
    "frame that stops refutes; detachment at another frame refutes, and so does no detachment once frame k+6 is "
    "recorded or the cut stopped by k+6. Coverage, tool and mouth are registered claims that cannot refute. The label "
    "clause (vision_source closure_hold, the four waived checks and a held-target age of (i - k)/10 s) cannot refute "
    "and holds only when every held frame carries the label, because the label is the protocol's definition of a held "
    "frame; the hold's own stop reasons are informational (a held frame that triggers one already stops). A run in "
    "the population whose recording ends before k+6 with neither refutation established is not evaluable.",
    "P6 counts 530 rows with outcome 'pass' whose capture was not refused; a graded row with other than 17 checks "
    "stops the scorer. configuration_matches and the report's arm checks are reported per run. " + GRADER_NOTE,
    "P7 replays each 530 run that held through the frozen tools/replay_jaw_self_mask.replay_run with both flags off: "
    "the baseline tracker on the run's own recorded RGB, depth and camera. Its tracker output does not depend on cut "
    "decisions (the tracker updates on every frame), so the replay's evaluable and exact_counterfactual flags are not "
    "used, and reproduces_recording_through_recorded_stop is false by design (the live run used the mask). The anchor "
    "is the baseline's own target at the live closure-start frame k: P7 is about the baseline tracker's drift while "
    "tool and spur are still, and the live closure reference comes from another (masked) tracker whose offset from "
    "the baseline at k is not drift. The shift from the live closure reference is reported for every frame, and a "
    "disagreement of the two anchors about the 3 mm threshold is flagged without changing the verdict. 'Before the "
    "deadline' is k+1..k+5 (k+6 is reported).",
    "P7: the baseline's window ends only at a latched loss (tracking_lost for any reason, or jaw_mask_occluded); "
    "invalid_depth and invalid_calibration do not latch (the tracker can track again on the next frame), so they are "
    "listed as gaps and measuring continues, and a tracking row without a target is a gap too. A run supports P7 when "
    "a tracking frame in k+1..k+5 before any latched loss is at least 3 mm (inclusive) from the anchor. It refutes P7 "
    "only when every frame k+1..k+5 tracks with a target and each is within 3 mm: 'stays within 3 mm' needs the "
    "target measured on every frame before the deadline, and on an unmeasured frame the unchanged gate stops, so the "
    "hold was not unnecessary there. Otherwise, with no 3 mm move, a run is indeterminate when a latched loss or a gap "
    "falls in k+1..k+5, and untested when a frame of k+1..k+5 is missing from the replay. A run whose baseline is not "
    "tracking at k or whose replay fails is untested. 530 runs that never held are reported. P7 is untested when no "
    "530 run held, or when no held run's baseline either moved or stayed within (each untested or indeterminate).",
    "P8's window is closure start k through detach d, inclusive. The spur is the selected piece's PhysX pose "
    "origin (selected_piece_pose_wxyz); the mouth is tool_mouth_geometry with the frame's recorded mouth offset and "
    "closing axis. 'Changes by' is the largest deviation from the value at k (the net change is reported) and both "
    "limits are strict. frames.json target_position_m and target_distance_m are constants of the initial target "
    "and are not used. 530 runs that never detached are reported; P8 is untested when no 530 run detaches.",
    "construction_exposure lists what was seen of the live runs before this scorer was finished; no clause reads any "
    "of it, the runs table carries each run's entry, and the per-run views of P4-P8 point to it for the one run whose "
    "records were seen.",
    "Each run is labelled with its GPU model from report.node (score_planned_approach.gpu_model).",
]


class EvidenceError(ValueError):
    """The evidence does not describe the registered six runs; nothing is scored."""


class FrozenCodeError(ValueError):
    """The code to be imported is not the frozen code the runs executed; nothing is scored."""


# ---------------------------------------------------------------------------------------------- inputs


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256(path):
    return sha256_bytes(Path(path).read_bytes())


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mm(value_m):
    return None if value_m is None else 1e3 * float(value_m)


def batch_path(root, batch_dir):
    path = Path(batch_dir)
    return path if path.is_absolute() else Path(root) / path


def run_label(index, row):
    """The run directory run_vision_experiment and aggregate_eval give a planned target-sweep strategy row."""
    return (
        f"run_{index:02d}_{row['daylight']}_tree{row['target_tree_index']}_v{row['component_first_vertex']}"
        f"_{row['strategy']['name']}"
    )


def check_plan(name, plan):
    """The plan-level registration: this protocol, 200 frames at 10 fps, the arm, source light, both flags on."""
    if plan.get("protocol") != PROTOCOL or plan.get("frames") != PLANNED_FRAMES or plan.get("fps") != PLANNED_FPS:
        raise EvidenceError(f"{name}: the plan is not this protocol's 200 frames at 10 fps")
    registered = [(row.get("target_tree_index"), row.get("component_first_vertex")) for row in plan.get("runs") or []]
    if registered != list(REGISTERED_RUNS[name]):
        raise EvidenceError(f"{name}: the plan's targets {registered} are not the registered {REGISTERED_RUNS[name]}")
    for index, row in enumerate(plan["runs"]):
        strategy = row.get("strategy") or {}
        if (
            row.get("index") != index
            or row.get("daylight") != DAYLIGHT
            or strategy.get("name") != STRATEGY
            or strategy.get("jaw_self_mask") is not True
            or strategy.get("closure_hold") is not True
        ):
            raise EvidenceError(f"{name}: plan row {index} is not a source-light {STRATEGY} run with both arms on")


def pair_rows(name, plan, rows):
    """Each plan row with its one evidence row (matched by run_directory), in run-index order."""
    by_directory = {}
    for row in rows:
        by_directory.setdefault(row.get("run_directory"), []).append(row)
    pairs = []
    for index, plan_row in enumerate(plan["runs"]):
        directory = run_label(index, plan_row)
        matches = by_directory.pop(directory, [])
        if len(matches) != 1:
            raise EvidenceError(f"{name}: {len(matches)} evidence rows for planned run {directory}")
        row = matches[0]
        for key in ("target_tree_index", "component_first_vertex", "daylight", "photometric_normalization"):
            if row.get(key) != plan_row.get(key):
                raise EvidenceError(f"{name}: {directory} has {key} {row.get(key)!r}, the plan {plan_row.get(key)!r}")
        if row.get("strategy") != STRATEGY:
            raise EvidenceError(f"{name}: {directory} is not a {STRATEGY} row")
        if row.get("status") == "graded" and row.get("checks_total") != len(GRADER_CHECKS):
            raise EvidenceError(f"{name}: {directory} was graded with {row.get('checks_total')} checks, not 17")
        pairs.append((plan_row, row))
    if by_directory:
        raise EvidenceError(f"{name}: evidence rows for runs the plan does not have: {sorted(map(str, by_directory))}")
    return pairs


def validated_evidence(root):
    """The three batches' evidence and plans, after checking that together they are exactly the registered six runs.

    Returns ``[(path, document, batch_dir, plan, pairs)]`` in batch order, ``pairs`` being ``[(plan_row,
    evidence_row)]`` in run-index order. Any departure raises EvidenceError and nothing is scored.
    """
    found = sorted((Path(root) / EVIDENCE_DIR).glob("jaw-hold-*.json"))
    if [path.stem for path in found] != list(BATCHES):
        raise EvidenceError(f"expected the per-batch evidence of {', '.join(BATCHES)}; found {[p.name for p in found]}")
    validated, targets = [], []
    for path in found:
        document = read_json(path)
        batches = document.get("batches") or []
        if len(batches) != 1:
            raise EvidenceError(f"{path.name}: {len(batches)} batches, not one")
        batch_dir = batches[0].get("batch_dir")
        if not batch_dir or Path(batch_dir).name != path.stem:
            raise EvidenceError(f"{path.name}: batch_dir {batch_dir!r} is not the batch the file is named for")
        if document.get("protocol") != PROTOCOL:
            raise EvidenceError(f"{path.name}: aggregated under protocol {document.get('protocol')!r}")
        plan = read_json(batch_path(root, batch_dir) / "plan.json")
        check_plan(path.stem, plan)
        rows = document.get("runs") or []
        if not batches[0].get("planned_runs_kept") == len(plan["runs"]) == len(rows):
            raise EvidenceError(
                f"{path.name}: planned_runs_kept {batches[0].get('planned_runs_kept')}, {len(plan['runs'])} planned "
                f"runs and {len(rows)} evidence rows"
            )
        pairs = pair_rows(path.name, plan, rows)
        targets += [int(plan_row["component_first_vertex"]) for plan_row, _ in pairs]
        validated.append((path, document, batch_dir, plan, pairs))
    if Counter(targets) != Counter(EXPECTED_RUNS):
        raise EvidenceError(f"the evidence holds targets {dict(Counter(targets))}, not {EXPECTED_RUNS}")
    return validated


def plan_summary(validated):
    return {
        Path(batch_dir).name: {
            "code_revision": plan.get("code_revision"),
            "protocol": plan.get("protocol"),
            "frames": plan.get("frames"),
            "fps": plan.get("fps"),
            "strategy": (plan.get("strategy") or {}).get("name"),
            "targets": [row["component_first_vertex"] for row in plan["runs"]],
        }
        for _, _, batch_dir, plan, _ in validated
    }


# ---------------------------------------------------------------------------------------------- frozen code


def snapshot_drift(code_dir, frozen, names=SCORED_SOURCES):
    """Each named file whose sha256 under ``code_dir`` differs from the plan's (a missing file counts)."""
    drift = {}
    for name in names:
        path = Path(code_dir) / name
        digest = sha256(path) if path.is_file() else None
        if digest is None or digest != frozen.get(name):
            drift[name] = {"plan": frozen.get(name), "snapshot": digest}
    return drift


def checkout_report(frozen, code_revision):
    """Reported only: the checkout's copies of the scored sources that differ from the plan, and the git diff."""
    drift = {}
    for name in SCORED_SOURCES:
        path = ROOT / name
        digest = sha256(path) if path.is_file() else None
        if digest != frozen.get(name):
            drift[name] = {"plan": frozen.get(name), "checkout": digest}
    stat = ""
    if drift:
        try:
            stat = subprocess.check_output(
                ["git", "-C", str(ROOT), "diff", "--stat", f"{code_revision}..HEAD", "--", *drift],
                stderr=subprocess.STDOUT,
            ).decode()
        except (OSError, subprocess.CalledProcessError) as error:
            stat = f"unavailable: {error!r}"
    return {
        "note": "not imported; the scorer imports the batch snapshot",
        "files_differing_from_plan": drift,
        "git_diff_stat_plan_to_head": stat,
    }


def bind_frozen_code(code_dir, frozen):
    """Import the scored modules from ``code_dir`` and rebind this module's FROZEN_NAMES to them.

    Every module the import loads from under ``code_dir`` must hash to ``frozen`` (the plan's source_sha256), and
    every module in FROZEN_MODULES must come from there; otherwise FrozenCodeError. Returns the provenance.
    """
    code_dir = Path(code_dir).resolve()
    sys.dont_write_bytecode = True  # never write __pycache__ into the frozen snapshot
    for stale in [key for key in sys.modules if key == "isaaclab_pruning" or key.startswith("isaaclab_pruning.")]:
        del sys.modules[stale]
    sys.modules.pop("replay_jaw_self_mask", None)
    before = set(sys.modules)
    sys.path[:0] = [str(code_dir / "source/isaaclab_pruning"), str(code_dir / "tools")]
    importlib.invalidate_caches()
    modules = {name: importlib.import_module(name) for name in FROZEN_MODULES}
    loaded = {}
    for name in sorted(set(sys.modules) - before):
        path = getattr(sys.modules[name], "__file__", None)
        if not path or code_dir not in Path(path).resolve().parents:
            continue
        relative = Path(path).resolve().relative_to(code_dir).as_posix()
        digest = sha256(path)
        if frozen.get(relative) != digest:
            raise FrozenCodeError(f"{relative} under {code_dir} does not hash to the plan's value")
        loaded[relative] = digest
    for name, module in modules.items():
        if code_dir not in Path(module.__file__).resolve().parents:
            raise FrozenCodeError(f"{name} was imported from {module.__file__}, not from {code_dir}")
    for module_name, names in FROZEN_NAMES.items():
        for name in names:
            globals()[name] = getattr(modules[module_name], name)
    return {
        "imported_tree": str(code_dir),
        "modules": {name: str(Path(module.__file__).resolve()) for name, module in modules.items()},
        "loaded_files_sha256_equal_plan": loaded,
    }


def frozen_code(root, validated):
    """Check every batch's snapshot against its plan, import the scored modules from batch A's, and say so."""
    first_plan = validated[0][3]
    frozen = first_plan.get("source_sha256") or {}
    for path, _, batch_dir, plan, _ in validated:
        if plan.get("source_sha256") != frozen or plan.get("code_revision") != first_plan.get("code_revision"):
            raise FrozenCodeError(f"{path.stem}: its plan froze other sources than {BATCHES[0]}'s")
        drift = snapshot_drift(batch_path(root, batch_dir) / "code", frozen)
        if drift:
            raise FrozenCodeError(f"{path.stem}: its snapshot differs from its plan: {json.dumps(drift)}")
    provenance = bind_frozen_code(batch_path(root, validated[0][2]) / "code", frozen)
    return {
        "imported_from": "the snapshot <batch>/code the runs executed (run_vision_experiment sets PRUNING_ROOT to it "
        "and verify_snapshot checked it against plan.source_sha256 before each run)",
        **provenance,
        "plan_code_revision": first_plan.get("code_revision"),
        "scored_sources_equal_plan_in_every_snapshot": {name: frozen.get(name) for name in SCORED_SOURCES},
        "checkout": checkout_report(frozen, first_plan.get("code_revision")),
    }


# ---------------------------------------------------------------------------------------------- geometry


def missing_geometry(report):
    """The renderer-side inputs P1 needs that the report lacks (a run that aborted before the camera block)."""
    camera, scene = report.get("camera") or {}, report.get("blender_scene") or {}
    wanted = {
        "camera.wrist_intrinsics": camera.get("wrist_intrinsics"),
        "camera.wrist_resolution": camera.get("wrist_resolution"),
        "blender_scene.visual_proxy_roll_rad": scene.get("visual_proxy_roll_rad"),
        "blender_scene.target.radius_m": (scene.get("target") or {}).get("radius_m"),
    }
    return [name for name, value in wanted.items() if value is None]


def geometry(report):
    """The renderer-side inputs P1 reconstructs from: intrinsics, image shape, jaw roll and branch radius."""
    camera, scene = report["camera"], report["blender_scene"]
    width, height = camera["wrist_resolution"]
    return {
        "camera_matrix": np.asarray(camera["wrist_intrinsics"], dtype=float),
        "shape": (int(height), int(width)),
        "roll_rad": float(scene["visual_proxy_roll_rad"]),
        "radius_m": float(scene["target"]["radius_m"]),
    }


def pixel_rays(camera_matrix, shape):
    """Optical-frame direction, scaled to z = 1, of every pixel-centre ray, row-major over [v, u]."""
    height, width = shape
    us, vs = np.meshgrid(np.arange(width) + PIXEL_CENTRE, np.arange(height) + PIXEL_CENTRE)
    homogeneous = np.stack([us.ravel(), vs.ravel(), np.ones(us.size)], axis=1)
    rays = homogeneous @ np.linalg.inv(np.asarray(camera_matrix, dtype=float)).T
    return rays / rays[:, 2:3]


def optical_transform(frame):
    """world_from_optical exactly as the runner passed it to observe."""
    transform = np.eye(4)
    transform[:3, :3] = np.asarray(frame["wrist_rotation_w_ros"], dtype=float)
    transform[:3, 3] = np.asarray(frame["wrist_position_w_m"], dtype=float)
    return transform


def box_depth(boxes, world_from_optical, rays, shape):
    """Optical Z where each pixel-centre ray first enters either jaw box (slab test); inf where none is hit."""
    transform = np.asarray(world_from_optical, dtype=float)
    rotation, origin = transform[:3, :3], transform[:3, 3]
    best = np.full(len(rays), np.inf)
    for centre, rotation_w_box, half in boxes:
        rotation_o_box = rotation.T @ np.asarray(rotation_w_box, dtype=float)
        centre_o = rotation.T @ (np.asarray(centre, dtype=float) - origin)
        origin_box = -(rotation_o_box.T @ centre_o)  # the camera centre in box coordinates
        direction_box = rays @ rotation_o_box  # each ray in box coordinates
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = 1.0 / direction_box
            t1 = (-np.asarray(half) - origin_box) * inverse
            t2 = (np.asarray(half) - origin_box) * inverse
            near = np.minimum(t1, t2).max(axis=1)
            far = np.maximum(t1, t2).min(axis=1)
            hit = (near <= far) & (far >= 0)
        entry = np.where(near >= 0, near, far)  # rays are z = 1, so the parameter is optical Z
        best = np.where(hit, np.minimum(best, entry), best)
    return best.reshape(shape)


def silhouette_distance(boxes, camera_matrix, transform, us, vs):
    """Signed distance (px, negative inside) of pixel centres (index coordinates) to the nearest float32 jaw hull."""
    distance = np.full(np.shape(us), np.inf)
    for projection in jaw_projection(boxes, camera_matrix, transform):
        distance = np.minimum(distance, polygon_signed_distance(projection["hull"], us, vs))
    return distance


def depth_test(boxes, camera_matrix, transform, rays, shape, mask, depth):
    """P1's depth clause on one frame: recorded depth against the analytic box depth on every predicted jaw pixel.

    A predicted jaw pixel is any pixel whose centre ray hits either box (the registered reading). The pixels whose
    centre lies within EDGE_BAND_PX of the undilated silhouette edge are counted apart, as a sensitivity only.
    """
    recorded = np.asarray(depth, dtype=float)
    if recorded.shape != tuple(shape):
        return {"depth_checked": False, "depth_shape": list(recorded.shape)}
    box_z = box_depth(boxes, transform, rays, shape)
    hit = np.isfinite(box_z)
    vs, us = np.nonzero(hit)
    signed = np.full(hit.shape, np.nan)
    signed[vs, us] = silhouette_distance(boxes, camera_matrix, transform, us.astype(float), vs.astype(float))
    interior = np.zeros(hit.shape, bool)
    interior[vs, us] = signed[vs, us] <= -EDGE_BAND_PX
    finite = np.isfinite(recorded)
    with np.errstate(invalid="ignore"):
        excess = np.where(hit, recorded - np.where(hit, box_z, 0.0), np.nan)
        beyond = hit & (~finite | (excess > DEPTH_TOLERANCE_M))
        nearer = hit & finite & (excess < -DEPTH_TOLERANCE_M)
    edge_beyond = beyond & ~interior
    measured = hit & finite
    return {
        "depth_checked": True,
        "hit_pixels": int(hit.sum()),
        "violations": int(beyond.sum()),
        "nonfinite_hits": int((hit & ~finite).sum()),
        "max_excess_m": float(excess[measured].max()) if measured.any() else None,
        "edge_band_pixels": int((hit & ~interior).sum()),
        "violations_outside_edge_band": int((beyond & interior).sum()),
        "edge_band_beyond": [
            {
                "v": int(v),
                "u": int(u),
                "signed_distance_px": float(signed[v, u]),
                "box_z_m": float(box_z[v, u]),
                "recorded_m": float(recorded[v, u]) if finite[v, u] else None,
                "excess_m": float(excess[v, u]) if finite[v, u] else None,
            }
            for v, u in zip(*np.nonzero(edge_beyond), strict=True)
        ],
        "nearer_pixels": int(nearer.sum()),
        "hit_outside_mask": int((hit & ~np.asarray(mask, bool)).sum()),
    }


def margin_slack(boxes, geo, transform):
    """Smallest |signed distance - 2.0 px| over all pixel centres: how close a pixel came to flipping."""
    height, width = geo["shape"]
    us, vs = np.meshgrid(np.arange(width, dtype=float), np.arange(height, dtype=float))
    distance = silhouette_distance(boxes, geo["camera_matrix"], transform, us, vs)
    return float(np.min(np.abs(distance - MASK_MARGIN_PX)))


def p1_frame(geo, rays, frame, depth):
    """P1 on one recorded frame, and the reconstructed jaw ``(boxes, mask, transform)`` (None if it failed)."""
    record = (frame.get("live_vision") or {}).get("jaw_self_mask")
    progress = frame.get("visual_jaw_closure_progress")
    row = {
        "index": int(frame["index"]),
        "rendered_progress": progress,
        "recorded_count": None if record is None else record.get("mask_pixel_count"),
    }
    try:
        transform = optical_transform(frame)
        pose = np.asarray(frame["tool_pose_wxyz"], dtype=float)
        boxes = jaw_boxes(pose, geo["roll_rad"], float(progress), geo["radius_m"])
        mask = jaw_mask(boxes, geo["camera_matrix"], transform, geo["shape"])
        summary = jaw_mask_summary(boxes, mask, geo["camera_matrix"], transform)
    except (KeyError, TypeError, ValueError) as error:
        row.update(reconstruction_error=repr(error), count_equal=False)
        return row, None
    row["reconstructed_count"] = summary["mask_pixel_count"]
    if record is not None:
        corners = record.get("corners_px")
        same_shape = corners is not None and np.shape(corners) == np.shape(summary["corners_px"])
        roll = record.get("roll_rad")
        roll_difference = float(roll) - geo["roll_rad"] if isinstance(roll, (int, float)) else None
        row.update(
            count_equal=row["recorded_count"] == summary["mask_pixel_count"],
            progress_echo_equal=record.get("closure_progress_used") == progress,
            roll_difference_rad=roll_difference,
            roll_equal=roll_difference is not None and abs(roll_difference) <= ROLL_TOLERANCE_RAD,
            bbox_equal=record.get("mask_bbox_xyxy") == summary["mask_bbox_xyxy"],
            corner_max_abs_px=float(np.max(np.abs(np.subtract(corners, summary["corners_px"]))))
            if same_shape
            else None,
        )
        if not row["count_equal"]:
            row["margin_slack_px"] = margin_slack(boxes, geo, transform)
    if depth is not None:
        row.update(depth_test(boxes, geo["camera_matrix"], transform, rays, geo["shape"], mask, depth))
    return row, (boxes, mask, transform)


def tracked_pixel_facts(geo, frame, jaw):
    """Signed distance of the recorded pixel to the undilated jaw (both conventions) and the touch fallback."""
    measurement = (frame.get("live_vision") or {}).get("measurement") or {}
    pixel = measurement.get("pixel_xy")
    facts = {"sd_px": None, "sd_px_minus_half": None, "touch_fallback": None}
    if jaw is None or pixel is None:
        return facts
    boxes, mask, transform = jaw
    point = np.asarray(pixel, dtype=float)
    facts["sd_px"] = float(jaw_signed_distance_px(boxes, geo["camera_matrix"], transform, point)[0])
    facts["sd_px_minus_half"] = float(jaw_signed_distance_px(boxes, geo["camera_matrix"], transform, point - 0.5)[0])
    if "jaw_mask_patch_unmasked" not in measurement:
        # The tracker's own footprint: any nonzero getRectSubPix weight counts as masked.
        patch = cv2.getRectSubPix(mask.astype(np.float32), (PATCH_SIDE, PATCH_SIDE), (float(point[0]), float(point[1])))
        facts["touch_fallback"] = int(np.count_nonzero(patch > 0))
    return facts


def preview_masks(report, geo):
    """Informational: the preview and initialization mask counts, and a reconstruction with the rebuilt camera."""
    tracker = ((report.get("vision_initialization") or {}).get("tracker") or {}).get("jaw_self_mask") or {}
    initial = (report.get("initial_live_vision") or {}).get("jaw_self_mask") or {}
    out = {
        "initialization_mask_pixel_count": tracker.get("mask_pixel_count"),
        "preview_observation_mask_pixel_count": initial.get("mask_pixel_count"),
        "reconstructed_with_rebuilt_camera": None,
        "note": "not a recorded frame; its camera pose is rebuilt from the initial tool pose; not judged",
    }
    try:
        import replay_jaw_self_mask as replay

        pose = np.asarray(report["initial_tool_pose_wxyz"][0], dtype=float)
        boxes = jaw_boxes(pose, geo["roll_rad"], 0.0, geo["radius_m"])
        mask = jaw_mask(boxes, geo["camera_matrix"], replay.preview_transform(report), geo["shape"])
        out["reconstructed_with_rebuilt_camera"] = int(mask.sum())
    except (KeyError, TypeError, ValueError, IndexError) as error:
        out["reconstruction_error"] = repr(error)
    return out


def p1_summary(rows, frame_count_reported):
    """One run's P1 rows, summarized; every mismatch, violation and edge-band pixel beyond 1 mm is listed."""
    checked = [row for row in rows if "count_equal" in row]
    by_progress = {}
    for row in rows:
        if row.get("reconstructed_count") is not None and row.get("rendered_progress") is not None:
            by_progress.setdefault(f"{float(row['rendered_progress']):.6f}", set()).add(row["reconstructed_count"])
    depth_rows = [row for row in rows if row.get("depth_checked")]
    excess = [row["max_excess_m"] for row in depth_rows if row["max_excess_m"] is not None]
    hits = [row["hit_pixels"] for row in depth_rows]
    nearer = [row["nearer_pixels"] for row in depth_rows]
    corners = [row["corner_max_abs_px"] for row in checked if row.get("corner_max_abs_px") is not None]
    rolls = [abs(row["roll_difference_rad"]) for row in checked if row.get("roll_difference_rad") is not None]
    mismatch_keys = ("index", "rendered_progress", "recorded_count", "reconstructed_count", "margin_slack_px")
    return {
        "frames_recorded": len(rows),
        "frame_count_reported": frame_count_reported,
        "frames_checked": len(checked),
        "count_mismatches": [
            {key: row.get(key) for key in (*mismatch_keys, "reconstruction_error") if key in row}
            for row in checked
            if not row["count_equal"]
        ],
        "missing_mask_record_frames": [row["index"] for row in rows if "count_equal" not in row],
        "progress_echo_mismatch_frames": [row["index"] for row in checked if row.get("progress_echo_equal") is False],
        "roll_mismatch_frames": [row["index"] for row in checked if row.get("roll_equal") is False],
        "roll_max_abs_difference_rad": max(rolls, default=None),
        "bbox_mismatch_frames": [row["index"] for row in checked if row.get("bbox_equal") is False],
        "corner_max_abs_px": max(corners, default=None),
        "counts_by_rendered_progress": {key: sorted(value) for key, value in sorted(by_progress.items())},
        "depth": {
            "frames_checked": len(depth_rows),
            "missing_depth_frames": [row["index"] for row in rows if not row.get("depth_checked")],
            "unreadable_depth_frames": [row["index"] for row in rows if row.get("depth_note", "").startswith("unread")],
            "violations": sum(row["violations"] for row in depth_rows),
            "violation_frames": [
                {
                    "index": row["index"],
                    "violations": row["violations"],
                    "nonfinite_hits": row["nonfinite_hits"],
                    "within_edge_band": row["violations"] - row["violations_outside_edge_band"],
                    "max_excess_mm": mm(row["max_excess_m"]),
                }
                for row in depth_rows
                if row["violations"]
            ],
            "edge_band_sensitivity": {
                "edge_band_px": EDGE_BAND_PX,
                "violations_outside_edge_band": sum(row["violations_outside_edge_band"] for row in depth_rows),
                "edge_band_pixels": sum(row["edge_band_pixels"] for row in depth_rows),
                "edge_band_beyond": [
                    {"index": row["index"], **pixel} for row in depth_rows for pixel in row["edge_band_beyond"]
                ],
            },
            "max_excess_mm": mm(max(excess)) if excess else None,
            "hit_pixels_min_max": [min(hits), max(hits)] if hits else None,
            "nearer_pixels_min_max": [min(nearer), max(nearer)] if nearer else None,
            "hit_outside_mask": sum(row["hit_outside_mask"] for row in depth_rows),
        },
    }


# ---------------------------------------------------------------------------------------------- records


def frame_facts(frame):
    """The compact per-frame facts P2-P8 read, from one frames.json record (raw frames are not kept)."""
    live = frame.get("live_vision") or {}
    measurement = live.get("measurement") or {}
    cut = live.get("cut") or {}
    certificate = cut.get("certificate") or {}
    mask = live.get("jaw_self_mask")
    return {
        "index": int(frame["index"]),
        "phase": frame.get("phase"),
        "leg": (frame.get("visual_servo_decision") or {}).get("approach_phase"),
        "live_leg": live.get("approach_phase"),
        "measurement": {key: measurement[key] for key in MEASUREMENT_KEYS if key in measurement},
        "cut_phase": cut.get("phase"),
        "stopped_reason": cut.get("stopped_reason"),
        "detach_event": bool(cut.get("detach_event")),
        "closure_started_s": cut.get("closure_started_s"),
        "mouth_distance_m": certificate.get("mouth_distance_m"),
        "vision_source": certificate.get("vision_source"),
        "waived_checks": list(certificate.get("waived_checks") or []),
        "held_target_age_s": certificate.get("held_target_age_s"),
        "hold": live.get("closure_hold"),
        "shadow": live.get("cut_without_hold"),
        "mask_pixel_count": None if mask is None else mask.get("mask_pixel_count"),
        "rendered_progress": frame.get("visual_jaw_closure_progress"),
        "detach_requested": frame.get("detachment_requested_after_capture"),
        "tool_pose": frame.get("tool_pose_wxyz"),
        "piece_pose": frame.get("selected_piece_pose_wxyz"),
        "mouth_offset": live.get("mouth_offset_tool_m"),
        "closing_axis": live.get("closing_axis_tool"),
        "sd_px": None,
        "sd_px_minus_half": None,
        "touch_fallback": None,
    }


def _normalized(value):
    return json.loads(json.dumps(value))


def configuration_problems(report, plan_row):
    """Where the report departs from the planned condition: the arms and their registered values, the target, the
    light and the approach. The frame count is left to the grader (an incomplete capture is incomplete, not
    refused)."""
    problems = []
    jaw, hold = report.get("jaw_self_mask") or {}, report.get("closure_hold") or {}
    if jaw.get("enabled") is not True:
        problems.append("the report does not show jaw_self_mask enabled")
    if hold.get("enabled") is not True:
        problems.append("the report does not show closure_hold enabled")
    if _normalized(jaw.get("constants")) != _normalized(registered_jaw_self_mask()):
        problems.append("the report's jaw_self_mask constants are not the registered ones")
    if _normalized(hold.get("thresholds")) != _normalized(registered_closure_hold()):
        problems.append("the report's closure_hold thresholds are not the registered ones")
    if list(hold.get("waived_checks") or []) != list(CLOSURE_HOLD_WAIVED_CHECKS):
        problems.append("the report's waived checks are not the registered four")
    tracker = report.get("tracker_config")
    minimum = registered_jaw_self_mask()["min_unmasked_patch_elements"]
    if tracker is not None and tracker.get("min_unmasked_patch_elements") != minimum:
        problems.append(f"the tracker's min_unmasked_patch_elements is not {minimum}")
    scene = report.get("blender_scene")
    if scene is not None:
        expected = f"tree{plan_row['target_tree_index']}_SPUR_component_{plan_row['component_first_vertex']}"
        if (scene.get("target") or {}).get("id") != expected:
            problems.append(f"the report's target is not {expected}")
        if (scene.get("daylight") or {}).get("preset") != plan_row["daylight"]:
            problems.append(f"the report's daylight is not {plan_row['daylight']}")
    if report.get("photometric_normalization") != plan_row.get("photometric_normalization"):
        problems.append("the report's photometric normalization is not the plan's")
    recorded, wanted = report.get("approach_strategy") or {}, plan_row.get("strategy") or {}
    if any(recorded.get(key) != wanted.get(key) for key in ("mode", "standoff_m", "max_step_m")):
        problems.append("the report's approach mode, standoff or step is not the plan's")
    quat, wanted_quat = recorded.get("planned_tool_quat_wxyz"), wanted.get("planned_tool_quat_wxyz")
    if wanted_quat is not None and (
        quat is None or len(quat) != 4 or any(abs(float(a) - float(b)) >= 1e-6 for a, b in zip(quat, wanted_quat))
    ):
        problems.append("the report's planned tool orientation is not the plan's")
    return problems


def refusal(flag, report, plan_row):
    """``(refused, reasons, problems)``: the protocol's refusal of a capture without the arms, from its own records."""
    if report is None:
        return False, [], []
    problems = configuration_problems(report, plan_row)
    if flag is True:
        return False, [], problems  # configuration_matches accepted it; any problem is reported as a contradiction
    if flag is False and report.get("frame_count") == PLANNED_FRAMES:
        return True, ["configuration_matches is false on a complete capture", *problems], problems
    return bool(problems), problems, problems


def report_summary(report):
    """The run's configuration as recorded: arms, registered constants, target and the initial tracker."""
    jaw = report.get("jaw_self_mask") or {}
    hold = report.get("closure_hold") or {}
    tracker = (report.get("vision_initialization") or {}).get("tracker") or {}
    tracker_config = report.get("tracker_config") or (report.get("initial_live_vision") or {}).get("tracker_config")
    scene = report.get("blender_scene") or {}
    return {
        "jaw_self_mask_enabled": jaw.get("enabled"),
        "closure_hold_enabled": hold.get("enabled"),
        "registered_constants_recorded": _normalized(jaw.get("constants")) == _normalized(registered_jaw_self_mask())
        and _normalized(hold.get("thresholds")) == _normalized(registered_closure_hold())
        and list(hold.get("waived_checks") or []) == list(CLOSURE_HOLD_WAIVED_CHECKS),
        "min_unmasked_patch_elements": (tracker_config or {}).get("min_unmasked_patch_elements"),
        "target_id": (scene.get("target") or {}).get("id"),
        "daylight": (scene.get("daylight") or {}).get("preset"),
        "stage": report.get("stage"),
        "task_outcome": report.get("task_outcome"),
        "frame_count": report.get("frame_count"),
        "stopped_reason": report.get("stopped_reason"),
        "error": report.get("error"),
        "initial_tracker": {
            "state": tracker.get("state"),
            "feature_count": tracker.get("feature_count"),
            "jaw_mask_roi_pixels": tracker.get("jaw_mask_roi_pixels"),
        },
        "visual_proxy_roll_rad": scene.get("visual_proxy_roll_rad"),
    }


def base_record(root, batch_dir, plan_row, row):
    """A planned run's evidence row, before its own records are read."""
    failed = row.get("failed_checks")
    return {
        "batch": Path(batch_dir).name,
        "run_index": int(plan_row["index"]),
        "run_directory": row["run_directory"],
        "run_path": str(batch_path(root, batch_dir) / row["run_directory"]),
        "target": int(row["component_first_vertex"]),
        "target_tree_index": row.get("target_tree_index"),
        "strategy": row.get("strategy"),
        **{key: row.get(key) for key in ("outcome", "status", "checks_passed", "checks_total")},
        "failed_checks": sorted(failed.split(",")) if failed else [],
        "stop_reason": row.get("stop_reason"),
        "stop_frame": row.get("stop_frame"),
        "node": None,
        "gpu_model": None,
        "configuration_matches": None,
        "refused": False,
        "refused_reasons": [],
        "configuration_problems": [],
        "fps": None,
        "report": None,
        "frames": None,
        "not_recorded_reason": None,
        "p1": None,
        "baseline": None,
        "inputs": {},
    }


def read_capture(run, record, relative):
    """``(report, frames, why not recorded)``; a missing or unreadable file is recorded, never fatal."""
    paths = {name: run / name for name in ("report.json", "frames.json")}
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        return None, None, "no " + " and no ".join(missing)
    parsed = {}
    for name, path in paths.items():
        data = path.read_bytes()
        record["inputs"][f"{relative}/{name}"] = sha256_bytes(data)
        try:
            parsed[name] = json.loads(data)
        except ValueError as error:
            parsed[name] = error
        del data
    report, document = parsed["report.json"], parsed["frames.json"]
    if not isinstance(report, dict):
        return None, None, f"report.json is unreadable: {report!r}"
    frames = document.get("frames") if isinstance(document, dict) else None
    if not isinstance(frames, list):
        return report, None, "frames.json is unreadable or has no frames list"
    record["fps"] = document.get("fps")
    return report, frames, None


def read_depth(run, index, digest):
    """The frame's depth array, or None with a note (missing or unreadable)."""
    name = f"depth_{index:05d}.npy"
    path = run / "frames" / name
    if not path.is_file():
        return None, "missing"
    data = path.read_bytes()
    digest.update(f"{name} {sha256_bytes(data)}\n".encode())
    try:
        return np.load(io.BytesIO(data), allow_pickle=False), None
    except (ValueError, OSError, EOFError) as error:
        return None, f"unreadable: {error!r}"


def run_record(root, batch_dir, plan_row, row):
    """One planned run: its evidence row, its refusal status, its per-frame facts and P1 on every recorded frame."""
    record = base_record(root, batch_dir, plan_row, row)
    run, relative = Path(record["run_path"]), f"{batch_dir}/{row['run_directory']}"
    experiment = run / "experiment_result.json"
    if experiment.is_file():
        data = experiment.read_bytes()
        record["inputs"][f"{relative}/experiment_result.json"] = sha256_bytes(data)
        try:
            flag = json.loads(data).get("configuration_matches")
        except (ValueError, AttributeError):
            flag = None
        record["configuration_matches"] = flag if isinstance(flag, bool) else None
    report, frames, why = read_capture(run, record, relative)
    if report is not None:
        refused, reasons, problems = refusal(record["configuration_matches"], report, plan_row)
        record.update(
            node=report.get("node"),
            gpu_model=gpu_model(report.get("node")),
            report=report_summary(report),
            refused=refused,
            refused_reasons=reasons,
            configuration_problems=problems,
        )
        if why is None and missing_geometry(report):
            why = "the report lacks " + ", ".join(missing_geometry(report)) + " (aborted before the camera block)"
        elif why is None and not frames:
            why = "frames.json holds no frame"
    if why is not None:
        record["not_recorded_reason"] = why
        return record
    geo = geometry(report)
    rays = pixel_rays(geo["camera_matrix"], geo["shape"])
    facts, rows, digest = [], [], hashlib.sha256()
    for frame in frames:
        fact = frame_facts(frame)
        depth, note = read_depth(run, fact["index"], digest)
        row_p1, jaw = p1_frame(geo, rays, frame, depth)
        if note is not None:
            row_p1["depth_note"] = note
        fact.update(tracked_pixel_facts(geo, frame, jaw))
        facts.append(fact)
        rows.append(row_p1)
    record["inputs"][f"{relative}/frames/depth_*.npy (sha256 of '<name> <sha256>' lines in frame order)"] = (
        digest.hexdigest()
    )
    record["frames"] = facts
    record["p1"] = {**p1_summary(rows, report.get("frame_count")), "preview": preview_masks(report, geo)}
    return record


def baseline_replay(run_dir):
    """P7's baseline: the run's own recorded frames replayed through the frozen controller, both flags off."""
    import replay_jaw_self_mask as replay

    try:
        result = replay.replay_run(Path(run_dir), jaw_self_mask=False, closure_hold=False)
    except Exception as error:  # noqa: BLE001 - a failed replay is recorded, never dropped
        return {"error": repr(error)}
    initialization = {key: value for key, value in result["initialization"].items() if key != "jaw_self_mask"}
    baseline = {
        "replay_module": replay.__file__,
        "flags": result["flags"],
        "rows": [
            {
                "index": row["index"],
                "state": row["state"],
                "reason": row["reason"],
                "target": row["target_position_world_m"],
            }
            for row in result["frames_detail"]
        ],
        "initialization": initialization,
        "recorded_closure_start_frame": result["recorded"]["closure_start_frame"],
        "reproduces_recording_through_recorded_stop": result["reproduces_recording_through_recorded_stop"],
        "tracker_config_matches_recording": result["tracker_config_matches_recording"],
    }
    del result
    gc.collect()
    return baseline


def load(root, validated=None):
    """Every planned run of the three batches, in batch and run order, with the baseline replay where P7 needs it."""
    records = []
    batches = validated if validated is not None else validated_evidence(root)
    for _, _, batch_dir, _, pairs in batches:
        for plan_row, row in pairs:
            record = run_record(root, batch_dir, plan_row, row)
            if record["target"] == 530 and not record["refused"] and closure_facts(record)["held_frames"]:
                record["baseline"] = baseline_replay(record["run_path"])
            records.append(record)
            gc.collect()
    return records


# ---------------------------------------------------------------------------------------------- shared facts


def run_key(record):
    return f"{record['batch']} {record['target']}"


def runs_of(records, target):
    return [record for record in records if record["target"] == target]


def indexed(record):
    return {fact["index"]: fact for fact in record.get("frames") or []}


def state_of(fact):
    return None if fact is None else fact["measurement"].get("state")


def first_index(facts, predicate):
    return next((fact["index"] for fact in facts if predicate(fact)), None)


def recording_complete(record):
    """All 200 planned frames are recorded, in order."""
    return [fact["index"] for fact in record.get("frames") or []] == list(range(PLANNED_FRAMES))


def settled(record):
    """The recording shows all the run will do: it is complete, or the cut already stopped or detached."""
    facts = record.get("frames") or []
    return recording_complete(record) or any(f["cut_phase"] == "stopped" or f["detach_event"] for f in facts)


def last_frame(record):
    facts = record.get("frames") or []
    return facts[-1]["index"] if facts else None


def exclusion(record):
    """Why a run is judged by no prediction (refused or not recorded), or None."""
    if record["refused"]:
        return {"outcome": record["outcome"], "status": record["status"], "refused": record["refused_reasons"]}
    if not record.get("frames"):
        return {"outcome": record["outcome"], "status": record["status"], "not_recorded": record["not_recorded_reason"]}
    return None


def seen(record, view):
    """A per-run view, with a pointer to construction_exposure when this run's own records were seen."""
    if (record["batch"], record.get("run_index")) in RECORDS_SEEN:
        view = {**view, "records_seen_before_scoring": "see construction_exposure"}
    return view


def judged(pid, clauses, untested=None):
    """A prediction with the 'untested' override: refuted wins; otherwise ``untested`` names what never arose."""
    result = prediction(pid, TEXT[pid], clauses)
    if untested and result["verdict"] != "refuted":
        result["verdict"] = "untested"
        result["untested_reason"] = untested
    return result


def vacuous(text, observed, reason):
    """A clause whose condition never arose: it neither holds nor refutes, and says why."""
    return clause(text, {**observed, "vacuous": True, "untested": reason}, False, refutes=False)


def not_judged(text, entries):
    """Runs a prediction covers that could not be judged: never a refutation, but no support while they remain."""
    return clause(text + " (not judged; not a refutation condition)", entries, False)


def closure_facts(record):
    """Closure start, the first closure loss and its attribution, the held set, detach and stops of one run."""
    facts = record.get("frames") or []
    frames = indexed(record)
    k = first_index(facts, lambda f: f["cut_phase"] == "closing")
    stop = first_index(facts, lambda f: f["cut_phase"] == "stopped")
    shadow = first_index(facts, lambda f: (f["shadow"] or {}).get("phase") == "stopped")
    out = {
        "closure_start_frame": k,
        "closure_started_s": None if k is None else frames[k]["closure_started_s"],
        "mouth_distance_at_start_m": None if k is None else frames[k]["mouth_distance_m"],
        "deadline_frame": None if k is None else k + CLOSURE_FRAMES,
        "first_loss_frame": None,
        "scanned_through": None,
        "recorded_through_deadline": None,
        "cut_phase_k_plus_1": None,
        "attributable": False,
        "explanation": None,
        "detach_frame": first_index(facts, lambda f: f["detach_event"]),
        "cut_stop_frame": stop,
        "cut_stop_reason": None if stop is None else frames[stop]["stopped_reason"],
        "held_frames": [f["index"] for f in facts if (f["hold"] or {}).get("held")],
        "shadow_stop_frame": shadow,
        "shadow_stop_reason": None if shadow is None else frames[shadow]["shadow"].get("stopped_reason"),
    }
    if k is None:
        return out
    scanned = k
    for index in range(k + 1, k + CLOSURE_FRAMES + 1):
        if index not in frames:
            break
        scanned = index
        if state_of(frames[index]) != "tracking":
            out["first_loss_frame"] = index
            break
    out["scanned_through"] = scanned
    out["recorded_through_deadline"] = all(index in frames for index in range(k, k + CLOSURE_FRAMES + 1))
    out["cut_phase_k_plus_1"] = (frames.get(k + 1) or {}).get("cut_phase")
    loss = out["first_loss_frame"]
    if loss is not None:
        measurement = frames[loss]["measurement"]
        explained, explanation = closure_hold_explanation(measurement, latched_by_jaw=False)
        hold = frames[loss]["hold"] or {}
        out.update(
            attributable=bool(explained) and explanation in ATTRIBUTABLE,
            explanation=explanation,
            state_at_loss=measurement.get("state"),
            reason_at_loss=measurement.get("reason"),
            recorded_hold_at_loss={key: hold.get(key) for key in ("eligible", "stationary", "explained", "explanation")}
            | {"held": hold.get("held")},
        )
    return out


def closure_brief(record, c=None):
    """The closure facts every 530 table shows for a run outside a prediction's condition."""
    c = c or closure_facts(record)
    k = c["closure_start_frame"]
    return seen(
        record,
        {
            "outcome": record["outcome"],
            "closure_start_frame": k,
            "mouth_distance_at_start_mm": mm(c["mouth_distance_at_start_m"]),
            "first_loss_frame": c["first_loss_frame"],
            "first_loss_offset": None if c["first_loss_frame"] is None or k is None else c["first_loss_frame"] - k,
            "mask_explanation": c["explanation"],
            "held_frames": c["held_frames"],
            "detach_frame": c["detach_frame"],
            "cut_stop": [c["cut_stop_frame"], c["cut_stop_reason"]],
            "last_recorded_frame": last_frame(record),
        },
    )


# ---------------------------------------------------------------------------------------------- P1


def p1_views(p1):
    counts = {
        "frames_checked": p1["frames_checked"],
        "count_mismatches": p1["count_mismatches"],
        "counts_by_rendered_progress": p1["counts_by_rendered_progress"],
        "preview_not_judged": p1.get("preview"),
    }
    asserted = {
        name: p1[name]
        for name in (
            "roll_mismatch_frames",
            "roll_max_abs_difference_rad",
            "progress_echo_mismatch_frames",
            "bbox_mismatch_frames",
            "corner_max_abs_px",
        )
    }
    return counts, asserted


def score_p1(records):
    counts, depths, asserted, coverage = {}, {}, {}, {}
    mismatched, violated, outside_band, uncovered, asserted_off = [], [], [], [], []
    for record in records:
        key, p1, excluded = run_key(record), record.get("p1"), exclusion(record)
        if excluded is not None or p1 is None:
            coverage[key] = {"judged": False, **(excluded or {"not_recorded": record["not_recorded_reason"]})}
            uncovered.append(key)
            continue
        counts[key], asserted[key] = p1_views(p1)
        depths[key] = p1["depth"]
        coverage[key] = {
            "judged": True,
            "frames_recorded": p1["frames_recorded"],
            "frames_planned": PLANNED_FRAMES,
            "frame_count_reported": p1["frame_count_reported"],
            "missing_mask_record_frames": p1["missing_mask_record_frames"],
            "missing_depth_frames": p1["depth"]["missing_depth_frames"],
        }
        mismatched += [key] if p1["count_mismatches"] else []
        violated += [key] if p1["depth"]["violation_frames"] else []
        outside_band += [key] if p1["depth"]["edge_band_sensitivity"]["violations_outside_edge_band"] else []
        if (
            p1["missing_mask_record_frames"]
            or p1["depth"]["missing_depth_frames"]
            or p1["frames_recorded"] != PLANNED_FRAMES
        ):
            uncovered.append(key)
        if p1["roll_mismatch_frames"] or p1["progress_echo_mismatch_frames"]:
            asserted_off.append(key)
    planned = sum(EXPECTED_RUNS.values())
    coverage["planned_runs"] = {"expected": planned, "present": len(records)}
    checked = sum(entry["frames_checked"] for entry in counts.values())
    depth_checked = sum(entry["frames_checked"] for entry in depths.values())
    depth_observed = {
        "runs": depths,
        "runs_with_violations": violated,
        "violations": sum(entry["violations"] for entry in depths.values()),
        "edge_band_sensitivity": {
            "note": "not the registered reading and never a verdict: the count leaving out hits whose centre lies "
            "within 0.01 px of the undilated silhouette edge",
            "edge_band_px": EDGE_BAND_PX,
            "runs_with_violations_outside_edge_band": outside_band,
            "violations_outside_edge_band": sum(
                entry["edge_band_sensitivity"]["violations_outside_edge_band"] for entry in depths.values()
            ),
            "would_refute": bool(outside_band),
        },
    }
    clauses = [
        clause(
            "the recorded mask pixel count equals the offline reconstruction on every recorded frame",
            counts,
            checked > 0 and not mismatched,
            refutes=bool(mismatched),
        ),
        clause(
            "no predicted jaw pixel (any pixel whose centre ray hits either box) shows recorded depth more than 1 mm "
            "beyond the analytic box depth",
            depth_observed,
            depth_checked > 0 and not violated,
            refutes=bool(violated),
        ),
        clause(
            "the controller's mask record names the rendered jaw: its roll equals the renderer's within 1e-12 rad "
            "(its progress is a recording echo; bbox and corners reported; not a refutation condition)",
            asserted,
            bool(asserted) and not asserted_off,
        ),
        clause(
            "every frame of all six planned runs was recorded and checked (coverage; not a refutation condition)",
            coverage,
            not uncovered and len(records) == planned,
        ),
    ]
    return judged("P1", clauses, None if counts else "no planned run has judged frames")


# ---------------------------------------------------------------------------------------------- P2


def control_window(record):
    """P2's per-frame facts for one control: the window through detach, losses, masked patches, holds."""
    facts = record.get("frames")
    detaches = [f["index"] for f in facts if f["detach_event"]]
    detach = detaches[0] if detaches else None
    window = [f for f in facts if detach is None or f["index"] <= detach]

    def unmasked(fact):
        return fact["measurement"].get("jaw_mask_patch_unmasked")

    return {
        "detach_frame": detach,
        "detach_frames": detaches,
        "detach_requested_frames": [f["index"] for f in facts if f["detach_requested"]],
        "window": [window[0]["index"], window[-1]["index"]] if window else None,
        "window_complete": detach is not None or recording_complete(record),
        "not_tracking": [f["index"] for f in window if state_of(f) != "tracking"],
        "not_tracking_strict": [f["index"] for f in window if state_of(f) != "tracking" and f["index"] != detach],
        "masked": [
            {"frame": f["index"], "unmasked": unmasked(f)}
            for f in window
            if unmasked(f) is not None and unmasked(f) < PATCH_ELEMENTS
        ],
        "masked_strict_frames": [
            f["index"]
            for f in window
            if unmasked(f) is not None and unmasked(f) < PATCH_ELEMENTS and f["index"] != detach
        ],
        "patch_telemetry_missing_while_tracking": [
            f["index"] for f in window if state_of(f) == "tracking" and unmasked(f) is None
        ],
        "held_frames": [
            f["index"] for f in facts if (f["hold"] or {}).get("held") or f["vision_source"] == "closure_hold"
        ],
        "feature_drops_not_judged": {
            "frames_with_dropped_valid": [
                f["index"] for f in window if (f["measurement"].get("jaw_mask_dropped_valid") or 0) > 0
            ],
            "max_in_mask_any": max((f["measurement"].get("jaw_mask_in_mask_any") or 0 for f in window), default=0),
        },
    }


def score_p2(records):
    controls = [record for target in CONTROLS for record in runs_of(records, target)]
    grade, windows, excluded = {}, {}, {}
    for record in controls:
        key = run_key(record)
        grade[key] = {
            "outcome": record["outcome"],
            "status": record["status"],
            "checks": f"{record['checks_passed']}/{record['checks_total']}",
            "failed_checks": record["failed_checks"],
            "gpu_model": record["gpu_model"],
            "refused": record["refused_reasons"] if record["refused"] else False,
        }
        if exclusion(record) is not None:
            excluded[key] = exclusion(record)
        else:
            windows[key] = control_window(record)
    present = all(runs_of(records, target) for target in CONTROLS)
    every_pass = present and all(r["outcome"] == "pass" and not r["refused"] for r in controls)
    failed_graded = [
        run_key(r) for r in controls if not r["refused"] and r["status"] == "graded" and r["outcome"] != "pass"
    ]
    all_judged = present and not excluded and all(w["window_complete"] for w in windows.values())

    def view(*names):
        return {key: {name: w[name] for name in names} for key, w in windows.items()}

    lost = any(w["not_tracking"] for w in windows.values())
    masked = any(w["masked"] for w in windows.values())
    held = any(w["held_frames"] for w in windows.values())
    missing_telemetry = any(w["patch_telemetry_missing_while_tracking"] for w in windows.values())
    clauses = [
        clause(
            "14944 and 15004 pass 17/17",
            {**grade, "graded_with_a_failed_check": failed_graded},
            every_pass,
            refutes=bool(failed_graded),
        ),
        clause(
            "the tracker reports tracking on every frame before detach (0..d inclusive; 0..d-1 reported)",
            view(
                "detach_frame", "detach_frames", "detach_requested_frames", "window", "window_complete", "not_tracking"
            )
            | {"strict_window_not_tracking": {k: w["not_tracking_strict"] for k, w in windows.items()}},
            all_judged and not lost,
            refutes=lost,
        ),
        clause(
            "the mask removes no patch element before detach (jaw_mask_patch_unmasked == 169)",
            view(
                "masked", "masked_strict_frames", "patch_telemetry_missing_while_tracking", "feature_drops_not_judged"
            ),
            all_judged and not masked and not missing_telemetry,
            refutes=masked,
        ),
        clause("the hold never applies", view("held_frames"), all_judged and not held, refutes=held),
    ]
    if excluded:
        clauses.append(not_judged("controls refused or not recorded", excluded))
    return judged("P2", clauses)


# ---------------------------------------------------------------------------------------------- P3


def first_touch(facts):
    """First frame whose previous or current patch has a masked element: telemetry, else the reconstruction."""
    for fact in facts:
        measurement = fact["measurement"]
        if "jaw_mask_patch_unmasked" in measurement:
            masked_prev = measurement.get("jaw_mask_patch_masked_prev") or 0
            masked_cur = measurement.get("jaw_mask_patch_masked_cur") or 0
            if masked_prev + masked_cur > 0:
                return {
                    "frame": fact["index"],
                    "source": "telemetry",
                    "masked_prev_cur": [masked_prev, masked_cur],
                    "kept": measurement["jaw_mask_patch_unmasked"],
                }
        elif fact["touch_fallback"]:
            return {
                "frame": fact["index"],
                "source": "reconstructed mask footprint at the recorded pixel (no patch telemetry)",
                "masked_current": fact["touch_fallback"],
            }
    return None


def silhouette(facts):
    """Clause (b): tracking frames whose pixel lies inside the undilated jaw silhouette (both conventions)."""
    tracking = [f for f in facts if state_of(f) == "tracking" and f["sd_px"] is not None]
    # rint of a pixel outside the dilated mask is more than 2 - sqrt(0.5) px from the hull.
    bound = 2.0 - math.sqrt(0.5)
    return {
        "tracking_frames_checked": len(tracking),
        "tracking_frames_without_pixel_geometry": [
            f["index"] for f in facts if state_of(f) == "tracking" and f["sd_px"] is None
        ],
        "inside_frames": [f["index"] for f in tracking if f["sd_px"] <= 0],
        "min_signed_distance_px": min((f["sd_px"] for f in tracking), default=None),
        "sensitivity_pixel_minus_half": {
            "inside_frames": [f["index"] for f in tracking if f["sd_px_minus_half"] <= 0],
            "min_signed_distance_px": min((f["sd_px_minus_half"] for f in tracking), default=None),
        },
        "telemetry_cross_check_inconsistent_frames": [
            f["index"]
            for f in tracking
            if f["measurement"].get("jaw_mask_pixel_in_mask") is False and f["sd_px"] <= bound
        ],
    }


def stop_leg(frames, stop):
    """The leg of the stop frame's command, or, when that decision has none, the live leg of the frame before."""
    at_stop = frames.get(stop) if stop is not None else None
    if at_stop is None:
        return None, None
    if at_stop["leg"] is not None:
        return at_stop["leg"], "visual_servo_decision of the stop frame"
    before = frames.get(stop - 1)
    if before is not None and before["live_leg"] is not None:
        return before["live_leg"], "live approach_phase of the previous frame (the decision carries none)"
    return None, None


def p3_untested(touch, stop, record):
    if touch is None:
        reason = "the jaw never touched the tracked patch"
        if stop is not None:
            return reason + f"; the cut stopped at frame {stop}, before the jaw reached the patch"
        if not recording_complete(record):
            return reason + f"; the recording ends at frame {last_frame(record)} without a stop"
        return (
            reason + "; 19444 never stopped either, so the predicted stop did not occur, but no registered refutation "
            "condition applies"
        )
    if stop is not None and stop < touch["frame"]:
        return f"the cut stopped at frame {stop}, before the jaw first touched the patch at frame {touch['frame']}"
    return None


def score_p3(records):
    runs = runs_of(records, 19444)
    usable = [r for r in runs if exclusion(r) is None]
    if not usable:
        observed = {run_key(r): exclusion(r) for r in runs}
        reason = "19444's capture was refused" if any(r["refused"] for r in runs) else "19444 was not recorded"
        return judged(
            "P3", [vacuous("19444 stops on the final leg with a mask-attributable state", observed, reason)], reason
        )
    record = usable[0]
    key, facts, frames = run_key(record), record["frames"], indexed(record)
    stop = first_index(facts, lambda f: f["cut_phase"] == "stopped")
    at_stop = frames.get(stop) if stop is not None else None
    measurement = at_stop["measurement"] if at_stop else {}
    explained, explanation = closure_hold_explanation(measurement, latched_by_jaw=False)
    leg, leg_source = stop_leg(frames, stop)
    stop_observed = {
        "run": key,
        "outcome": record["outcome"],
        "gpu_model": record["gpu_model"],
        "stop_decision_frame": stop,
        "evidence_stop_frame": record["stop_frame"],
        "stopped_reason": at_stop and at_stop["stopped_reason"],
        "leg": leg,
        "leg_source": leg_source,
        "live_leg_at_stop": at_stop and at_stop["live_leg"],
        "state": measurement.get("state"),
        "reason": measurement.get("reason"),
        "kept_patch_elements": measurement.get("jaw_mask_patch_unmasked"),
        "jaw_mask_dropped_valid": measurement.get("jaw_mask_dropped_valid"),
        "jaw_mask_unmasked_flow_would_pass": measurement.get("jaw_mask_unmasked_flow_would_pass"),
        "mask_explanation": explanation,
        "recording_complete": recording_complete(record),
    }
    final_attributable = at_stop is not None and leg == "final" and explained
    expected = stop in EXPECTED_STOP_FRAMES_19444 and measurement.get("state") == "jaw_mask_occluded"
    inside = silhouette(facts)
    touch = first_touch(facts)
    untested = p3_untested(touch, stop, record)
    after_text = "from the jaw's first touch of the patch on, no stop on a loss the mask does not explain and no align"
    if touch is None:
        after = vacuous(after_text, {"first_touch": None}, untested)
    else:
        unexplained_stop = (
            stop is not None and stop >= touch["frame"] and measurement.get("state") != "tracking" and not explained
        )
        aligned = [f["index"] for f in facts if f["index"] >= touch["frame"] and f["cut_phase"] in ("align", "closing")]
        after = clause(
            after_text,
            {"first_touch": touch, "unexplained_loss_stop": unexplained_stop, "align_or_closing_frames": aligned},
            not unexplained_stop and not aligned,
            refutes=unexplained_stop or bool(aligned),
        )
    clauses = [
        clause("19444 stops on the final leg with a mask-attributable state", stop_observed, final_attributable),
        clause(
            "the stop state is jaw_mask_occluded at frame 67 or 68 (an expectation, not a refutation condition)",
            {"stop_decision_frame": stop, "state": measurement.get("state")},
            expected,
        ),
        clause(
            "no frame reports tracking with the tracked pixel inside the jaw silhouette",
            {**inside, "recording_complete": recording_complete(record)},
            recording_complete(record)
            and not inside["inside_frames"]
            and not inside["tracking_frames_without_pixel_geometry"],
            refutes=bool(inside["inside_frames"]),
        ),
        after,
    ]
    return judged("P3", clauses, untested)


# ---------------------------------------------------------------------------------------------- P4


def mechanism(record):
    """P4's view of one 530 run: closure start, the first loss and its attribution, shadow and render check."""
    c = closure_facts(record)
    frames = indexed(record)
    k, loss = c["closure_start_frame"], c["first_loss_frame"]
    out = {
        "outcome": record["outcome"],
        "gpu_model": record["gpu_model"],
        "closure_start_frame": k,
        "closure_started_s": c["closure_started_s"],
        "mouth_distance_at_start_m": c["mouth_distance_at_start_m"],
        "mouth_distance_at_start_mm": mm(c["mouth_distance_at_start_m"]),
        "first_loss_frame": loss,
        "first_loss_offset": None if loss is None or k is None else loss - k,
        "scanned_through": c["scanned_through"],
        "cut_phase_k_plus_1": c["cut_phase_k_plus_1"],
        "cut_stop": [c["cut_stop_frame"], c["cut_stop_reason"]],
        "state_at_loss": c.get("state_at_loss"),
        "reason_at_loss": c.get("reason_at_loss"),
        "mask_explanation": c["explanation"],
        "attributable": c["attributable"],
        "recorded_hold_at_loss": c.get("recorded_hold_at_loss"),
        "shadow_stop_frame": c["shadow_stop_frame"],
        "shadow_stop_reason": c["shadow_stop_reason"],
        "last_recorded_frame": last_frame(record),
    }
    if k is None:
        return seen(record, out)
    after = frames.get(k + NO_TRACKING_OFFSET)
    out["state_at_k_plus_3"] = state_of(after) if after else None
    out["frame_k_plus_3_recorded"] = after is not None
    out["tracking_through_k_plus_3"] = loss is None and c["scanned_through"] >= k + NO_TRACKING_OFFSET
    out["rendered_progress_k_plus_1_2"] = [
        (frames.get(k + 1) or {}).get("rendered_progress"),
        (frames.get(k + 2) or {}).get("rendered_progress"),
    ]
    return seen(record, out)


def p4_category(record, view):
    """Where a recorded, unrefused 530 run falls for P4, and why."""
    k = view["closure_start_frame"]
    if k is None:
        if settled(record):
            return "reported", "no closure start"
        return "not_evaluable", f"the recording ends at frame {view['last_recorded_frame']} with the cut still live"
    start = view["mouth_distance_at_start_m"]
    if start is None:
        return "not_evaluable", "no gate mouth distance at closure start"
    if start > MAX_CLOSURE_START_MOUTH_M:
        return "reported", "closure started above 4 mm"
    if view["first_loss_frame"] == k + 1:
        return "predicted", None
    if view["cut_phase_k_plus_1"] is None:
        return "not_evaluable", "frame k+1 is not recorded"
    if view["cut_phase_k_plus_1"] != "closing":
        return "premise", f"the cut left closing at k+1 ({view['cut_stop']}) while tracking; k+2 was rendered open"
    if view["first_loss_frame"] is not None or view["tracking_through_k_plus_3"]:
        return "predicted", None
    return "not_evaluable", "the recording ends inside the closure window before a loss"


def p4_untested(groups):
    if groups["predicted"]:
        return None
    reasons = []
    if groups["premise"]:
        reasons.append(
            "in every 530 run that started closure at <= 4 mm the cut left closing at k+1, so the jaw was never "
            "rendered closing"
        )
    if groups["not_evaluable"]:
        reasons.append(f"the recordings of {', '.join(sorted(groups['not_evaluable']))} end before P4 is settled")
    if groups["excluded"]:
        reasons.append(f"{', '.join(sorted(groups['excluded']))} refused or not recorded")
    if groups["reported"] and reasons:
        reasons.append("the other 530 runs started closure above 4 mm or never started")
    return "; ".join(reasons) or "no 530 run started closure with a gate mouth distance of at most 4 mm"


def score_p4(records):
    groups = {name: {} for name in ("predicted", "reported", "premise", "not_evaluable", "excluded")}
    for record in runs_of(records, 530):
        key = run_key(record)
        if exclusion(record) is not None:
            groups["excluded"][key] = exclusion(record)
            continue
        view = mechanism(record)
        category, why = p4_category(record, view)
        groups[category][key] = view if why is None else {**view, "why": why}
    predicted = groups["predicted"]
    on_time = {k: v["first_loss_offset"] == FIRST_LOSS_OFFSET for k, v in predicted.items()}
    tracking_on = {k: v["state_at_k_plus_3"] == "tracking" for k, v in predicted.items()}
    unexplained = {k: v["first_loss_frame"] is not None and not v["attributable"] for k, v in predicted.items()}

    def recorded_agrees(v):
        hold = v["recorded_hold_at_loss"] or {}
        return hold.get("explained") is True and hold.get("explanation") == v["mask_explanation"]

    def shadow_ok(v):
        return v["shadow_stop_frame"] == v["first_loss_frame"] and v["shadow_stop_reason"] == "vision_invalid"

    def render_ok(v):
        progress = v["rendered_progress_k_plus_1_2"]
        return progress[0] == 0 and progress[1] is not None and progress[1] > 0

    def subset(*names):
        return {k: {name: v[name] for name in names if name in v} for k, v in predicted.items()}

    clauses = [
        clause(
            "the first closure loss comes at closure start + 2",
            subset(
                "closure_start_frame",
                "mouth_distance_at_start_mm",
                "first_loss_frame",
                "first_loss_offset",
                "records_seen_before_scoring",
            ),
            bool(predicted) and all(on_time.values()),
            refutes=not all(on_time.values()),
        ),
        clause(
            "tracking does not continue at closure start + 3",
            subset("state_at_k_plus_3", "frame_k_plus_3_recorded"),
            bool(predicted) and all(v["frame_k_plus_3_recorded"] and not tracking_on[k] for k, v in predicted.items()),
            refutes=any(tracking_on.values()),
        ),
        clause(
            "the first closure loss is mask-attributable (too_few_unmasked_patch_pixels or "
            "flow_failed_only_after_mask_drop)",
            subset("state_at_loss", "reason_at_loss", "mask_explanation", "attributable"),
            bool(predicted) and all(v["attributable"] for v in predicted.values()),
            refutes=any(unexplained.values()),
        ),
        clause(
            "the shadow controller (cut_without_hold) stops at the loss with vision_invalid (not a refutation "
            "condition)",
            subset("first_loss_frame", "shadow_stop_frame", "shadow_stop_reason"),
            bool(predicted) and all(shadow_ok(v) for v in predicted.values()),
        ),
        clause(
            "frame k+1 is rendered with the jaw open and k+2 is the first frame rendered closing (not a refutation "
            "condition)",
            subset("rendered_progress_k_plus_1_2"),
            bool(predicted) and all(render_ok(v) for v in predicted.values()),
        ),
        clause(
            "recording consistency, informational: the live closure_hold record at the loss equals "
            "closure_hold_explanation(recorded measurement, latched_by_jaw=False), equal by construction since "
            "nothing is held before the loss",
            {
                k: {"recorded_hold_at_loss": v["recorded_hold_at_loss"], "consistent": recorded_agrees(v)}
                for k, v in predicted.items()
                if v["first_loss_frame"] is not None
            },
            True,
        ),
        clause(
            "reported, not predicted: 530 runs whose closure started above 4 mm or, in a settled recording, never "
            "started",
            groups["reported"] or {"none": True},
            True,
        ),
    ]
    if groups["premise"]:
        clauses.append(
            vacuous(
                "530 runs that started closure at <= 4 mm but left closing at k+1 for a non-vision reason, so the jaw "
                "was never rendered closing",
                groups["premise"],
                "P4's premise (a first frame rendered closing at k+2) never arose in these runs",
            )
        )
    if groups["not_evaluable"] or groups["excluded"]:
        clauses.append(
            not_judged(
                "530 runs refused, not recorded, or whose recording cannot settle P4",
                {**groups["excluded"], **groups["not_evaluable"]},
            )
        )
    return judged("P4", clauses, p4_untested(groups))


# ---------------------------------------------------------------------------------------------- P5


def p5_category(record, c):
    """Whether P4's loss occurred in a recorded, unrefused 530 run, and why not."""
    k, loss = c["closure_start_frame"], c["first_loss_frame"]
    if k is None:
        if settled(record):
            return "reported", "no closure start"
        return "not_evaluable", f"the recording ends at frame {last_frame(record)} with the cut still live"
    if loss == k + FIRST_LOSS_OFFSET and c["attributable"] and c["cut_phase_k_plus_1"] == "closing":
        return "applies", None
    if loss is not None and loss != k + FIRST_LOSS_OFFSET:
        return "reported", f"the first closure loss is at closure start + {loss - k}"
    if c["cut_phase_k_plus_1"] not in (None, "closing"):
        return "reported", "the cut left closing at k+1, so k+2 was not rendered closing"
    if loss is None and c["scanned_through"] < k + FIRST_LOSS_OFFSET:
        return "not_evaluable", "frame k+1 or k+2 is not recorded"
    if loss is None:
        return "reported", "frame k+2 tracked: no closure loss at closure start + 2"
    return "reported", "the first closure loss is not mask-attributable"


def hold_coverage(record):
    """P5's view of one 530 run where P4's loss occurred (the first closure loss at k+2, mask-attributable)."""
    c = closure_facts(record)
    frames = indexed(record)
    k, loss, deadline = c["closure_start_frame"], c["first_loss_frame"], c["deadline_frame"]
    held, expected = c["held_frames"], list(range(loss, deadline + 1))
    recorded = c["recorded_through_deadline"]
    fps = float(record.get("fps") or PLANNED_FPS)
    reference = frames[k]["tool_pose"]
    motion = [
        pose_change(reference, frames[i]["tool_pose"])
        for i in range(k + 1, deadline + 1)
        if i in frames and frames[i]["tool_pose"] is not None and reference is not None
    ]
    mouth_start = c["mouth_distance_at_start_m"]
    deviations = [
        abs(frames[i]["mouth_distance_m"] - mouth_start)
        for i in held
        if frames[i]["mouth_distance_m"] is not None and mouth_start is not None
    ]
    mislabelled = [
        i
        for i in held
        if frames[i]["vision_source"] != "closure_hold"
        or frames[i]["waived_checks"] != list(CLOSURE_HOLD_WAIVED_CHECKS)
        or frames[i]["held_target_age_s"] is None
        or abs(frames[i]["held_target_age_s"] - (i - k) / fps) > 1e-6
        or frames[i]["held_target_age_s"] > CLOSURE_FRAMES / fps + 1e-9
    ]
    detach, stop = c["detach_frame"], c["cut_stop_frame"]
    no_detach_settled = detach is None and (recorded or (stop is not None and stop <= deadline))
    detach_other = (detach is not None and detach != deadline) or no_detach_settled
    held_stops = [i for i in held if frames[i]["cut_phase"] == "stopped"]
    recorded_motion = [frames[i]["hold"] or {} for i in held]
    return seen(
        record,
        {
            "outcome": record["outcome"],
            "gpu_model": record["gpu_model"],
            "closure_start_frame": k,
            "mouth_distance_at_start_mm": mm(mouth_start),
            "first_loss_frame": loss,
            "held_frames": held,
            "expected_held_frames": expected,
            "recorded_through_deadline": recorded,
            "evaluable": bool(recorded or detach_other or held_stops),
            "covered": bool(recorded) and held == expected,
            "tool_frames_checked": len(motion),
            "tool_max_translation_mm": mm(max(t for t, _ in motion)) if motion else None,
            "tool_max_rotation_deg": max((r for _, r in motion), default=None),
            "tool_within": len(motion) == CLOSURE_FRAMES
            and all(t < TOOL_TRANSLATION_M and r < TOOL_ROTATION_DEG for t, r in motion),
            "recorded_dt_mm_dr_deg_max": [
                max((h.get("dt_mm") or 0.0 for h in recorded_motion), default=None),
                max((h.get("dr_deg") or 0.0 for h in recorded_motion), default=None),
            ],
            "held_mouth_max_deviation_mm": mm(max(deviations)) if deviations else None,
            "mouth_within": bool(held) and len(deviations) == len(held) and max(deviations) <= HELD_MOUTH_TOLERANCE_M,
            "detach_frame": detach,
            "detach_on_time": detach == deadline,
            "detach_other": detach_other,
            "held_stop_frames": held_stops,
            "mislabelled_held_frames": mislabelled,
            "hold_failure_stops": sorted(
                {f["stopped_reason"] for f in record["frames"] if f["stopped_reason"] in HOLD_FAILURE_STOPS}
            ),
            "cut_stop": [stop, c["cut_stop_reason"]],
            "last_recorded_frame": last_frame(record),
        },
    )


def p5_untested(applies, groups):
    if any(v["evaluable"] for v in applies.values()):
        return None
    if applies:
        return "every 530 run where P4's loss occurred ends before the closure deadline with nothing settled"
    reasons = ["no 530 run had P4's loss (a mask-attributable first closure loss at closure start + 2)"]
    if groups["not_evaluable"] or groups["excluded"]:
        reasons.append("530 runs refused, not recorded or not settled remain")
    return "; ".join(reasons)


def score_p5(records):
    applies, groups = {}, {name: {} for name in ("reported", "not_evaluable", "excluded")}
    for record in runs_of(records, 530):
        key = run_key(record)
        if exclusion(record) is not None:
            groups["excluded"][key] = exclusion(record)
            continue
        c = closure_facts(record)
        category, why = p5_category(record, c)
        if category == "applies":
            applies[key] = hold_coverage(record)
        else:
            groups[category][key] = {**closure_brief(record, c), "why": why}

    def subset(*names):
        return {k: {name: v[name] for name in names if name in v} for k, v in applies.items()}

    def every(name):
        return bool(applies) and all(v[name] for v in applies.values())

    late = any(v["detach_other"] for v in applies.values())
    stopped = any(v["held_stop_frames"] for v in applies.values())
    truncated = {k: v for k, v in applies.items() if not v["recorded_through_deadline"]}
    clauses = [
        clause(
            "the hold covers every frame from P4's loss (k+2) to the closure deadline (not a refutation condition)",
            subset(
                "first_loss_frame",
                "held_frames",
                "expected_held_frames",
                "recorded_through_deadline",
                "records_seen_before_scoring",
            ),
            every("covered"),
        ),
        clause(
            "the tool stays within 0.5 mm and 0.25 deg of its closure-start pose over k+1..k+6 (not a refutation "
            "condition)",
            subset(
                "tool_frames_checked", "tool_max_translation_mm", "tool_max_rotation_deg", "recorded_dt_mm_dr_deg_max"
            ),
            every("tool_within"),
        ),
        clause(
            "the gate's mouth distance on held frames equals its closure-start value within 0.05 mm (not a refutation "
            "condition)",
            subset("mouth_distance_at_start_mm", "held_mouth_max_deviation_mm"),
            every("mouth_within"),
        ),
        clause(
            "detachment comes at closure start + 6",
            subset("closure_start_frame", "detach_frame", "cut_stop", "recorded_through_deadline"),
            every("detach_on_time"),
            refutes=late,
        ),
        clause(
            "no held frame stops",
            subset("held_stop_frames", "cut_stop"),
            every("recorded_through_deadline") and not stopped,
            refutes=stopped,
        ),
        clause(
            "every held frame carries the protocol's definition of a held frame: vision_source closure_hold, the four "
            "waived checks and a held target age of (i - k) / 10 s (not a refutation condition)",
            subset("mislabelled_held_frames"),
            bool(applies) and not any(v["mislabelled_held_frames"] for v in applies.values()),
        ),
        clause(
            "informational: the hold's own stop reasons (closure_hold_outside_closing, closure_hold_target_mismatch, "
            "invalid_vision_source); a held frame that triggers one is already a held frame that stops",
            subset("hold_failure_stops") or {"none": True},
            True,
        ),
        clause(
            "reported, not predicted: 530 runs where P4's loss did not occur",
            groups["reported"] or {"none": True},
            True,
        ),
    ]
    if truncated:
        clauses.append(
            not_judged(
                "530 runs with P4's loss whose recording ends before the closure deadline",
                {
                    k: {name: v[name] for name in ("last_recorded_frame", "held_frames", "cut_stop", "evaluable")}
                    for k, v in truncated.items()
                },
            )
        )
    if groups["not_evaluable"] or groups["excluded"]:
        clauses.append(
            not_judged(
                "530 runs refused, not recorded, or whose recording cannot settle whether P4's loss occurred",
                {**groups["excluded"], **groups["not_evaluable"]},
            )
        )
    return judged("P5", clauses, p5_untested(applies, groups))


# ---------------------------------------------------------------------------------------------- P6


def check_rows(records):
    """A graded row must carry the unchanged grader's 17 checks; anything else stops the scorer."""
    for record in records:
        if record["status"] == "graded" and record["checks_total"] != len(GRADER_CHECKS):
            raise EvidenceError(f"{run_key(record)}: graded with {record['checks_total']} checks, not 17")


def score_p6(records):
    rows = runs_of(records, 530)
    observed = {
        run_key(r): seen(
            r,
            {
                "outcome": r["outcome"],
                "status": r["status"],
                "checks": f"{r['checks_passed']}/{r['checks_total']}",
                "failed_checks": r["failed_checks"],
                "stop_reason": r["stop_reason"],
                "evidence_stop_frame": r["stop_frame"],
                "gpu_model": r["gpu_model"],
                "configuration_matches": r["configuration_matches"],
                "refused": r["refused_reasons"] if r["refused"] else False,
                "configuration_problems_in_report": r["configuration_problems"],
            },
        )
        for r in rows
    }
    passes = [run_key(r) for r in rows if r["outcome"] == "pass" and not r["refused"]]
    return judged(
        "P6",
        [
            clause(
                "at least one of the three 530 runs passes all 17 checks of the unchanged grader",
                {
                    "label": LABEL,
                    "passes": passes,
                    "refused_passes_not_counted": [run_key(r) for r in rows if r["outcome"] == "pass" and r["refused"]],
                    "runs": observed,
                    "missing_runs": max(0, EXPECTED_RUNS[530] - len(rows)),
                    "grader_note": GRADER_NOTE,
                },
                bool(passes),
                refutes=not passes,
            )
        ],
    )


# ---------------------------------------------------------------------------------------------- P7


def drift_status(rows, k, anchor):
    """Shifts of the baseline target from ``anchor`` over k+1..k+6 and P7's status (see interpretations)."""
    shifts, gaps, missing, latched = {}, [], [], None
    for index in range(k + 1, k + CLOSURE_FRAMES + 1):
        row = rows.get(index)
        if latched is not None:
            break
        if row is None:
            missing.append(index)
        elif row["state"] == "tracking" and row["target"] is not None:
            shifts[index] = float(np.linalg.norm(np.asarray(row["target"], dtype=float) - anchor))
        elif row["state"] in GAP_STATES or row["state"] == "tracking":
            gaps.append({"frame": index, "state": row["state"], "reason": row["reason"]})
        else:
            latched = {"frame": index, "state": row["state"], "reason": row["reason"]}
    deadline = k + CLOSURE_FRAMES
    window = {i: v for i, v in shifts.items() if i < deadline}
    moved_at = next((i for i in sorted(window) if window[i] >= DRIFT_M), None)
    latched_in_window = latched is not None and latched["frame"] < deadline
    missing_in_window = [i for i in missing if i < deadline]
    if moved_at is not None and (latched is None or moved_at < latched["frame"]):
        status = "moved"
    elif latched_in_window:
        status = "indeterminate"
    elif missing_in_window:
        status = "untested"
    elif len(window) < CLOSURE_FRAMES - 1:
        status = "indeterminate"  # a gap in k+1..k+5: the target was not measured there, so it cannot have stayed
    else:
        status = "stayed_within"  # measured on every frame k+1..k+5 and never 3 mm from the anchor
    return status, shifts, gaps, missing, latched


def drift(record):
    """P7's view of one 530 run that held: the baseline target's shift from its own closure-start value."""
    c = closure_facts(record)
    k = c["closure_start_frame"]
    baseline = record.get("baseline") or {}
    out = seen(
        record,
        {
            "outcome": record["outcome"],
            "gpu_model": record["gpu_model"],
            "closure_start_frame": k,
            "held_frames": c["held_frames"],
            "live_initial_tracker": (record.get("report") or {}).get("initial_tracker"),
        },
    )
    if not baseline or baseline.get("error"):
        out.update(status="untested", why=f"baseline replay failed: {baseline.get('error') or 'not run'}")
        return out
    out.update(
        replay_module=baseline.get("replay_module"),
        replay_flags=baseline.get("flags"),
        replay_initialization=baseline.get("initialization"),
        replay_closure_start_frame=baseline.get("recorded_closure_start_frame"),
        replay_reproduces_recording=baseline.get("reproduces_recording_through_recorded_stop"),
    )
    rows = {row["index"]: row for row in baseline.get("rows") or []}
    at_k = rows.get(k) if k is not None else None
    if at_k is None or at_k["state"] != "tracking" or at_k["target"] is None:
        state = None if at_k is None else at_k["state"]
        out.update(status="untested", why=f"the baseline is not tracking at closure start {k} (state {state})")
        return out
    anchor = np.asarray(at_k["target"], dtype=float)
    status, shifts, gaps, missing, latched = drift_status(rows, k, anchor)
    live = (indexed(record)[k]["measurement"] or {}).get("target_position_world_m")
    live_status, live_shifts = None, {}
    if live is not None:
        live_status, live_shifts, *_ = drift_status(rows, k, np.asarray(live, dtype=float))
    window = [v for i, v in shifts.items() if i < k + CLOSURE_FRAMES]
    out.update(
        status=status,
        shift_mm_from_baseline_target_at_k={str(i): mm(v) for i, v in shifts.items()},
        shift_mm_from_live_closure_reference={str(i): mm(v) for i, v in live_shifts.items()},
        anchor_difference_mm=None if live is None else mm(np.linalg.norm(anchor - np.asarray(live, dtype=float))),
        status_against_live_closure_reference=live_status,
        anchors_disagree=live_status is not None and live_status != status,
        max_shift_before_deadline_mm=mm(max(window)) if window else None,
        shift_at_deadline_mm=mm(shifts.get(k + CLOSURE_FRAMES)),
        gap_frames=gaps,
        missing_frames=missing,
        first_latched_loss=latched,
    )
    if status == "indeterminate" and latched is not None and latched["frame"] < k + CLOSURE_FRAMES:
        out["why"] = f"the baseline latched {latched['state']} at frame {latched['frame']} before moving 3 mm"
    elif status == "indeterminate":
        unmeasured = [gap["frame"] for gap in gaps if gap["frame"] < k + CLOSURE_FRAMES]
        out["why"] = f"frames {unmeasured} of k+1..k+5 measured no target and no tracking frame moved 3 mm"
    elif status == "untested":
        out["why"] = f"frames {missing} of the window are missing before either outcome was established"
    return out


def score_p7(records):
    views, reported, excluded = {}, {}, {}
    for record in runs_of(records, 530):
        key = run_key(record)
        if exclusion(record) is not None:
            excluded[key] = exclusion(record)
        elif closure_facts(record)["held_frames"]:
            views[key] = drift(record)
        elif settled(record):
            reported[key] = {**closure_brief(record), "why": "never held"}
        else:
            excluded[key] = {**closure_brief(record), "why": "never held before its recording ended with the cut live"}
    tested = {k: v for k, v in views.items() if v["status"] != "untested"}
    untested_runs = {k: v for k, v in views.items() if v["status"] == "untested"}
    stayed = [k for k, v in tested.items() if v["status"] == "stayed_within"]
    decided = [k for k, v in tested.items() if v["status"] in ("moved", "stayed_within")]
    clauses = [
        clause(
            "the baseline tracker moves its measured target at least 3 mm from the closure-start target before the "
            "deadline (k+1..k+5)",
            tested,
            bool(tested) and all(v["status"] == "moved" for v in tested.values()),
            refutes=bool(stayed),
        ),
        clause("reported, not predicted: 530 runs that never held", reported or {"none": True}, True),
    ]
    if untested_runs:
        clauses.append(
            vacuous("530 runs that held but whose baseline could not be judged", untested_runs, "see each run's why")
        )
    if excluded:
        clauses.append(not_judged("530 runs refused, not recorded, or not settled", excluded))
    if not views:
        untested = "no 530 run held" + ("; 530 runs refused, not recorded or not settled remain" if excluded else "")
    elif not tested:
        untested = "no 530 run that held has a baseline that could be judged"
    elif not decided:
        untested = (
            "no 530 run that held has a baseline that moved 3 mm or was measured within 3 mm on every frame of "
            "k+1..k+5 (each is indeterminate: see each run's why)"
        )
    else:
        untested = None
    return judged("P7", clauses, untested)


# ---------------------------------------------------------------------------------------------- P8


def static_branch(record):
    """P8's view of one 530 run that detached: the piece and the true mouth-to-spur distance over [k, d]."""
    c = closure_facts(record)
    k, d = c["closure_start_frame"], c["detach_frame"]
    frames = indexed(record)
    out = seen(record, {"outcome": record["outcome"], "gpu_model": record["gpu_model"], "window": [k, d]})
    window = [frames.get(i) for i in range(k, d + 1)] if k is not None else []
    if not window or any(
        f is None or f["piece_pose"] is None or f["tool_pose"] is None or f["mouth_offset"] is None for f in window
    ):
        out.update(status="untested", why="the closure window lacks a frame, a piece pose, a tool pose or a mouth")
        return out
    piece = np.asarray([f["piece_pose"][:3] for f in window], dtype=float)
    mouths = np.asarray(
        [
            tool_mouth_geometry(f["tool_pose"], f["mouth_offset"], f["closing_axis"] or (1.0, 0.0, 0.0))[0]
            for f in window
        ]
    )
    motion = np.linalg.norm(piece - piece[0], axis=1)
    distance = np.linalg.norm(mouths - piece, axis=1)
    change = np.abs(distance - distance[0])
    out.update(
        status="tested",
        piece_motion_max_mm=mm(motion.max()),
        piece_rotation_max_deg=max(
            rotation_angle_deg(window[0]["piece_pose"][3:], f["piece_pose"][3:]) for f in window
        ),
        true_distance_at_start_mm=mm(distance[0]),
        true_distance_at_detach_mm=mm(distance[-1]),
        true_distance_change_max_mm=mm(change.max()),
        true_distance_change_net_mm=mm(distance[-1] - distance[0]),
        piece_static=bool(motion.max() < PIECE_MOTION_M),
        distance_static=bool(change.max() < DISTANCE_CHANGE_M),
    )
    return out


def score_p8(records):
    views, reported, excluded = {}, {}, {}
    for record in runs_of(records, 530):
        key = run_key(record)
        if exclusion(record) is not None:
            excluded[key] = exclusion(record)
        elif closure_facts(record)["detach_frame"] is not None:
            views[key] = static_branch(record)
        elif settled(record):
            reported[key] = {**closure_brief(record), "why": "never detached"}
        else:
            excluded[key] = {**closure_brief(record), "why": "no detach before its recording ended with the cut live"}
    tested = {k: v for k, v in views.items() if v["status"] == "tested"}

    def subset(*names):
        return {k: {name: v[name] for name in names if name in v} for k, v in tested.items()}

    piece_moved = any(not v["piece_static"] for v in tested.values())
    distance_moved = any(not v["distance_static"] for v in tested.values())
    clauses = [
        clause(
            "the selected piece moves less than 0.1 mm between closure start and detach",
            subset("window", "piece_motion_max_mm", "piece_rotation_max_deg", "records_seen_before_scoring"),
            bool(tested) and not piece_moved,
            refutes=piece_moved,
        ),
        clause(
            "the true mouth-to-spur distance changes by less than 0.1 mm between closure start and detach",
            subset(
                "window",
                "true_distance_at_start_mm",
                "true_distance_at_detach_mm",
                "true_distance_change_max_mm",
                "true_distance_change_net_mm",
            ),
            bool(tested) and not distance_moved,
            refutes=distance_moved,
        ),
        clause("reported, not predicted: 530 runs that never detached", reported or {"none": True}, True),
    ]
    untested_runs = {k: v for k, v in views.items() if v["status"] != "tested"}
    if untested_runs:
        clauses.append(vacuous("530 runs that detached but cannot be measured", untested_runs, "see each run's why"))
    if excluded:
        clauses.append(not_judged("530 runs refused, not recorded, or not settled", excluded))
    remaining = "; 530 runs refused, not recorded or not settled remain" if excluded else ""
    if tested:
        untested = None
    elif views:
        untested = "no detached 530 run could be measured"
    else:
        untested = "no 530 run detached" + remaining
    return judged("P8", clauses, untested)


# ---------------------------------------------------------------------------------------------- document


def score(records):
    check_rows(records)
    return [
        score_p1(records),
        score_p2(records),
        score_p3(records),
        score_p4(records),
        score_p5(records),
        score_p6(records),
        score_p7(records),
        score_p8(records),
    ]


def run_summary(record):
    """One row of the runs table; every planned run appears, whatever its outcome."""
    c = closure_facts(record)
    return {
        "batch": record["batch"],
        "run_index": record.get("run_index"),
        "run_directory": record["run_directory"],
        "target": record["target"],
        "label": LABEL,
        "outcome": record["outcome"],
        "status": record["status"],
        "checks": f"{record['checks_passed']}/{record['checks_total']}",
        "failed_checks": record["failed_checks"],
        "stop_reason": record["stop_reason"],
        "evidence_stop_frame": record["stop_frame"],
        "configuration_matches": record["configuration_matches"],
        "refused": record["refused"],
        "refused_reasons": record["refused_reasons"],
        "configuration_problems_in_report": record["configuration_problems"],
        "not_recorded_reason": record.get("not_recorded_reason"),
        "frames_recorded": None if record["frames"] is None else len(record["frames"]),
        "frames_planned": PLANNED_FRAMES,
        "recording_complete": recording_complete(record),
        "cut_stop_decision_frame": c["cut_stop_frame"],
        "cut_stop_reason": c["cut_stop_reason"],
        "closure_start_frame": c["closure_start_frame"],
        "detach_frame": c["detach_frame"],
        "held_frames": c["held_frames"],
        "shadow_stop": [c["shadow_stop_frame"], c["shadow_stop_reason"]],
        "node": record["node"],
        "gpu_model": record["gpu_model"],
        "seen_before_scoring": SEEN_BEFORE_SCORING.get((record["batch"], record.get("run_index"))),
        "report": record["report"],
    }


def build_document(records, plans, scored_code, inputs, code_revision, code_tree_dirty):
    """The output document: provenance, the exposure, the interpretations, every run and every verdict."""
    return {
        "schema_version": 1,
        "scope": SCOPE,
        "label": LABEL,
        "protocol": PROTOCOL,
        "construction_exposure": list(CONSTRUCTION_EXPOSURE),
        "code_revision": code_revision,
        "code_tree_dirty": code_tree_dirty,
        "scored_code": scored_code,
        "plans": plans,
        "inputs_sha256": inputs,
        "grader_checks": list(GRADER_CHECKS),
        "interpretations": list(INTERPRETATIONS),
        "runs": [run_summary(record) for record in records],
        "predictions": score(records),
    }


def json_safe(value):
    """Strict JSON: numpy values become Python and non-finite floats become null."""
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT, help="Where docs/evidence and artifacts/ are read")
    parser.add_argument("--output", type=Path, help="New JSON file; an existing file is never overwritten")
    args = parser.parse_args(argv)
    if args.output is not None and args.output.exists():
        parser.error(f"Refusing to overwrite existing output: {args.output}")
    try:
        validated = validated_evidence(args.root)
        scored_code = frozen_code(args.root, validated)
    except (EvidenceError, FrozenCodeError) as error:
        parser.error(str(error))
    cv2.setNumThreads(1)
    records = load(args.root, validated)
    head = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"]).decode().strip()
    dirty = bool(subprocess.check_output(["git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no"]))
    inputs = {}
    for path, _, batch_dir, _, _ in validated:
        inputs[str(path.relative_to(args.root))] = sha256(path)
        inputs[f"{batch_dir}/plan.json"] = sha256(batch_path(args.root, batch_dir) / "plan.json")
    for record in records:
        inputs.update(record["inputs"])
    document = build_document(records, plan_summary(validated), scored_code, inputs, head, dirty)
    serialized = json.dumps(json_safe(document), indent=2, allow_nan=False) + "\n"
    if args.output is not None:
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(serialized)
    print(json.dumps({p["id"]: p["verdict"] for p in document["predictions"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
