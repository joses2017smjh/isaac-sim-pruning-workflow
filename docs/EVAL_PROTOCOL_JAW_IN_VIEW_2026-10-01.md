# Jaw in the camera's view: self-mask and closure hold — protocol, October 1, 2026

Registered **before** submission. The
[known-map approach](EVAL_PROTOCOL_PLANNED_APPROACH_2026-09-27.md#result--september-30-2026)
let contact target 530 reach alignment at 0 N in 3 of 3 runs. No planned
target passed, and both remaining stops are the robot's own jaw in its
camera's view:
- **530.** During closure the tracker follows the closing jaw. The tool and
  spur are still, yet the measured spur drifts from about 4 mm to 10 mm until
  the 8 mm gate trips.
- **19444.** On the final leg the open jaw passes in front of the tracked
  patch.

An offline study replayed every recorded run (129 runs, 10,121 tracker
updates) with a jaw self-mask and a closure hold. Two verifiers reproduced its
numbers independently, and an adversarial review corrected it. This experiment
runs the surviving design live.

## The arm: `planned_pose_jaw_hold`

It is the `planned_pose` strategy plus two default-off controller flags
(`PRUNING_JAW_SELF_MASK`, `PRUNING_CLOSURE_HOLD`). Both are recorded in every
report, and `configuration_matches` refuses a capture without them.

**Jaw self-mask (perception).** It is computed from what the controller already
knows: the rendered tool pose, its own closing axis, the branch radius and the
closure progress that was rendered (the value before the update). The jaw is
the two visual-surrogate boxes, half extents 3 × 12.5 × 15 mm at 70 mm down
the tool axis. The mask is every pixel centre within 2.0 px (signed distance)
of either box's projected hull. In the tracker:
- (a) a feature that lands in the mask, or left it, cannot vote on the flow;
- (b) the mask is excluded from feature detection;
- (c) the 0.35 appearance check compares only patch elements unmasked in both
  frames;
- (d) fewer than 140 of 169 kept elements gives a new explicit stop state,
  `jaw_mask_occluded`, checked before the 0.35 threshold.

No tracker threshold moves. N = 140 was fixed by a rule stated before the
outcomes were known.

**Closure hold (a change to what the cut gate certifies).** During `closing`
only, a frame is held when all three of these are true:
- the tool is within 0.5 mm and 0.25° of its pose at the closure-start step;
- the measurement is explained by the mask: `jaw_mask_occluded`, a flow failure
  that only the mask's feature drop caused, or the latched loss that follows
  one of these within this closure;
- the cut phase is `closing`.

A held frame carries the closure-reference target, `vision_source =
closure_hold` and the held target's own timestamp. On held frames the gate
waives four checks: vision-invalid, freshness (the held target is up to 0.6 s
old against a 0.25 s limit), timestamp regression and frame reuse. The
certificate lists them on every held frame.

Still binding on held frames:
- mouth distance (frozen target against the live mouth) ≤ 8 mm;
- perpendicularity ≤ 15°;
- mouth speed;
- hazard contact;
- the 0.6 s closure clock.

A hold outside closing, or one whose target differs from the closure
reference, stops the run. Every other loss during closure stops as today.

A second, shadow cut controller receives the honest (unheld) observation on
every frame and is recorded as `cut_without_hold`. This is the mask-only
verdict, so each run also shows what the unchanged gate would have done.

**User confirmation.** The user approved the experiment's 270 GPU-minutes on
September 30. On October 1, after the waiver was put to them explicitly, they
chose this arm with the waiver and with controls.

**Label.** Every result is *known-map plan; jaw self-mask; closure hold with
freshness and frame-reuse checks waived on held frames*. It is reported apart
from every unchanged-gate result and never pooled.

**What it does not change.** The approach and align logic, the grader's 17
checks, the scene (the jaw stays visible and still casts shadows) and the
camera mount. It uses no ground truth. It cannot help 19444, which stops on the
approach, where the hold never applies, or the low-sun shadow stops.

## Runs

The planned-pose register is split per batch, because a plan may not repeat a
target. Three `afterany`-chained batches, each `0-1%1`:

| Batch | Register | Targets |
|---|---|---|
| A | [eval_targets_jaw_hold_a](evidence/eval_targets_jaw_hold_a_2026-09-30.json) | 530, 19444 |
| B | [eval_targets_jaw_hold_b](evidence/eval_targets_jaw_hold_b_2026-09-30.json) | 530, 14944 |
| C | [eval_targets_jaw_hold_c](evidence/eval_targets_jaw_hold_c_2026-09-30.json) | 530, 15004 |

That is 530 three times, 19444 once, and controls 14944 and 15004 once each.
All runs use source light and 200 frames, on `gpu,ampere` with the constraint
`a40|rtx8000` and 45-minute tasks. Each run is labelled with its GPU model.

## Pre-registered predictions

- **P0 (pre-submission gate, CPU).** With both flags off, the repository code
  replays all 129 recorded runs exactly: state, reason and pixel exact,
  correlation within 1e-6, cut phase and stop exact. With the mask on, it
  reproduces the study's per-frame outputs, including 19444 r3 frame 67 at 139
  kept elements. The hold applies only in the three 530 runs, at frames 76–79
  up to the recorded stop. It also applies at the deadline frame 80, after the
  recorded stop; that frame is not evaluable. *Any mismatch blocks submission.*
  Evidence:
  [P0](evidence/jaw_in_view_p0_replay_2026-10-01.json), regenerated from the
  registration commit. Rechecks by the hardened checker:
  [October 1](evidence/jaw_in_view_p0_replay_recheck_2026-10-01.json) and
  [October 2](evidence/jaw_in_view_p0_replay_recheck_2026-10-02.json). The
  October 2 recheck compares sources with the registration commit's own
  files, and adds the named morning-15004 check; both passed.
- **P1 (mask fidelity).** On every frame of all six runs, the recorded mask
  pixel count equals an offline reconstruction from the recorded pose,
  progress and camera. No predicted jaw pixel shows recorded depth more than
  1 mm beyond the analytic box depth. *Refuted by* any mismatch.
- **P2 (controls, no harm).** 14944 and 15004 pass 17/17. The mask removes no
  patch element before detach, the tracker reports `tracking` on every frame
  before detach, and the hold never applies. *Refuted if* either fails a check,
  loses tracking before detach, masks a patch element before detach, or holds.
- **P3 (19444, explicit stop before the jaw is measured).** 19444 stops on the
  final leg with a mask-attributable state (`jaw_mask_occluded` expected at
  frames 67–68). No frame reports `tracking` with the tracked pixel inside the
  jaw silhouette. *Refuted if* such a frame occurs, or if, after the jaw first
  touches the patch, it stops on a loss the mask does not explain or continues
  to align. A stop before the jaw reaches the patch leaves P3 untested.
- **P4 (530 mechanism).** This is predicted only for runs whose closure starts
  with a gate mouth distance of at most 4 mm (recorded 2.69–3.05 mm). The first
  closure loss comes at closure start + 2, the first frame rendered with the
  jaw closing, and is mask-attributable. The shadow controller
  (`cut_without_hold`) stops there with `vision_invalid`. *Refuted if*, in such
  a run, tracking continues at closure start + 3, or the first loss comes at
  another frame or is not mask-attributable. Runs that start closure above
  4 mm are reported, not predicted.
- **P5 (530 hold to detachment).** In every 530 run where P4's loss occurs, the
  hold covers every frame from it to the closure deadline. The tool stays
  within 0.5 mm and 0.25°. The gate's mouth distance on held frames equals its
  closure-start value within 0.05 mm, and detachment comes at closure start + 6.
  *Refuted if* any held frame stops or detachment comes at another frame.
- **P6 (530 outcome under the label).** At least one of the three 530 runs
  passes all 17 checks of the unchanged grader. *Refuted if* none does. Every
  failed check is reported per run.
- **P7 (the hold covered a real drift).** In each 530 run that holds, the
  baseline tracker replayed on that run's own live frames moves its measured
  target at least 3 mm from the closure-start target before the deadline.
  *Refuted if* it stays within 3 mm; the hold was then unnecessary in that run.
- **P8 (the static-branch premise, evaluation only).** In every 530 run that
  detaches, the selected piece moves less than 0.1 mm, and the true
  mouth-to-spur distance changes by less than 0.1 mm, between closure start and
  detach. *Refuted otherwise.*

## What this cannot show

- **A simulator self-model.** The jaw is the two-box visual surrogate, with
  exact poses and commanded closure. A real jaw's shape, calibration error and
  latency would need their own margin, and the tool body is not masked.
- **The static-branch premise holds only in simulation.** The hold assumes the
  branch does not move while the jaw closes. In simulation the spur is
  kinematic, so P8 checks the premise there and nowhere else. A real branch can
  be pushed.
- **No generalization test.** The hold was designed on the same three 530
  recordings it runs on, so the live runs confirm it in closed loop but do not
  test generalization. Whether the hold lets 530 detach and pass is exactly
  what offline replay could not show: frame 80 was rendered under the recorded
  stop.
- **Render noise.** RGB renders are not repeatable, so a class change near a
  floor counts only if it repeats. 19444 r3 sat one element from N.

## Cost

6 runs of 200 frames, 45 minutes reserved each, one task at a time: **270
GPU-minutes reserved** (approved). About 2.7 GB of captures on the hpc-share.
