# Closure hold on unseen contact targets — protocol, October 4, 2026

Registered **before** submission. The jaw self-mask and closure hold carried
contact target 530 through closure in 3 of 3 runs
([jaw in view](EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md#result--october-2-2026)).
The hold was designed on 530's own recordings, so that result confirmed it in
closed loop without testing it elsewhere. This experiment runs the same arm,
unchanged, on three targets it has never seen.

**User confirmation.** On October 4 the user chose this target set and
approved **495 GPU-minutes**. The runs are submitted only after the P0 gate
(H0) and a scorer reviewed blind have passed.

## Targets

The rebuilt known-map planner (`tools/known_map_planner.py`) passed its
acceptance test and found 34 qualifying targets
([scope, steps 1–2](SCOPE_CLOSURE_HOLD_GENERALIZATION_2026-10-02.md#steps-1-and-2--october-4-2026)).
Each meets three conditions:
- **(a)** a clear re-oriented path;
- **(b)** the open jaw leaves the patch on the approach;
- **(c)** the closing jaw covers it at the first frame rendered closing.

For 31 of them, a checked plan also exists that keeps the jaw out of view, so
the hold would not be needed there. The three targets below have no such
plan among the orientations checked, so a pass on them means the hold was
needed.

| Target | Plan (planner) | Standoff | Approach kept (min, of 169) | Closure kept at progress 1/6 | Register |
|---|---|---|---|---|---|
| tree0 3721 | 38.3° re-orientation | 60 mm | 162 | 125 | [a](evidence/eval_targets_hold_gen_a_2026-10-04.json) |
| tree1 36196 | 38.3° | 60 mm | 148 | 77 | a |
| tree1 18143 | 38.3° | 100 mm | 163 | 130 | [18143](evidence/eval_targets_hold_gen_18143_2026-10-04.json) |

At a 60 mm standoff, 18143's planned approach touches a neighbouring spur. At
100 mm it is clear, with 2.1 mm at its closest. All three planned
orientations are the same world orientation: a tool axis about 5° from
home's, rolled 45° about it.

## The arm

The arm is the jaw-in-view arm, unchanged: `planned_pose_jaw_hold` with the
jaw self-mask and the closure hold, under two names that register it to this
protocol:
- `planned_pose_jaw_hold_gen`;
- `planned_pose_jaw_hold_gen_s100`, which differs only in its 100 mm standoff.

A test pins that the standoff is the only difference. The mask, the hold, the
four waived checks on held frames, every threshold, the grader and the scene
are as in the jaw-in-view protocol.

**Label.** Every result is *known-map plan; jaw self-mask; closure hold with
freshness and frame-reuse checks waived on held frames*. It is reported apart
from every unchanged-gate result and never pooled.

## Runs

Seven `afterany`-chained batches, one task at a time, source light, 200
frames, on `gpu,ampere` with the constraint `a40|rtx8000`, in 45-minute tasks.
Each run is labelled with its GPU model.

| Batch | Targets | Strategy |
|---|---|---|
| `hold-gen-a-r1-20261004`, `-r2`, `-r3` | 3721, 36196 | `planned_pose_jaw_hold_gen` |
| `hold-gen-18143-r1-20261004`, `-r2`, `-r3` | 18143 | `planned_pose_jaw_hold_gen_s100` |
| `hold-gen-ctl-20261004` | 530, 14944 ([jaw_hold_b](evidence/eval_targets_jaw_hold_b_2026-09-30.json)) | `planned_pose_jaw_hold_gen` |

## Pre-registered predictions

Frame conventions are those of the jaw-in-view protocol:
- 0-based `frames.json` indices;
- closure start k is the first frame whose cut phase is `closing`;
- the deadline is k + 6;
- a stop is decided on the first frame whose cut phase is `stopped`.

"A selected target" means 3721, 36196 or 18143. A run is scored only if it
was graded, its capture completed, and it ran the arm with its registered
constants; any other run stays in every table and decides nothing.

- **H0 (pre-submission gate, CPU).** `tools/check_hold_gen_p0.py` passes:
  - **A.** With every arm off, the controller reproduces the 141 published
    recordings.
  - **B.** With the mask and the hold, it reproduces the 6 live jaw-in-view
    recordings, holds included.
  - **C.** The registers equal the committed planner evidence.
  - **D.** The strategies differ from the jaw-in-view arm only in 18143's
    standoff.

  *Any mismatch blocks submission.*
- **H1 (the planned approach reaches closure).** Each selected target starts
  closure in at least 2 of its 3 runs.
  - *Refuted if* a target with at least 2 scored runs starts closure in none
    of them.
  - Every stop before closure is reported with its frame, cut reason and
    tracker state.
- **H2 (mechanism, as P4).** This is predicted for selected-target runs whose
  closure starts with a gate mouth distance of at most 4 mm. The first
  closure loss comes at k + 2, the first frame rendered closing, and is
  mask-attributable. The shadow controller (`cut_without_hold`) stops there.
  - *Refuted if*, in such a run, tracking continues at k + 3, or the first
    loss comes at another frame or is not mask-attributable.
  - Runs that start closure above 4 mm are reported, not predicted.
- **H3 (hold to detachment, as P5).** In every selected-target run where H2's
  loss occurs:
  - the hold covers every frame from it to the deadline;
  - the tool stays within 0.5 mm and 0.25°;
  - the gate's mouth distance on held frames equals its closure-start value
    within 0.05 mm;
  - detachment comes at k + 6.

  *Refuted if* any held frame stops, or detachment comes at another frame.
- **H4 (outcome).** Each selected target passes all 17 checks of the
  unchanged grader in at least 1 of its runs.
  - *Refuted if* a target with at least one scored run that started closure
    has no pass.
  - A target none of whose scored runs starts closure leaves H4 untested for
    that target (H1 covers it).
  - Every failed check is reported per run.
- **H5 (static-branch premise, as P8).** In every selected-target run that
  detaches, the selected piece moves less than 0.1 mm, and the true
  mouth-to-spur distance changes by less than 0.1 mm, between closure start
  and detach. *Refuted otherwise.*
- **H6 (mask fidelity, P1 as amended October 2).** On every frame of all 11
  runs:
  - the recorded mask pixel count equals an offline reconstruction from the
    recorded pose, progress and camera;
  - no predicted jaw pixel shows recorded depth more than 1 mm beyond the
    analytic box depth.

  A predicted jaw pixel is one whose centre ray hits either box and whose
  centre lies at least 0.01 px inside the undilated silhouette (the
  [October 2 amendment](EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md#registered-for-future-runs--october-2-2026)).
  *Refuted by* any mismatch. The literal count, without the 0.01 px band, is
  reported.
- **H7 (controls).**
  - 530 passes 17/17, its hold covers k + 2 to k + 6, and it detaches at
    k + 6.
  - 14944 passes 17/17, the hold never applies, and the mask removes no patch
    element before detach.
  - *Refuted if* either control fails a check, 14944 holds or has a patch
    element masked before detach, or a held frame of 530 stops.

Grading uses only `tools/validate_vision_sequence.py`, through
`tools/aggregate_eval.py --protocol` per batch, and every run stays in every
table. H1–H7 are scored by a committed scorer, reviewed blind and committed
before submission. The runner exits 0 only on a 17/17 pass with a matching
configuration, so Slurm states reveal outcomes. Because the scorer is
committed before submission, they are recorded as they arrive.

## What this cannot show

- **Three targets, one orientation family.** All three plans share one world
  orientation (a 45° roll near home), on two trees in source light.
- **A simulator self-model.** The jaw is the two-box surrogate, with exact
  poses and commanded closure, and the spur is kinematic: H5 checks the
  static-branch premise only in simulation.
- **The planner's predictions are not tested independently.** H1 and H2 test
  them, but a failure there is a finding about the planner as much as about
  the hold.
- **Render noise.** RGB renders are not repeatable run to run.

## Cost

11 runs of 200 frames, 45 minutes reserved each, one task at a time: **495
GPU-minutes reserved** (approved October 4). About 5 GB of captures on the
hpc-share, whose project quota is above the 1.5 TiB soft limit and below the
2 TiB hard limit.

## Result — October 5, 2026

All 11 runs recorded 200 frames on an A40 and used 155 of the 495 reserved
GPU-minutes. `tools/validate_vision_sequence.py` graded them, through
`aggregate_eval.py` (CPU job `21552057`). The scorer
`tools/score_hold_generalization.py` ran from a clean clone at `81bcfc8` (CPU
job `21552075`). It, and every scoring module it imports, is byte-identical
to the copy frozen in every batch's plan.

Evidence: [verdicts](evidence/hold_gen_verdicts_2026-10-05.json) and the
per-batch [grades](evidence/hold_gen_2026-10-04/).

| Run | Grader's outcome | Where and why it stopped |
|---|---|---|
| tree0 3721, ×3 | 8/17 each | frame 0: initialization rejected at home. The seed pixel's depth is 0.347 m against the 0.393 m expected (`selected_branch_occluded_or_wrong_surface`) |
| tree1 36196, ×3 | 9/17 each | frame 69, on the final leg: `jaw_mask_occluded`, the open jaw covering the patch |
| tree1 18143, ×3 | 9/17 each | the re-orientation leg: tracking lost at frame 0 (optical flow, 2 runs) and frame 23 (depth rejected, 1 run) |
| 530 (control) | pass 17/17 | none: held 76–80, detached at 80 |
| 14944 (control) | pass 17/17 | none: never held |

| | Registered | Measured | Verdict |
|---|---|---|---|
| **H0** | the P0 gate | [passed](evidence/hold_gen_p0_2026-10-04.json), checks A–D | passed |
| **H1** | each target starts closure in at least 2 of 3 runs | 0 of 3 for every target | **refuted** |
| **H2** | first closure loss at k + 2, mask-attributable | no selected run started closure | untested |
| **H3** | hold to detachment | no run had H2's loss | untested |
| **H4** | each target passes at least once | no selected run passed or started closure | untested |
| **H5** | static-branch premise | no selected run detached | untested |
| **H6** | mask fidelity (P1 as amended) | counts exact on every frame; no violation, with or without the 0.01 px band | supported |
| **H7** | controls | 530 passed with the hold over 76–80 and detach at 80; 14944 passed and never held | supported |

**Reading.** Every result carries the label *known-map plan; jaw self-mask;
closure hold with freshness and frame-reuse checks waived on held frames*.
- **The hold was never exercised on an unseen target.** None of the 9 runs
  reached closure, so H2–H5 are untested. This experiment says nothing either
  way about whether the hold generalizes.
- **Each target failed on a planner prediction about perception, in all 3 of
  its runs.** The planner's geometry passed its acceptance test (contacts,
  time-of-flight and refusals). Its three perception predictions did not
  transfer to the live runs:
  - **3721, visibility at home.** The planner's line-of-sight check called the
    seed visible, but the rendered depth shows a surface 47 mm in front of the
    spur, and the tracker refuses to start. This happens at home, so it would
    stop any approach to this target.
  - **36196, the jaw on the approach.** Condition (b) predicted at least 148
    of 169 patch elements kept on the approach, the smallest margin of the
    three. Live, the open jaw covered the patch at frame 69, as it did 19444's.
  - **18143, tracking through the re-orientation.** The tracker initialized,
    then lost the target during the 38° re-orientation at the 100 mm standoff.
    The scope listed this as untested.
- **The controls confirm the code.** At this commit 530 passed with the hold
  exactly as on October 2, and 14944 passed with no hold. The mask matched its
  reconstruction on every frame of all 11 runs.

**Seen before scoring.** The Slurm states (9 FAILED, 2 COMPLETED) were seen
before the scorer ran. They reveal H4's and H7's pass clauses. The scorer was
committed before submission and was not amended, so nothing seen could shape
it.

**What would be needed to test the hold on another target.** A selection that
checks perception on the renderer, not by ray casting:
- the seed's rendered depth at home;
- the jaw mask on every approach frame, with a margin;
- tracking through the re-orientation (a replay or a short render).

That is a new proposal; nothing has been started.
