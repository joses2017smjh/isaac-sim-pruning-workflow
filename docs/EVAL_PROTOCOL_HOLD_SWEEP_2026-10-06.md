# Closure-hold sweep over the planner's targets — protocol, October 6, 2026

Registered **before** submission. The jaw self-mask and closure hold carried
contact target 530 through closure in 3 of 3 runs
([jaw in view](EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md#result--october-2-2026)).
On October 4 the same arm ran on three unseen targets, and none of the 9 runs
reached closure. Each stopped on a planner prediction about perception
([result](EVAL_PROTOCOL_HOLD_GENERALIZATION_2026-10-04.md#result--october-5-2026)),
so the hold has still not been exercised on any target but 530.

This experiment runs the same arm, unchanged, once on every remaining target
the planner qualified with an executable plan. It does not pick the targets
most likely to reach closure, because the CPU checks that could pick them
were not good enough (below). It runs them all and reports how many reach
closure and what the hold does when they do.

**User confirmation.** On October 6 the user asked for the remaining CPU and
GPU work to be run overnight. The budget, **1,350 GPU-minutes reserved**, was
stated to the user before submission. The runs are submitted only after the
P0 gate (S0) has passed and the scorer is committed.

## Why a sweep, not a selection

After October 5, three CPU checks were tried as perception-side selection
rules, each against recorded runs, before any of them was used:

- **Home visibility (validated, used here as prediction S1).** A full-scene ray
  cast from the camera's seed pixel applies the runner's initialization rule:
  the seed's depth must lie within 0.02 m of the expected optical Z. It
  includes the ground, the orchard, both trees, the robot's render meshes, the
  jaw boxes and the wrist mount. It reproduces all 174 recorded
  initializations (163 visible, 11 rejected) with no mismatch
  ([evidence](evidence/home_visibility_check_2026-10-05.json)). It also explains
  3721's October 4 stop: the planner's line-of-sight model has no ground plane,
  and the seed ray hits the ground 47 mm in front of the spur.
- **Jaw margin on the approach (does not discriminate).** An emulation of the
  open-jaw mask on the approach scores 530 at 8.29 px and 36196 at 7.87 px.
  530 kept its patch and 36196 lost it, yet every candidate falls between
  6.7 and 10.1 px. No threshold separates them.
- **Tracking through the re-orientation (unfinished).** Not used.

So only the visibility check is usable, and it is registered as a prediction,
not a filter. Every other way a run might stop is measured live.

## Targets

All 34 qualifying targets of the
[planner search](evidence/search_known_map_planner_2026-10-04.json), except:
- **tested October 4:** tree0 3721, tree1 36196 and tree1 18143;
- **only a shortened plan qualifies:** tree1 22182, 22242 and 34185. Their
  qualifying plans put the mouth 8 mm short of the target. The runner's
  planned-pose mode has no such parameter (it drives the mouth onto the
  target), so it cannot carry them out unchanged.

That leaves 28 targets: 18 on tree0 and 10 on tree1. Each has the screened
pool's geometry and the search's planned orientation, at a 60 mm standoff with
no shortening ([register](evidence/eval_targets_hold_sweep_2026-10-06.json)).
For 27 of them the plan is the same world orientation, the 38.3° rolled pose
of October 4; tree0 11607's is a 141.0° re-orientation. For 31 of the
original 34, a jaw-free plan also exists
([caveat](SCOPE_CLOSURE_HOLD_GENERALIZATION_2026-10-02.md#steps-1-and-2--october-4-2026)).
The rolled pose is what makes the closing jaw cover the patch, and so it is
what gives the hold something to do.

| Tree | Targets |
|---|---|
| tree0 | 4438, 7049, 7644, 7760, 8175, 8532, 11607, 11727, 12261, 12321, 12381, 13213, 16660, 21268, 21386, 28493, 28610, 29554 |
| tree1 | 3433, 3673, 20698, 20995, 21648, 26425, 27362, 33128, 34125, 36786 |

The controls are 530 and 14944 from the
[jaw_hold_b register](evidence/eval_targets_jaw_hold_b_2026-09-30.json), as on
October 4.

## The arm

The arm is the jaw-in-view arm, unchanged: `planned_pose_jaw_hold`, with the
jaw self-mask and the closure hold, under the name `planned_pose_jaw_hold_sweep`.
That name registers it to this protocol. A test and P0 check D pin that the
name is the only difference. The mask, the hold, the four checks waived on
held frames, every threshold, the grader and the scene are as in the
jaw-in-view protocol.

**Label.** Every result is *known-map plan; jaw self-mask; closure hold with
freshness and frame-reuse checks waived on held frames*. It is reported apart
from every unchanged-gate result and never pooled.

## Runs

Two `afterany`-chained batches, one task at a time, source light, 200 frames,
on `gpu,ampere` with the constraint `a40|rtx8000`, in 45-minute tasks. Each run
is labelled with its GPU model. Both are submitted from the registration
commit, so they freeze the same code.

| Batch | Runs | Register | Strategy |
|---|---|---|---|
| `hold-sweep-20261006` | the 28 targets, once each | [hold_sweep](evidence/eval_targets_hold_sweep_2026-10-06.json) | `planned_pose_jaw_hold_sweep` |
| `hold-sweep-ctl-20261006` | 530, 14944 | [jaw_hold_b](evidence/eval_targets_jaw_hold_b_2026-09-30.json) | `planned_pose_jaw_hold_sweep` |

One run per target trades repeats for coverage. Render noise can change a
single run's outcome, so the sweep reports rates over targets, not verdicts
per target.

## Pre-registered predictions

Frame conventions are those of the jaw-in-view protocol:
- 0-based `frames.json` indices;
- closure start k is the first frame whose cut phase is `closing`;
- the deadline is k + 6;
- a stop is decided on the first frame whose cut phase is `stopped`.

"A sweep run" means a run on one of the 28 targets. A run is scored only if it
was graded, its capture completed, and it ran the arm with its registered
constants. Any other run stays in every table and decides nothing.

- **S0 (pre-submission gate, CPU).** `tools/check_hold_sweep_p0.py` passes:
  - **A.** With every arm off, the controller reproduces the 141 published
    recordings through each recorded stop.
  - **B.** With the mask and the hold, it reproduces the 6 live jaw-in-view
    recordings, holds included.
  - **C.** The sweep register is exactly the qualifying targets minus the six
    exclusions above. Each entry has its pool geometry and its plan's
    orientation, at 60 mm with no shortening.
  - **D.** The sweep strategy is `planned_pose_jaw_hold` under its own name and
    protocol.

  *Any mismatch blocks submission.* **Passed** on October 6 at `b130167`, from
  a clean clone (CPU job `21600128`): A reproduced 141 of 141, B 6 of 6, and
  C and D found no problem ([evidence](evidence/hold_sweep_p0_2026-10-06.json)).
- **S1 (home visibility, predicted on the CPU).** The three sweep targets the
  visibility check rejects (tree0 4438, tree1 3433 and tree1 3673) are
  rejected at home: the cut stops at frame 0 with
  `initial_target_not_visible`. The other 25 initialize.
  - *Refuted by* any scored sweep run that departs, either way.
- **S2 (mechanism, as P4).** This is predicted for sweep runs whose closure
  starts with a gate mouth distance of at most 4 mm. The first closure loss
  comes at k + 2, the first frame rendered closing, and is mask-attributable.
  The shadow controller (`cut_without_hold`) stops there.
  - *Refuted if*, in such a run, tracking continues at k + 3, or the first loss
    comes at another frame or is not mask-attributable.
  - Runs that start closure above 4 mm are reported, not predicted.
- **S3 (hold to detachment, as P5).** In every sweep run where S2's loss
  occurs:
  - the hold covers every frame from it to the deadline;
  - the tool stays within 0.5 mm and 0.25°;
  - the gate's mouth distance on held frames equals its closure-start value
    within 0.05 mm;
  - detachment comes at k + 6.

  *Refuted if* any held frame stops, or detachment comes at another frame.
- **S4 (outcome where the hold engaged).** Among the sweep runs where S2's loss
  occurs, at least half pass all 17 checks of the unchanged grader.
  - Judged only with at least 3 such scored runs; with fewer, S4 is untested.
  - *Refuted if* fewer than half pass.
  - Every failed check is reported per run, and so is every sweep run that
    never started closure, with its stop frame, cut reason and tracker state
    (reported, not predicted).
- **S5 (static-branch premise, as P8).** In every sweep run that detaches, the
  selected piece moves less than 0.1 mm, and the true mouth-to-spur distance
  changes by less than 0.1 mm, between closure start and detach. *Refuted
  otherwise.*
- **S6 (mask fidelity, P1 as amended October 2).** On every frame of all 30
  runs:
  - the recorded mask pixel count equals an offline reconstruction from the
    recorded pose, progress and camera;
  - no predicted jaw pixel shows recorded depth more than 1 mm beyond the
    analytic box depth.

  A predicted jaw pixel is one whose centre ray hits either box and whose
  centre lies at least 0.01 px inside the undilated silhouette. *Refuted by*
  any mismatch. The literal count, without the band, is reported.
- **S7 (controls).**
  - 530 passes 17/17, its hold covers k + 2 to k + 6, and it detaches at
    k + 6.
  - 14944 passes 17/17, the hold never applies, and the mask removes no patch
    element before detach.
  - *Refuted if* either control fails a check, 14944 holds or has a patch
    element masked before detach, or a held frame of 530 stops.

**Reported, not predicted.** How many of the 25 targets predicted to
initialize reach closure, and where and why each of the others stops. That
rate is the sweep's first finding whatever S2–S4 say. If fewer than 3 runs
reach S2's loss, the hold is still untested beyond 530, and the result says
so.

Grading uses only `tools/validate_vision_sequence.py`, through
`tools/aggregate_eval.py --protocol` per batch, and every run stays in every
table. The grade files are committed before they are opened. S1–S7 are scored
by `tools/score_hold_sweep.py`, committed with this protocol, before
submission, from a clean clone.

**Scorer construction.** The scorer is the October 4 scorer, which was reviewed
blind and mutation-tested, with its registry replaced:
- S2, S3, S5, S6 and S7 keep its H2, H3, H5, H6 and H7 logic.
- S1 and S4 are new. The independent reviewer was unavailable (the account's
  subagent usage limit runs until October 8). So they were checked by their
  own boundary tests and by nine one-line mutations of S1, S4 and the
  register lookup; the tests kill all nine. That is a weaker check than a
  blind review, and the result says so.

**Exposures.** The runner exits 0 only on a 17/17 pass with a matching
configuration, so Slurm states reveal pass or fail as runs finish. The scorer
is committed before submission, so nothing seen afterwards can shape it.

## What this cannot show

- **One run per target.** A single run's outcome can turn on render noise.
  Rates over 25 targets are reported; per-target claims are not made.
- **One orientation family.** 27 of the 28 plans are the same 38.3° rolled
  pose, on two trees in source light.
- **Selection by the planner.** Every target passed the planner's three
  conditions. A target that stops before closure is a finding about the
  planner's perception predictions as much as about the hold.
- **A simulator self-model.** The jaw is the two-box surrogate, with exact
  poses and commanded closure, and the spur is kinematic: S5 checks the
  static-branch premise only in simulation.

## Cost

30 runs of 200 frames, 45 minutes reserved each, one task at a time: **1,350
GPU-minutes reserved**. October 4's runs used about 14 minutes each, so the
sweep should take about 7–8 hours. Captures are about 0.47 GB per run, about
14 GB in all. The hpc-share project quota held 1.62 TiB on October 6, above the
1.5 TiB soft limit, with 3 weeks 1 day of grace left, and below the 2 TiB hard
limit.
