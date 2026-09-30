# A known-map re-oriented approach for the contact targets — protocol, September 27, 2026

Registered **before** submission. The
[contact diagnosis](evidence/contact_diagnosis_2026-09-27.json) reproduced all
seven recorded contact and ToF stops with a known-map swept-geometry model, and
showed that no path change with the home tool orientation clears any of them.
Only re-orienting the tool reaches a gate-clear final pose, and only two
targets, 530 and 19444, also have a clear path to it. This protocol tests that
plan in the simulator.

## Design

A new labelled approach mode, `planned_pose_standoff`
(`vision_demo_controller.py`, strategy `planned_pose` in
`tools/queue_vision_robustness.py`):

- The final tool orientation comes from a **known-map plan** per target
  ([register](evidence/eval_targets_planned_pose_2026-09-27.json)): 530
  re-orients 55.5°, 19444 re-orients 83.6° from home. This is scene geometry
  used at run time, and it is labelled as such.
- Everything else is perception. At the first tracking frame the controller
  freezes the tracked target and places a standoff point 60 mm back along the
  planned tool axis. It rotates at most 1.5° per frame toward the planned
  orientation while the mouth moves at most 4 mm per frame to that point, then
  finishes along the axis toward the live tracked target. After detachment it
  backs out to the standoff point and returns home, rotating back.
- The jaw closing axis is set to straddle the branch **at the planned
  orientation** instead of at home. The cut gate's thresholds (the 15°
  perpendicularity limit, the 8 mm mouth tolerance, stability, speed) are
  unchanged; what changes is the jaw orientation the gate evaluates. The
  camera mount stays byte-identical to the baseline.
- Controls: 14944 and 15004, the two recorded tree1 passes, run through the same
  mode with an identity plan (final orientation = home), which reduces to a
  60 mm standoff along the home tool axis.

**Labelling, fixed before submission (September 28).** Because the cut gate
evaluates a different jaw orientation, every result of this experiment is
labelled *known-map plan, jaw orientation set at the planned pose* and is
reported apart from the unchanged-gate results, never pooled with them.

No tracker setting, threshold or grader check changes. Three separate batches
of the four targets (the launcher refuses a repeated target inside one plan),
baseline tracker, source light, 200 frames.

## Pre-registered predictions

- **Q1.** 530 and 19444 reach the `align` phase with no contact over 5 N and no
  ToF stop in 3 of 3 repeats each. *Refuted if* any repeat stops on
  `hazard_contact` or `tof_minimum_clearance` before `align`. A
  `vision_invalid` stop during the re-orientation leg is a perception failure
  and leaves that repeat's geometry claim untested; this is stated in advance.
- **Q2.** 14944 and 15004 pass 17 of 17 checks in 3 of 3 repeats under the
  identity plan. *Refuted if* either fails any check.
- **Q3.** At least one of 530 and 19444 passes all 17 checks in at least one
  repeat. This is the first attempt at a full pass on a registered target that
  failed on contact under every earlier strategy. *Refuted if* neither passes
  in any repeat.

## What this cannot show

It is a known-map plan: the orientation comes from the exported scene, not from
the robot's sensors. A perception-only planner from the home depth image could
not see the arm-side obstacles (0 pixels on three of them), so this is an upper
bound on what re-orientation buys, not a deployable planner. The swept model
does not include dynamics or orientation drift; whether the tracker survives a
55–84° camera rotation is untested and is part of what this measures.

## Cost

12 trials of 200 frames on `ampere` A40, 25 minutes reserved each: **300
GPU-minutes reserved**, one task at a time, about 5 GB of captures on the
hpc-share.

## Amendment before any run — September 28, 2026

The user approved this change. The first submission (`21461306` → `21461307` →
`21461308`, A40 only, 25-minute tasks) never started, because every `ampere`
node was drained for maintenance until October 1. It was cancelled before any
task ran and resubmitted:

- **Where it runs.** Partitions `gpu` and `ampere` with the constraint
  `a40|rtx8000`, so an RTX 8000 or an A40; both have RT cores. H100 (`dgxh`) is
  excluded because it has no RT cores for the capture's path tracing.
- **Time and budget.** Tasks get 45 minutes, since a recorded run takes about
  15 minutes on an A40 and the RTX 8000 is expected to be slower. The
  reservation is 12 × 45 = **540 GPU-minutes**, approved.
- **Labelling by GPU model.** Each run's report names its node; `cn-r-*` and
  `cn-s-*` are A40, `cn-gpu*` are RTX 8000. Each run's GPU model is reported
  beside its outcome. The RGB render already differs run to run on a single GPU
  model (perception round), and a change of model may add a systematic
  difference. Repeats are therefore also reported per GPU model, and the 14944
  and 15004 controls are compared with their A40 passes with that caveat.
- **What does not change.** The targets, the `planned_pose` strategy, every
  gate and threshold, and predictions Q1–Q3.
- **Code revision.** The code revision moves from `3cd1024`. The only change
  since then in code these runs execute is two report keys that name the render
  profile in force; the launcher gained the placement option used here.

## Result — September 30, 2026

Every result here is labelled *known-map plan, jaw orientation set at the
planned pose* and is reported apart from the unchanged-gate results. All 12
planned runs are graded. Slurm marked the six planned-target runs FAILED and
the six control runs COMPLETED; the grader decides, not Slurm.
([Per-batch evidence](evidence/planned_approach_2026-09-30/),
[verdicts](evidence/planned_approach_verdicts_2026-09-30.json), code
`da38bb3`.)

| Target | r1 | r2 | r3 |
|---|---|---|---|
| 530, re-oriented 55.5° | 9/17, gate lost during closure (A40) | 9/17, same (RTX 8000) | 9/17, same (A40) |
| 19444, re-oriented 83.6° | 9/17, `vision_invalid` at 68 (A40) | 9/17, same at 68 (RTX 8000) | 9/17, same at 70 (RTX 8000) |
| 14944, identity plan | **pass** 17/17 (A40) | **pass** (RTX 8000) | **pass** (RTX 8000) |
| 15004, identity plan | **pass** 17/17 (A40) | **pass** (RTX 8000) | **pass** (RTX 8000) |

The largest contact force in all 12 runs is 0 N.

- **Q1, partly supported.**
  - *530* reaches `align` in 3 of 3 with no contact and no ToF stop. That is
    the first time the geometry of a contact target has cleared under any
    strategy.
  - *19444* never reaches `align`. The tracker's appearance check stops it on
    the final leg in all three repeats, which does not trigger the protocol's
    refutation (a contact or ToF stop). The protocol anticipated a vision stop
    only on the re-orientation leg, so for 19444 the geometry claim is
    untested.
- **Q2, supported.** Both controls pass 3 of 3, on both GPU models.
- **Q3, refuted.** Neither planned target passes in any repeat.

**Why the two planned targets stop (post hoc, from recorded ground truth and
depth).**
- **530.** It aligns with the mouth 4.3 mm from the spur and 0° off
  perpendicular, then starts to close. During closure the tool moves at most
  0.4 mm and the spur not at all. Yet the tracked point slides about 16 px (r1),
  the tracked features fall from 17 to as few as 7, and the measured spur drifts
  from 4.2–4.4 to 9.8–10.3 mm off its true position until the 8 mm gate trips. The tracker follows
  the closing jaw, whose pads sit at the spur's depth. The controls show none
  of this.
- **19444.** At the stop, pixels nearer than the tracked depth by over 1 cm
  enter the 13×13 appearance patch: 31, 42 and 89 of 169 in the three repeats.
  The open jaw passes between the camera and the spur on the final leg, a view
  that the 84° re-orientation creates.

Both stops are the robot's own jaw in its camera's view, not the orchard. A
fix would be a labelled perception experiment; see the roadmap. It is proposed,
not registered.
