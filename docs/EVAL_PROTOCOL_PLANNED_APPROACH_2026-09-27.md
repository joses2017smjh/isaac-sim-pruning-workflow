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
