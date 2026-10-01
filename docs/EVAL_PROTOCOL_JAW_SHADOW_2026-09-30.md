# Jaw-shadow counterfactual — protocol, September 30, 2026

Registered **before** submission. The
[post hoc diagnosis](EVAL_PROTOCOL_PERCEPTION_2026-09-26.md#corrections-and-diagnosis-of-the-light-dependent-stops--september-28-2026-post-hoc)
found that all three low-sun appearance stops (evening 14944 and 15004 at 67,
morning 15004 at 75) coincide, pixel by pixel and frame by frame, with the
shadow of the visual jaw surrogate crossing the tracker's 13×13 patch. No
intervention was run. This batch intervenes: it removes that one shadow and
nothing else.

## Design

- **Arm A.** The `baseline` strategy, unchanged: the jaw surrogate casts
  shadows.
- **Arm B.** The labelled `jaw_no_shadow` strategy. It is the baseline with
  `primvars:doNotCastShadows` set on both jaw cubes
  (`BlenderDemoScene.set_jaw_shadow_casting`). The jaws stay visible in RGB
  and depth. The value is read back from both prims into each report, and
  `configuration_matches` refuses a run whose readback does not show the arm.
- **Conditions.** Evening 14944 and 15004, and morning 15004
  ([evening register](evidence/eval_targets_jaw_shadow_evening_2026-09-30.json),
  [morning register](evidence/eval_targets_jaw_shadow_morning_2026-09-30.json)).
  These are exactly the registered targets whose low-sun runs stopped on the
  appearance check. Morning 14944 passed and evening 14884 was never affected,
  so both are left out.
- **Repeats.** Two per arm and condition, 12 runs in 8 batches: daylight is
  per batch, and a plan may not repeat a target. The batches form one
  `afterany` chain.
- **What does not change.** No tracker setting, gate, threshold or grader
  check; `tools/validate_vision_sequence.py` alone grades.
- **Placement.** `gpu,ampere` with the constraint `a40|rtx8000` and 45-minute
  tasks. Each run is labelled with its GPU model. The RGB render is not
  repeatable run to run, which is why there are two repeats.

## Measures

- The grader's outcome, and the recorded patch correlation at 57–59 and 67
  (evening) and 73–75 (morning).
- **Patch darkening.** The change in mean grey of the tracker's 13×13 patch
  (the tracker's own sampling at the recorded pixel) from 56 to 59 (evening)
  and from 72 to 75 (morning). Recorded arm A values
  ([diagnosis evidence](evidence/appearance_loss_diagnosis_2026-09-28.json)):
  14944 146.7 → 77.6 (−47%), 15004 123.2 → 60.8 (−51%), morning 15004 92.2 →
  53.6 (−42%).

## Pre-registered predictions

- **J1 (the switch works).** Every arm-B run reads back
  `doNotCastShadows = true` on both cubes, and its patch darkens by less than
  15% over 56–59 (evening) or 72–75 (morning). *Refuted if* any arm-B patch
  darkens by 30% or more over those frames, which would mean the shadow still
  falls.
- **J2 (evening cause).** Arm B evening 14944 and 15004 have no
  `appearance_changed_or_occluded` loss during the approach, and their patch
  correlation is at least 0.85 at 57–59 and 67, in 2 of 2 repeats each.
  *Refuted if* any arm-B evening repeat stops on an appearance loss during the
  approach; the jaw shadow is then not the sole cause.
- **J3 (morning cause).** Arm B morning 15004 completes closure without an
  appearance loss, with correlation at least 0.9 at 73–75, in 2 of 2. *Refuted
  if* either repeat loses appearance during closure.
- **J4 (arm A reproduces).** Arm A evening 14944 loses appearance at the last
  approach frame (correlation under 0.5 at 67) in 2 of 2. 15004's margin (0.297
  recorded, 0.053 under the gate) and arm A morning 15004 are reported as
  measurements, not predicted. *Refuted if* arm A evening 14944 passes in both
  repeats; the recorded stop was then render noise.
- **J5 (what the shadow cost).** Arm B evening 14944 and 15004 pass 17 of 17 in
  at least 3 of their 4 runs. *Refuted if* fewer than 2 of 4 pass.

## What this cannot show

- The caster is a surrogate. A real jaw would cast its own shadow with
  different timing and extent, so this measures the surrogate's effect in the
  simulator, not a field effect.
- Removing the shadow is not a remedy a robot can apply. It isolates the
  cause, and any fix is a separate labelled tracker experiment, for example a
  depth-aware appearance check.
- Three targets, two repeats each: this is a mechanism test, not a rate.

## Cost

12 runs of 200 frames, 45 minutes reserved each, one task at a time: **540
GPU-minutes reserved** (approved by the user on September 30). About 5.4 GB of
captures on the hpc-share.

## Result — October 1, 2026

All 12 runs are graded and all 5 predictions are supported
([verdicts](evidence/jaw_shadow_verdicts_2026-10-01.json),
[per-batch evidence](evidence/jaw_shadow_2026-10-01/), code `e1684a8`). Every
run ran on an A40.

| Condition | Arm A (jaw casts shadow) | Arm B (jaw casts no shadow) |
|---|---|---|
| Evening 14944, r1 / r2 | stops at 67, appearance loss (correlation 0.07 / −0.05) | **pass** 17/17 / **pass** |
| Evening 15004, r1 / r2 | stops at 67, appearance loss (0.19 / 0.20) | **pass** / **pass** |
| Morning 15004, r1 / r2 | stops at 75 in closure, appearance loss (0.22 / 0.33) | **pass** / **pass** |

- **J1.** Both jaw cubes read back `doNotCastShadows = true` in every arm-B
  run. Over the shadow frames the arm-B patch changes by −3.7% to +0.6%, while
  the arm-A patch darkens 41–54%, as recorded before.
- **J2.** Arm-B evening correlation is at least 0.99 at 57–59 and 67, with no
  appearance loss.
- **J3.** Arm-B morning correlation is 1.00 at 73–75, and closure completes.
- **J4.** Arm A reproduces the original stops at the original frames in all
  six runs. Evening 15004 again falls under the gate (0.19, 0.20).
- **J5.** 4 of 4 arm-B evening runs pass.

**What this settles.** The
[post hoc diagnosis](EVAL_PROTOCOL_PERCEPTION_2026-09-26.md#corrections-and-diagnosis-of-the-light-dependent-stops--september-28-2026-post-hoc)
found an association; this is an intervention. Removing only the visual jaw
surrogate's shadow, with the jaw still visible in RGB and depth, turns 6 of 6
low-sun appearance stops into passes. Keeping the shadow reproduces 6 of 6 at
the same frames. In this simulator the light-dependent tree1 outcomes
(perception round, P6) are caused by the jaw surrogate's low-sun shadow
crossing the tracker's appearance patch.

**Limits.**
- **The caster is the two-box surrogate**, not the pruner CAD. A real jaw also
  casts a shadow, with different timing and extent.
- **Removing a shadow is not a remedy.** The candidate remedy is the
  user-suggested depth-aware appearance check. As agreed, it comes next as a
  zero-GPU held-out replay on these 12 recordings: arm A's appearance events
  played no part in setting its thresholds, and arm B is a negative control
  where it must not act.
- **This is a mechanism test, not a rate:** three targets, two repeats each.
