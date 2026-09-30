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
