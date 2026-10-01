# Depth-aware appearance check: held-out offline replay — protocol, October 1, 2026

Registered **before** the replay is run on its held-out data. The user
suggested giving the tracker more inputs, so that lighting would not stop it.
An offline study on the earlier recordings then designed and verified a
depth-aware appearance check, D.

The [jaw-shadow counterfactual](EVAL_PROTOCOL_JAW_SHADOW_2026-09-30.md#result--october-1-2026)
showed that the low-sun appearance stops are caused by the jaw surrogate's
shadow, so D's premise holds: the patch changes in brightness, not in surface.

D's thresholds were fixed after the study had seen the 8 earlier grey-check
failures. The counterfactual's 12 recordings are new data: none of its
appearance events played any part in setting them. The user chose this
zero-GPU held-out test before any closed-loop run.

## The variant: `D_strict + J` (offline only; no controller change)

A tracker variant that differs from the repository tracker only when the grey
NCC appearance check fails (below 0.35). It then runs a static-world depth test
on the previous 13×13 patch:
- back-project it with the previous frame's optical-Z and pose (pixel centres
  at index + 0.5);
- reproject it with the current pose, and sample the current depth with the
  tracker's own 3×3 median rule;
- keep the pixels within 25 mm of the tracked depth.

The test accepts only if all of these hold:
1. at least 75% of the patch pixels are verified;
2. at least 20 same-surface pixels remain;
3. at most 5% of the pixels are nearer than predicted by more than 10 mm;
4. the median |measured − predicted| is at most 3 mm;
5. the propagated pixel is within 3 px of the static-world prediction;
6. (J) the 2.0 px jaw silhouette from the committed
   `perception/jaw_self_mask.py` touches none of the previous, reprojected or
   propagated patches.

On acceptance the 0.35 gate passes. The raw NCC stays in the unchanged
confidence `min(1, n/n0) · NCC · depth_valid_fraction ≥ 0.15`, which is why
this is the strict variant. A secondary arm, labelled `agreement`, replaces NCC
by the depth-agreement fraction in that confidence. It is reported separately
because it changes the confidence gate's input.

**Label.** *Simulator depth: RTX optical-Z ground truth, noise-free, perfectly
registered, exact camera motion; not a sensor. The variant changes the 0.35
appearance gate's rule.*

The tool is `tools/replay_depth_appearance.py` (CPU). It must reproduce the
recording exactly whenever the grey check passes.

## Data

- **Held out.** The 12 jaw-shadow recordings
  ([evidence](evidence/jaw_shadow_2026-10-01/)). Arm A has 6 appearance stops
  (evening 14944 and 15004 at 67, morning 15004 at 75). Arm B has none.
- **Regression.** Every earlier recorded run with frames (129).

Only decisions up to each run's recorded stop are evaluable. Frames after a
recorded stop show a stopped robot and are not counterfactual.

## Pre-registered predictions

- **H1 (the test classifies the shadow events as illumination).** At each of
  the 6 arm-A appearance-stop frames, the depth test accepts: the silhouette is
  clear, there are no nearer pixels beyond the limit, and the median is within
  3 mm. *Refuted if* it rejects any of the 6. Each rejection is reported with
  the condition that failed.
- **H2 (what the strict variant does with an accepted event).** At each
  accepted event, D_strict continues if and only if the strict confidence is at
  least 0.15. The recorded correlations put evening 14944 (0.07, −0.05) below
  that, so a low-confidence stop is expected there. For evening and morning
  15004 (correlation 0.19–0.33) the decision is reported, not predicted,
  because it depends on the feature ratio at that frame. The agreement arm
  continues at every accepted event. *Refuted if* D_strict's decision
  disagrees with its own computed confidence, or if the agreement arm stops at
  an accepted event.
- **H3 (negative control).** In all 6 arm-B runs no grey check fails, so D
  never acts and the replay equals the recording on every frame. *Refuted by*
  any divergence.
- **H4 (regression).** On the 129 earlier runs, D_strict + J diverges from the
  recording only at the earlier low-sun appearance events. Every real occlusion
  or wrong-surface stop is kept: 19444 r1–r3 (the jaw), 12142 (the wire), and
  every `mixed_surfaces` stop. *Refuted if* any of these continues, or anything
  else diverges.

If H1, H3 and H4 hold, a closed-loop GPU test of D_strict + J is proposed to
the user, with its own protocol and budget. If H1 fails, D's thresholds do not
generalize, and no GPU test is proposed.

## What this cannot show

- **Not a closed loop.** The replay sees the recorded frames up to each stop.
  It can show that the stop would not have happened at that frame, but not what
  follows.
- **Perfect depth.** No sensor noise, pose error or time-sync error is
  modelled here.
- **Same-depth occluders are untested.** Ties, leaves, or a wire touching the
  spur were never recorded.
