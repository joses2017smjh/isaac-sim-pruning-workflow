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

## Implementation notes fixed before the run (October 1)

An adversarial review of `tools/replay_depth_appearance.py` found no defect in
the variant's arithmetic. The details this text leaves open are fixed here,
before any held-out recording is replayed, and the tool records each one in
its output.
- **Depth sampling.** The 3×3 median rule (median of the valid samples, invalid
  below 75% valid, with no spread or centre check) gives both the current
  depth and the previous patch's optical-Z. This is the study's median3 path.
- **Denominators.** The verified fraction (condition 1) is over the
  same-surface elements, those within 25 mm of the tracked depth. The nearer
  fraction (3), the median (4) and the agreement fraction are over the
  verified elements.
- **J.** "Touches" means any pixel of the committed 2.0 px mask in the
  tracker's `getRectSubPix` footprint of the previous patch (previous frame's
  mask) or the propagated patch (current mask), or at a rounded reprojected
  same-surface element (current mask). J fails closed when the silhouette is
  undefined. This is slightly stricter than the study's undilated window
  guard, and it touches none of the three earlier low-sun events.
- **Missing values.** A condition that has no value (no tracked depth, or
  nothing verified) is reported as not evaluated, and the event is rejected.
- **Strict confidence.** It is computed on every accepted event, and is the
  confidence a continuing strict result reports. H2 is judged on
  `strict_confidence`, together with the gate that ended the update.
- **Data.** The held-out set is exactly the 12 `jaw-shadow-*-20260930`
  recordings. The regression set is exactly the 129 earlier runs, excluding
  `jaw-shadow-*` and `jaw-hold-*`. A run still recording, or one recorded with
  the jaw arms, is never replayed.

## Result — October 1, 2026

All four predictions are supported. Evidence:
[verdicts](evidence/depth_heldout_verdicts_2026-10-01.json), with the
[held-out replay](evidence/depth_heldout_replay_2026-10-01.json) and the
[regression replay](evidence/depth_regression_replay_2026-10-01.json). The
replays ran at `ff363c9` (CPU job `21502037`), and the scorer was committed
at `e6fcdcc` before any held-out output existed. The base tracker reproduces
all 12 held-out and all 129 regression recordings exactly.

| Arm-A appearance stop | Depth test | Median \|Δ\| | Nearer >10 mm | Pixel vs static | Strict confidence | D_strict | Agreement arm |
|---|---|---|---|---|---|---|---|
| Evening 14944 r1 @67 | accept | 0.02 mm | 0% | 0.33 px | 0.025 | stops (low confidence) | continues |
| Evening 14944 r2 @67 | accept | 0.07 mm | 0% | 0.53 px | 0.000 | stops (low confidence) | continues |
| Evening 15004 r1 @67 | accept | 0.04 mm | 0% | 0.25 px | 0.192 | continues | continues |
| Evening 15004 r2 @67 | accept | 0.05 mm | 0% | 0.25 px | 0.204 | continues | continues |
| Morning 15004 r1 @75 | accept | 0.00 mm | 0% | 0.05 px | 0.189 | continues | continues |
| Morning 15004 r2 @75 | accept | 0.00 mm | 0% | 0.06 px | 0.265 | continues | continues |

- **H1 (supported).** The depth test accepts all 6 shadow events. The jaw
  silhouette is clear in each.
- **H2 (supported).** D_strict's decision agrees with its own confidence at
  all 6 accepted events. Evening 14944 stops on low confidence, as expected.
  The agreement arm continues at all 6.
- **H3 (supported).** In all 6 arm-B runs no grey check fails, and both arms
  equal the recording on every frame.
- **H4 (supported).** On the 129 earlier runs, D_strict diverges only at the
  three earlier low-sun events. It keeps all 22 real stops:
  - 19444 r1–r3 (the jaw);
  - 5 stops on the wire target 12142;
  - 14 `mixed_surfaces` stops.

**What this settles.** On recordings that played no part in setting its
thresholds, the depth test classifies every low-sun shadow event as a change of
brightness on an unchanged surface. Across 141 recordings it falsely accepts
no real occlusion or wrong-surface stop. With the raw NCC kept in the
confidence, D_strict would have continued past 4 of the 6 stops; the
agreement arm would have continued past all 6. As registered, the next step is
a closed-loop GPU test of D_strict + J. It needs its own protocol and the
user's approval of a budget.

**Limits.**
- **Not a closed loop.** Each result shows that the stop would not have
  happened at that frame, not what the robot does next.
- **Perfect depth.** Simulator optical-Z: no sensor noise, pose error or time
  sync.
- **Held out in recordings, not in scene.** The 12 recordings are new, but they
  are the same two targets and the same two low suns as the events that set
  the thresholds.
- **Same-depth occluders remain untested.**
