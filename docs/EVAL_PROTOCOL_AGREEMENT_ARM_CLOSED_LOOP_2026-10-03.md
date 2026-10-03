# Evening 14944: the agreement arm in closed loop — protocol, October 3, 2026

Registered **before** submission. The
[depth closed loop](EVAL_PROTOCOL_DEPTH_APPEARANCE_CLOSED_LOOP_2026-10-01.md#result--october-2-2026)
turned the low-sun shadow stops of 15004 into passes, but evening 14944 still
stopped at frame 67 (C3). In all five shadowed evening 14944 recordings, at that
frame:
- optical flow still tracks;
- the static-world depth test accepts;
- only the appearance score collapses.

The [scope](SCOPE_AGREEMENT_ARM_CLOSED_LOOP_2026-10-02.md) sets out the case
for testing the arm the held-out protocol already registered for exactly
that.

**User confirmation.** On October 3 the user approved 270 GPU-minutes. The
runs are submitted only after the design study, the C0 gate (A0) and a scorer
reviewed blind have all passed.

## The arm: `agreement` (+ J)

It is the depth-aware check of the depth closed loop with its other
registered arm. Its depth test is unchanged: thresholds, 3×3 median sampling,
the jaw guard J (`perception/depth_appearance.py`). The difference is what
happens on a frame where that test accepts a grey-check failure:
- **Strict (D_strict + J).** The 0.35 gate passes, and the raw NCC stays in the
  confidence `min(1, n/n0) · NCC · depth_valid_fraction ≥ 0.15`.
- **Agreement.** The 0.35 gate passes, and the depth-agreement fraction
  replaces the NCC in that confidence. Condition 4 keeps the fraction at 0.5 or
  more, so the confidence gate can stop an accepted frame only when the
  feature ratio times the valid-depth fraction is below 0.3.

On every other frame the two arms are identical. The controller and the cutter
read only the tracker's state and target, never its confidence.

**How it is selected.** `VisionPruningDemo(depth_appearance=True,
depth_appearance_arm="agreement")` builds the tracker with this arm. The
runner sets it from `PRUNING_DEPTH_APPEARANCE=agreement`. Three strategies
forward it:
- `baseline_depth_agreement`;
- `planned_pose_depth_agreement`;
- `tool_axis_standoff_depth_agreement`.

Each capture reports the arm's registered constants, `arm: agreement`
included, and `configuration_matches` refuses a capture that ran another arm.
The arm is never combined with the jaw self-mask or the closure hold, and the
jaw keeps its shadow.

**What it gives up.** On an accepted frame, the NCC no longer backs up the
depth test. A real occluder that passes the depth test, for example one at
the surface's own depth, would not be caught by the confidence gate. The
nearest recorded miss is the open jaw on 19444: the depth test rejected it
only on J and on a median surface change of 4.1 mm against the 3 mm limit.

**Label.** Every result is *agreement arm of the depth-aware appearance check
(+ J): on acceptance the 0.35 gate passes and the depth-agreement fraction
replaces the NCC in the confidence, which changes the confidence gate's
input; simulator depth (RTX optical-Z ground truth, noise-free, perfectly
registered); not a sensor*. It is reported apart from D_strict + J and from
every unchanged-gate result, and never pooled.

## Evidence before submission

- **Design study: replay past the stop**
  ([evidence](evidence/agreement_design_replay_2026-10-03.json),
  `tools/replay_past_stop.py`). The approach ends at frame 67 in all seven
  evening 14944 recordings: the five shadowed runs stop there, and the two
  shadow-removed runs enter align. The stopped runs' poses stay within about
  1 mm and 0.03° of the passing runs' through frame 72. Replayed through frame
  72 on the five shadowed recordings, the agreement arm keeps tracking every
  frame:
  - the appearance score recovers on its own at frame 68 (0.82–0.88) and
    stays at 0.978 or above from 69;
  - all 16 features are back by 69;
  - no further grey-check failure occurs;
  - the mouth stays 3.4–5.1 mm from the target, inside the 8 mm gate.

  The strict arm stops at 67, as recorded. Frames 73–77 (closure) are not in
  any shadowed recording.
- **Design study: the closing jaw's shadow**
  ([evidence](evidence/agreement_design_closure_shadow_2026-10-03.json),
  `tools/project_closure_shadow.py`). A direct-sun projection of the two-box
  jaw, validated on the recorded shadow events, predicts that the closing
  jaw's shadow reaches evening 14944's patch:
  - 36–40% of the patch shadowed at frame 73;
  - 65–70% at 74;
  - 83–89% at 75, after which it stays covered.

  In the closest validated case, evening 15004's closure, the appearance score
  fell only to 0.49–0.52. The projection predicts where and when the patch's
  sun state changes, not the score, so whether the grey check fails during
  closure is open.
- **C0 replay gate**
  ([evidence](evidence/agreement_c0_2026-10-03.json),
  `tools/check_agreement_c0.py`). See A0.

## Runs

Six batches of one run each, `afterany`-chained and one task at a time. All
use 200 frames on `gpu,ampere` with the constraint `a40|rtx8000`, in 45-minute
tasks. Each run is labelled with its GPU model.

| Batch | Light | Register | Target | Strategy |
|---|---|---|---|---|
| `agree-eve-r1-20261003` | evening | [14944](evidence/eval_targets_agreement_14944_2026-10-03.json) | 14944 | `baseline_depth_agreement` |
| `agree-eve-r2-20261003` | evening | 14944 | 14944 | `baseline_depth_agreement` |
| `agree-eve-r3-20261003` | evening | 14944 | 14944 | `baseline_depth_agreement` |
| `agree-eve-r4-20261003` | evening | 14944 | 14944 | `baseline_depth_agreement` |
| `agree-ctl-19444-20261003` | source | [19444](evidence/eval_targets_agreement_19444_2026-10-03.json) | 19444 | `planned_pose_depth_agreement` |
| `agree-ctl-12142-20261003` | source | [12142](evidence/eval_targets_agreement_12142_2026-10-03.json) | 12142 | `tool_axis_standoff_depth_agreement` |

Only evening 14944 can show the arm's effect. It is the only recorded scene
where the strict confidence is below 0.15 at an accepted event: 0.000–0.045
in all five recordings. There the agreement arm reaches 0.375–0.50. On 15004,
both arms continue at all 10 recorded accepted events, so its runs would
repeat D_strict + J. Source-light runs have no recorded grey failure. The two
controls are the jaw and the wire, the real stops the arm must keep.

## Pre-registered predictions

- **A0 (pre-submission gate, CPU).** `tools/check_agreement_c0.py` passes
  checks A–G, which together cover:
  - the offline tool and the controller, with every arm off, with the strict
    arm and with the agreement arm, against the published replays of 141
    recordings;
  - the strict arm on the 10 live depth closed-loop recordings;
  - the agreement arm's departures from those 10 recordings: exactly where an
    accepted event ended in a low-confidence stop;
  - the jaw arms on the 6 live jaw-in-view recordings.

  Discrete fields must be exact and floats within 1e-6. *Any mismatch blocks
  submission.*
- **A1 (the shadow event is accepted, and the arm continues).** In each
  evening 14944 run, the depth test accepts the first grey-check failure in
  frames 56–67, and no accepted event ends in a low-confidence stop. *Refuted
  if* the depth test rejects that first failure in any run, or an accepted
  event ends in a low-confidence stop. A run with no grey failure in frames
  56–67 leaves the first clause untested for that run, and this is reported.
  A stop at an accepted event on another gate (depth window, calibration,
  world jump) is reported and counts against A2, not A1.
- **A2 (evening 14944 passes).** At least 3 of the 4 runs pass all 17 checks
  of the unchanged grader, and none stops on appearance in frames 56–67.
  - *Refuted if* fewer than 2 pass, or at least 2 stop on appearance in that
    window.
  - Between these, A2 is partly supported.
  - Each failed run is reported with its stop frame, tracker state and reason,
    depth events and failed checks.
- **A3 (closure; an expectation, not a refutation condition).** The design
  study predicts the closing jaw's shadow on the patch at frames 73–75. A frame
  counts as closure when its own recorded cut phase, or the previous frame's, is
  `closing`; this includes a stop during closure and the frame that completes
  it. Every grey-check failure during closure in the evening 14944 runs is reported with
  its frame, depth-test decision, J and confidence. The expectation is that
  each is accepted with J clear.
- **A4 (real stops are kept).** Neither control passes, and the depth test
  rejects every grey-check failure in both.
  - 19444 should stop when the open jaw reaches the patch.
  - 12142 should stop at the wire.

  *Refuted if* either control passes, or the depth test accepts any grey
  failure in either. A control with no grey failure leaves the second clause
  untested.
- **A5 (live equals offline).** For every run, the agreement arm of
  `tools/replay_depth_appearance.py` replays that run's own recorded frames.
  Through the run's own stop, it equals the recorded live measurement, with
  the same depth event at every grey failure. *Refuted by* any mismatch.

Grading uses only `tools/validate_vision_sequence.py`, through
`tools/aggregate_eval.py --protocol` per batch, and every run stays in every
table. A1–A5 are scored by a committed scorer, reviewed blind and committed
before submission. The runner exits 0 only on a 17/17 pass with a matching
configuration, so the Slurm states reveal outcomes. Because the scorer is
committed before submission, they are recorded as they arrive.

## What this cannot show

- **Simulator depth.** A real depth sensor's noise, holes and registration
  error would need their own margins, and would bear directly on what the arm
  gives up.
- **One target in one light.** Four repeats of evening 14944, plus two
  controls, cannot show how often the agreement arm accepts a real occluder.
- **Closure is a new regime for this target.** No shadowed evening 14944 run
  has reached closure. A failure there is a finding about closure under this
  arm, not about the frame-67 event.
- **Render noise.** RGB renders are not repeatable run to run.

## Cost

6 runs of 200 frames, 45 minutes reserved each, one task at a time: **270
GPU-minutes reserved** (approved October 3). About 3 GB of captures on the
hpc-share, whose project quota is above the 1.5 TiB soft limit and below the
2 TiB hard limit.
