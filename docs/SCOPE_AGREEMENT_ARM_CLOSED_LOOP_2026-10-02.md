# Evening 14944: the agreement arm in closed loop — scope, October 2, 2026

**Status: scoping only.** Not registered. The user approved this follow-up in
principle on October 2. No GPU work runs until the user approves the budget
below. Independent verifiers checked every factual claim here against the
recordings, the code and the committed evidence. Their corrections are
applied.

## Question

The depth-aware check D_strict + J turned the low-sun shadow stops of 15004
into passes, but evening 14944 still stopped (C3 of the
[closed-loop result](EVAL_PROTOCOL_DEPTH_APPEARANCE_CLOSED_LOOP_2026-10-01.md#result--october-2-2026)).

The [held-out protocol](EVAL_PROTOCOL_DEPTH_APPEARANCE_HELDOUT_2026-10-01.md)
already registered a secondary arm, `agreement`. When the depth test accepts
a grey-check failure, this arm replaces the NCC in the confidence with the
depth-agreement fraction. Does that arm let evening 14944 continue and pass in
closed loop, while the two recorded occlusion controls still stop?

## What the recordings already show

Every evening 14944 recording with the jaw's shadow stops at frame 67: one in
the perception round, two in the jaw-shadow counterfactual's shadow-kept arm,
and two in the depth closed loop. Frame numbers are 0-based `frames.json`
indices, and 67 is the first frame whose cut phase is `stopped`. The
per-batch evidence records the same stop as `stop_frame` 68, the first
`stopped_failure` frame. At frame 67 in all five:
- **Optical flow still tracks, on fewer features.** 6–8 of the 16 tracked
  features survive the forward-backward check, down from 13–14 at frame 66.
  The tracker refuses the flow below 4. The median round-trip error is at
  most 0.04 px.
- **Only the appearance score fails.** The frame-to-frame patch correlation
  drops from about 0.9 to between −0.048 and 0.103.
- **The depth test accepts in all five.** It ran live in the two closed-loop
  runs, and in the
  [regression](evidence/depth_regression_replay_2026-10-01.json) and
  [held-out](evidence/depth_heldout_replay_2026-10-01.json) replays of the
  other three:
  - median surface change 0.02–0.08 mm (live: 0.04 and 0.07 mm);
  - the flow position lies 0.32–0.60 px from the static-world prediction
    (live: 0.32 and 0.60 px);
  - all 169 patch elements are verified, with agreement fraction 1.0.

The position is still measured, and only the appearance score collapses. A
rule change in the confidence gate addresses that; a new sensor input is not
needed.

The agreement arm is that rule change. It is already implemented
(`DepthAppearanceTracker(arm="agreement")` in
`perception/depth_appearance.py`) and already replayed:
- **Held-out events (H2).** It continued at all 6 held-out shadow events. At
  evening 14944 its confidence is only the surviving feature fraction,
  because the agreement and valid-depth fractions are both 1.0. That is 0.375
  (6 of 16) in four recordings and 0.50 (8 of 16) in one, against the 0.15
  gate.
- **The 129 earlier runs.** It was recorded on all of them: 126 equal the
  recording, and the 3 that diverge are exactly the earlier low-sun events
  (`agreement_arm_regression_divergences` in the
  [held-out verdicts](evidence/depth_heldout_verdicts_2026-10-01.json)). That
  was reported, not a registered prediction: H4 covers D_strict + J only.
- **Real stops.** The depth test runs only where the grey check fails. In the
  129 earlier runs it ran at 4 of the 22 kept real stops: the jaw on 19444
  r1–r3 and the wire on 12142. It rejected all 4. The other 18 passed the grey
  check and stopped at later gates, which the arm does not change. Live, it
  rejected the one grey failure of each control in the depth closed loop (C4).

## What the arm gives up

On frames where the depth test accepts, the arm removes a backstop:
- **D_strict.** A falsely accepted occluder still faces the 0.15 confidence
  gate, with the raw NCC in it.
- **The agreement arm.** The NCC leaves the confidence. Condition 4 keeps the
  agreement fraction at 0.5 or more, so the gate can stop an accepted frame
  only when the feature ratio times the valid-depth fraction is below 0.3.
- **The nearest miss.** The depth test rejected the open jaw on 19444 (the
  closed-loop control at frame 70) on the silhouette condition J and on a
  median surface change of 4.1 mm against the 3 mm limit. Same-depth occluders
  were never recorded.

Elsewhere the arm is identical to D_strict + J. The controller and the cutter
read only the tracker's state and target, never its confidence. So wherever
both arms continue, a run is the same under either.

## Data split

- **Design set:** every existing recording, since all of them have been seen
  at least in summary. Nothing new is fitted on them: the arm, its thresholds
  and the 0.15 gate were registered on October 1 and are unchanged.
- **How the arm was chosen.** It was chosen for evening 14944 after two
  results on these recordings: offline, it continued at that target's shadow
  events (H2), and live, D_strict + J stopped there (C3). Its continuation at
  frame 67 has already been seen in replay, so that frame is not a test.
- **Test:** what follows frame 67 in new closed-loop runs registered before
  submission, and their grade.

## What would be built (CPU, before any GPU run)

1. **An offline design study first.** It writes an evidence JSON committed
   before the protocol is registered.
   - **Frames 68–72 can be replayed; the closure cannot.** The approach ends at
     frame 67 whether or not a run stops. The two shadow-removed runs
     (`jaw-shadow-eve-b`) entered align at 67 and closing at 71, and their jaw
     first moved at 73. Before replaying, the study records how far the
     stopped runs' tool poses lie from theirs (about 1 mm and 0.03°).
   - **The replay past the stop.** `tools/replay_depth_appearance.py` gains an
     option to continue past a recorded stop. With it, the agreement arm is
     replayed on the five shadowed recordings through frame 72, and no
     further.
   - **An uncommitted pointer, to be reproduced.** A verifier's uncommitted
     exploratory replay suggests the grey check passes again on its own at
     frame 68 (NCC 0.82–0.88) and stays at 0.98 or above through 72. This
     is not evidence until the study reproduces it with the committed tool.
   - **The closure frames, 73–77.** They are not in any stopped recording. A
     CPU projection, starting from the committed shadow model
     (`tools/diagnose_appearance_loss.py`) and the no-shadow runs' geometry,
     predicts whether the closing jaw's shadow reaches the patch there.
2. **The controller path.** Every step from the plan row to the tracker
   assumes the strict arm today, so each one changes:
   - **The launcher.** `run_environment` in `tools/run_vision_experiment.py`
     turns any true value into `1`, which launches D_strict + J. It must
     forward the arm.
   - **The runner.** `hpc/inner/render_pruning_workflow.py` accepts `0`, `1`
     or `agreement`. Its refusals (Blender vision only, never with the jaw
     self-mask or the closure hold) and the reported `enabled` apply to any
     value other than `0`.
   - **The tracker and report.** `VisionPruningDemo` takes the arm instead of
     building `arm="strict"`. The report records the arm the controller
     built: today the only recorded arm is `constants.arm`, hard-coded
     `strict`.
   - **The configuration check.** `configuration_matches` refuses a capture
     that ran another arm.
3. **Launcher strategies and a register.** The strategy
   `baseline_depth_agreement`, plus the existing control strategies with the
   agreement arm. A register holding 14944 alone: a plan takes one light and
   one strategy and may not repeat a target, so each repeat is its own batch.
4. **A C0 replay gate.** Any mismatch blocks submission. The controller is
   driven through `tools/replay_jaw_self_mask.py`, through each recorded stop:
   - **A.** With every arm off, it reproduces the 141 recordings of the
     held-out replay exactly.
   - **B.** With the strict arm, it reproduces the 10 depth closed-loop
     recordings exactly, so adding the new value does not change D_strict.
   - **C.** With the mask and the hold, it reproduces the 6 jaw-in-view
     recordings exactly.
   - **D.** With the agreement arm, it measures what the published agreement
     arm measured: the same divergent frames, measurements and depth events.
5. **A scorer**, adapted from `tools/score_depth_loop.py`, reviewed blind and
   committed before submission.
   - The runner exits 0 only on a 17/17 pass with a matching configuration, so
     Slurm states reveal outcomes. Run times do as well: in the depth closed
     loop the failed runs took 23–25 minutes and the passes 26–27.
   - Until the scorer is committed, completion is checked only by whether a
     job is still in the queue, never by its state or elapsed time.

## Proposed runs (to be registered)

Only evening 14944 can show the arm's effect. It is the only recorded scene
where the strict confidence is below 0.15 at an accepted event: 0.000–0.045
in all five recordings, against 0.375–0.50 for the agreement arm. On 15004,
both arms continue at all 10 recorded accepted events, and the controller
ignores the confidence, so its runs would repeat D_strict + J. Source-light
runs have no recorded grey failure. The C0 gate covers both.

| Batch | Light | Target | Runs | What it shows |
|---|---|---|---|---|
| Evening 14944, r1–r4 | evening | 14944 | 4 | the arm's effect: what follows frame 67, and the grade |
| Control: jaw | source | 19444 (planned pose) | 1 | the nearest miss stays rejected with the new arm live |
| Control: wire | source | 12142 (tool-axis standoff) | 1 | the wire stays rejected with the new arm live |

Draft predictions, to be fixed in the protocol, each with what refutes it:
- **A1 (the event is accepted, and the arm continues).** In each evening 14944
  run, the depth test accepts the first grey-check failure in frames 56–67,
  and the arm continues at every accepted event. *Refuted if* the depth test
  rejects that first failure in any run, or the arm stops at an accepted
  event. A run with no grey failure in the window leaves A1 untested for that
  run, and this is reported.
- **A2 (evening 14944 passes).** At least 3 of the 4 runs pass all 17 checks
  of `tools/validate_vision_sequence.py`, and none stops on appearance in
  frames 56–67. *Refuted if* fewer than 2 pass, or at least 2 stop on
  appearance in that window. Each failed run is reported with its stop frame,
  tracker state and reason, depth events and failed checks.
- **A3 (real stops are kept).** Neither control passes, and the depth test
  rejects every grey failure in both. *Refuted if* either control passes, or
  the depth test accepts any grey failure in either.
- **A4 (live equals offline).** For every run, the agreement arm of the offline
  tool replays the run's own frames through its stop and equals the recorded
  live measurement, with discrete fields exact and floats within 1e-6, and
  the same depth event at every grey failure. *Refuted by* any mismatch.

Every result carries the label: *agreement (+ J): on acceptance the 0.35 gate
passes and the depth-agreement fraction replaces the NCC in the confidence,
which changes the confidence gate's input; simulator depth*. It is reported
apart from D_strict + J and never pooled.

## Cost

- **CPU.** A few hours on `share`: the design study, the C0 replays, tests and
  the scorer's blind review.
- **GPU.** 6 runs × 45 minutes = **270 GPU-minutes reserved**. The depth
  closed loop used 256 of its 450. The user will be asked to approve this
  figure before submission, after the design study.
- **Storage.** About 0.47 GB of captures per run, so about 3 GB. The share's
  project quota held 1.504 TiB on October 2: above the 1.5 TiB soft quota,
  below the 2 TiB hard limit, with 3 weeks 4 days of grace left.

## Risks

- **Closure, frames 73–77, is not observed for shadowed 14944.** Its evening
  shadow may cross the patch again as the jaw closes.
  - On 15004 in the same light, the patch correlation fell to 0.49–0.52 at
    frame 75, still above 0.35.
  - In the morning it fell to 0.24–0.25. The depth test accepted both events
    with J clear, and the agreement arm continued there in replay (H2).
  - Each grey failure in closure must be accepted with J clear and at least 4
    flow inliers.
- **A real occluder at the surface's own depth would not be caught.** With the
  NCC out of the confidence, only the depth test stands in the way. Simulator
  depth is noise-free and exact; a real sensor would need its own margins.
