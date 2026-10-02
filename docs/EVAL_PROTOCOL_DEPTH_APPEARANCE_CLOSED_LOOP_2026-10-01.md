# Depth-aware appearance check in closed loop — protocol, October 1, 2026

Registered **before** submission. The
[held-out replay](EVAL_PROTOCOL_DEPTH_APPEARANCE_HELDOUT_2026-10-01.md#result--october-1-2026)
supported all four of its predictions:
- the static-world depth test accepted all 6 low-sun shadow events on
  recordings that played no part in setting its thresholds;
- across 141 recordings it accepted no real occlusion and kept all 22 jaw,
  wire and mixed-surface stops;
- D_strict + J would have continued past 4 of the 6 held-out stops.

A replay cannot show what the robot does after a stop that did not happen.
This test runs D_strict + J inside the live controller.

## The arm: `depth_appearance` (D_strict + J)

The variant was moved unchanged into
`source/isaaclab_pruning/isaaclab_pruning/perception/depth_appearance.py`:
- the thresholds, the 3×3 median sampling, the denominators and the jaw guard
  are as fixed in the held-out protocol's implementation notes;
- the offline tool now imports it from there.

`VisionPruningDemo(depth_appearance=True)` builds the tracker as the strict
arm. Each frame it hands the tracker the robot's own jaw boxes, built from the
same self-model as the jaw self-mask:
- the rendered tool pose;
- the closure progress rendered into the frame (the cut step before the
  update);
- the attachment roll and the scene's branch radius.

The flag is default off. It is forwarded as `PRUNING_DEPTH_APPEARANCE` by three
strategies:
- `baseline_depth_appearance` (straight approach);
- `planned_pose_depth_appearance` (known-map approach);
- `tool_axis_standoff_depth_appearance`.

Each capture reports the arm's registered constants, and
`configuration_matches` refuses a capture without them. The arm is never
combined with the jaw self-mask or the closure hold. The jaw keeps its shadow.

**What it changes.** Only the 0.35 appearance gate's rule, and only on a frame
where the grey NCC fails it. On acceptance the raw NCC stays in the unchanged
confidence `min(1, n/n0) · NCC · depth_valid_fraction ≥ 0.15`. Not changed:
the cut gate, every other tracker check, the grader's 17 checks, the scene,
the camera mount and the approach logic.

**Label.** Every result is *depth-aware appearance check D_strict + J; changes
the 0.35 appearance gate's rule; simulator depth (RTX optical-Z ground truth,
noise-free, perfectly registered); not a sensor*. It is reported apart from
every unchanged-gate result and never pooled.

## Runs

The batches are `afterany`-chained behind the jaw-in-view batches and run one
task at a time. All use 200 frames on `gpu,ampere` with the constraint
`a40|rtx8000`, in 45-minute tasks. Each run is labelled with its GPU model.

| Batch | Light | Register | Targets | Strategy |
|---|---|---|---|---|
| `depth-loop-eve-r1-20261001` | evening | [tree1](evidence/eval_targets_depth_loop_tree1_2026-10-01.json) | 14944, 15004 | `baseline_depth_appearance` |
| `depth-loop-mor-r1-20261001` | morning | [15004](evidence/eval_targets_depth_loop_15004_2026-10-01.json) | 15004 | `baseline_depth_appearance` |
| `depth-loop-eve-r2-20261001` | evening | tree1 | 14944, 15004 | `baseline_depth_appearance` |
| `depth-loop-mor-r2-20261001` | morning | 15004 | 15004 | `baseline_depth_appearance` |
| `depth-loop-src-20261001` | source | tree1 | 14944, 15004 | `baseline_depth_appearance` |
| `depth-loop-ctl-19444-20261001` | source | [19444](evidence/eval_targets_depth_loop_19444_2026-10-01.json) | 19444 | `planned_pose_depth_appearance` |
| `depth-loop-ctl-12142-20261001` | source | [12142](evidence/eval_targets_depth_loop_12142_2026-10-01.json) | 12142 | `tool_axis_standoff_depth_appearance` |

The recorded behaviour of each scene without the arm:

| Scene | Recorded stop | Offline depth test at the stop | Offline strict confidence |
|---|---|---|---|
| Evening 14944 | 67, appearance (3 of 3) | accept | 0.000–0.039 |
| Evening 15004 | 67, appearance (3 of 3) | accept | 0.19–0.30 |
| Morning 15004 | 75 in closure, appearance (3 of 3) | accept | 0.19–0.27 |
| 19444, planned pose | 68 / 68 / 70, appearance: the open jaw | reject (J; nearer 49–55%; median 3.3–28 mm) | — |
| 12142, tool-axis standoff | 29, appearance: a wire | reject (nearer 93%, median 242 mm) | — |
| Source 14944, 15004 | none: both pass in every source batch | not consulted | — |

## Pre-registered predictions

- **C0 (pre-submission gate, CPU).** `tools/check_depth_loop_c0.py` runs on
  the 141 recordings of the held-out replay. *Any mismatch blocks submission.*
  - **A.** The moved implementation reproduces the published held-out and
    regression replays run for run.
  - **B.** The controller with every arm off reproduces every recording
    through its recorded stop.
  - **C.** The controller with the arm on measures exactly what the published
    strict arm measured. Its divergent frames, measurements and depth events
    must match field for field.
  - **D.** It changes a cut decision only in runs where that arm diverged.

  "Exactly" means discrete fields exact and floats within 1e-6. OpenCV and
  OpenBLAS kernels differ in the last bits between CPU models.
- **C1 (the shadow events are accepted live).** In each of the 6 low-sun runs,
  the depth test accepts the first grey-check failure inside the shadow
  window: evening frames 56–67, morning 72–75. *Refuted if* it rejects one. A
  run with no grey failure in its window leaves C1 untested for that run, and
  this is reported.
- **C2 (15004 continues and passes).** Each of the 4 15004 runs (evening and
  morning, two repeats each) is judged on two things: whether it stops on
  appearance in its shadow window, and whether it passes all 17 checks of the
  unchanged grader.
  - Supported if at least 3 of the 4 pass and none of the 4 stops on
    appearance in its shadow window.
  - *Refuted if* at least 2 of the 4 stop in the shadow window, or fewer than 2
    pass.

  The no-shadow arm of the counterfactual passed 4 of 4. Here the shadow
  stays, and the margin is modest: the offline strict confidences are 0.19 to
  0.30 against the 0.15 gate.
- **C3 (evening 14944 still stops).** In both evening 14944 runs, the strict
  confidence at the accepted shadow event is below 0.15, so the run stops
  there on `low_confidence`. *Refuted if* either continues past the event, or
  stops there with the depth test rejecting.
- **C4 (real stops are kept).** Neither control passes. In both control runs,
  the depth test rejects every grey-check failure.
  - 19444 should stop when the open jaw reaches the patch.
  - 12142 should stop at the wire.

  *Refuted if* either control passes, or if the depth test accepts any grey
  failure in either control run. A control with no grey failure leaves the
  second clause untested.
- **C5 (no harm).** Source-light 14944 and 15004 both pass 17/17. *Refuted if*
  either fails a check. The test is expected never to act in these runs; any
  grey failure there is reported.
- **C6 (live equals offline).** For every run, the strict arm of
  `tools/replay_depth_appearance.py` replays that run's own recorded frames.
  On every frame through the run's own stop, it equals the recorded live
  measurement, with the same depth event at every grey failure. *Refuted by*
  any mismatch.

Grading uses only `tools/validate_vision_sequence.py`, through
`tools/aggregate_eval.py --protocol` per batch. C1–C6 are scored by a
committed scorer written before the runs are read.

## What this cannot show

- **Perfect depth.** Ground-truth optical-Z, exact poses and an exact jaw
  model. A sensor needs its own noise margin. The study's noise check passed,
  but no live sensor was tested.
- **Two targets, two low suns.** These are the same scenes that motivated the
  check. Its thresholds were fixed on the earlier recordings of these scenes
  and held on new ones, but no other tree, light or occluder is tested.
- **Same-depth occluders** (ties, leaves, a wire touching the spur) were never
  recorded, so they remain untested.
- **Render noise.** RGB renders are not repeatable, so the strict confidence
  near 0.15 can move from run to run. A class change counts only if it
  repeats.

## Cost

10 runs of 200 frames, 45 minutes reserved each, one task at a time: **450
GPU-minutes reserved**. The user approved this on October 1. About 4.7 GB of
captures go on the hpc-share.

## Result — October 2, 2026

All 10 runs recorded on RTX 8000 (`cn-gpu5`) and used 256 of the 450
reserved GPU-minutes. `tools/validate_vision_sequence.py` graded the 7
batches, through `aggregate_eval.py`. `tools/score_depth_loop.py` scored them:
- it was committed before submission (`1bd1ea1`);
- it was amended twice after blind reviews, before any grade file was opened
  (`0e6d92a`, `8acbf19`);
- its construction notes were corrected at `3ac43cb`, with no logic change;
- it ran from a clean clone at `45763d0`.

Evidence: [verdicts](evidence/depth_loop_verdicts_2026-10-02.json) and the
per-batch [grades](evidence/depth_loop_2026-10-01/).

| | Registered | Measured | Verdict |
|---|---|---|---|
| **C0** | the replay gate, before submission | [passed](evidence/depth_loop_c0_2026-10-01.json) on 141 recordings | passed |
| **C1** | the first grey failure in each low-sun window is accepted | 6 of 6 accepted (evening frame 67, morning frame 75) | supported |
| **C2** | at least 3 of 4 15004 runs pass, and none stops on appearance in its window | 4 of 4 pass 17/17; none stops | supported |
| **C3** | both evening 14944 runs stop at the accepted event on low confidence | both stop at frame 67; strict confidence 0.0005 and 0.045 | supported |
| **C4** | neither control passes, and every grey failure in them is rejected | 19444 stops at 70, rejected on `J_jaw_silhouette` and `4_median_abs`; 12142 stops at 29, rejected on `3_near_fraction` and `4_median_abs` | supported |
| **C5** | source-light 14944 and 15004 pass 17/17 | both pass; no grey failure | supported |
| **C6** | live equals the offline strict arm through each run's stop | 10 of 10 equal; floats within 6.5e-7 | supported |

**Reading.** Every result carries the label *depth-aware appearance check
D_strict + J; changes the 0.35 appearance gate's rule; simulator depth*.
- **Shadow stops on 15004 became passes in 4 of 4 runs.** With the shadow
  kept and the unchanged gate, all 6 low-sun runs had stopped at these frames
  ([jaw-shadow counterfactual](EVAL_PROTOCOL_JAW_SHADOW_2026-09-30.md#result--october-1-2026)).
  The strict confidence at the event was 0.18–0.20 against the 0.15 gate: the
  modest margin the protocol expected.
- **Evening 14944 still stops at the same frame**, as predicted. The depth test
  accepts the event, but the shadow leaves almost no correlation.
- **Real stops are kept.** The depth test rejects the jaw (19444) and the wire
  (12142) on the conditions the protocol expected, and both controls stop.
- **Nothing changes at source light.** There is no grey failure, and both runs
  pass.
- **Live equals offline** in all 10 runs. The held-out replay's estimate held:
  the check continued past 4 of the 6 low-sun stops.

**Seen before scoring.** The Slurm states of all ten runs were seen before
the scorer's amendments. The runner exits 0 only on a 17/17 pass, so those
states revealed:
- C2's pass count;
- C5's outcome;
- C4's pass clause;
- that both evening 14944 runs failed.

They also suggested C1 for the four 15004 runs. The scorer's
`construction_notes` disclose this; they were corrected at `3ac43cb`. An
earlier scoring run at `0282145`, which carried the old wording, was never
opened.

**Limits.**
- The depth is simulator ground truth. A real depth sensor's noise and holes
  would need their own margins.
- The shadow cases are one tree (tree1) under two lights, with two repeats
  each.
- 15004 clears the gate by only 0.03–0.05.
- Evening 14944 is unchanged. Passing it needs something other than this
  check.
