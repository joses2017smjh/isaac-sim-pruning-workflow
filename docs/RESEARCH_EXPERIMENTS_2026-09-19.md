# Outdoor visual-servo robustness pilot — September 19, 2026

September 20 result: all six array `21360571` tasks completed and pass fresh
17/17 sequence checks. Raw and CLAHE each succeed under source/morning/evening
lighting on the same original tree0 target. This does not establish a CLAHE
advantage or Envy/UFO/learned-depth generalization.
[Reconciled results and next research phase](RESEARCH_AUDIT_2026-09-20.md) ·
[machine evidence](evidence/repository_audit_2026-09-20.json).
The protocol and execution record below describe the September 19 checkpoint.

**September 23 decision: CLAHE is dropped.** See
[Decision on the CLAHE candidate](#decision-on-the-clahe-candidate) below and
[the regenerated pilot results](evidence/lighting_pilot_results_2026-09-23.json).

This batch tests whether fixed local contrast normalization helps the existing
seeded RGB-D tracker under lighting changes. It compares the current controller
with an opt-in CLAHE variant, first on saved captures and then in six complete
Isaac sequences. CLAHE is a diagnostic hypothesis, not an established gain or a
state-of-the-art pruning method. Both improvements and regressions count as
results.

The scope ends at preparing, testing, documenting and queueing these experiments.
Existing HPC jobs must remain intact. The six new GPU tasks run through a serial
array (`%1`); the experiment does not require waiting for GPU completion.

## Why this experiment

The successful two-tree recording and earlier closure failure expose a concrete
weakness: the tool can lose usable image features near the branch. The current
tracker already uses forward/backward pyramidal Lucas–Kanade checks, patch
appearance checks, optical-Z depth validation and same-depth feature maintenance.
Its accepted capture is evidence for one initialized target and trajectory, not
for general outdoor robustness.

Local contrast normalization could retain usable texture under dark patches or
exposure transitions. It can also amplify image noise and change local appearance
between frames, potentially worsening LK correspondence or confidence. The
unchanged raw-grayscale tracker is therefore an essential paired control. The
choice of `clipLimit=2.0`, `tileGridSize=(8, 8)` comes from the official OpenCV
example and is fixed before this batch; it is not tuned to these recordings.
The grid denotes eight tiles along each image dimension, not 8-pixel tiles.
[OpenCV tutorial](https://docs.opencv.org/4.13.0/d5/daf/tutorial_py_histogram_equalization.html)
and [CLAHE API](https://docs.opencv.org/4.13.0/d6/db6/classcv_1_1CLAHE.html),
version 4.13.0 documentation generated December 31, 2025, accessed September 19,
2026. The experiment records the actual installed OpenCV version separately.

## Fixed methods and controls

| Method | Image preprocessing | Other behavior |
|---|---|---|
| Raw baseline | Existing RGB-to-grayscale conversion | Current feature maintenance and all tracking/control gates |
| CLAHE candidate | Grayscale followed by CLAHE, clip limit 2.0, tile grid 8×8 | Same configuration and gates as baseline |

The baseline includes `feature_quality_level=0.005`, `depth_radius_px=1` and
`replenish_features=True`. These existing settings are not credited as new
improvements. CLAHE applies consistently to initialization and subsequent image
processing. Saved RGB and depth remain available for independent inspection.

Retain the existing tracking-confidence, depth, freshness, contact, closure and
geometry thresholds. Tracking loss remains latched until explicit initialization;
neither normalization nor replay provides automatic oracle reseeding. Missing
vision/depth must not authorize approach or release. Post-release home-directed
retreat remains a separate phase where target tracking is no longer required.

## CPU replay protocol

Replay the saved successful two-tree capture and the saved closure-failure
capture. Each method receives identical initial data, camera transforms, frame
order and corruption schedule. Run these paired conditions:

- Unmodified capture as a reference.
- Exposure transitions to challenge photometric consistency.
- RGB blackout as a negative control for tracking and motion authorization.
- Depth dropout as a negative control for metric position authorization.

The generated replay report must state the exact corruption parameters and frame
intervals, input paths, method settings and software versions. Transformations
are synthetic stress tests, not measured outdoor camera response. Evaluate
pre-release tracking separately from the moving detached piece and retreat.

Report accepted-frame coverage and time/frame of first tracking loss alongside
available position-error metrics. Reporting error only on surviving frames would
favor a method that rejects difficult observations. Record rejection reasons,
feature counts and confidence, and whether any invalid observation is accepted
for a command. Confidence is a heuristic score, not calibrated uncertainty.

When a reference position is available, label its origin and report the error
only over its valid interval. A mesh cut point, an initial visible surface point
and a tracker-derived reference are different references; none should silently
be substituted for another. Replay follows the trajectory recorded by the
original controller. It cannot establish that a different controller would
complete the task in closed loop.

## Six full GPU sequences

| Lighting | Raw baseline | CLAHE candidate |
|---|---|---|
| Source scene light | One 200-frame / 20-second sequence | One matched sequence |
| Morning preset | One 200-frame / 20-second sequence | One matched sequence |
| Evening preset | One 200-frame / 20-second sequence | One matched sequence |

Use the same selected branch, initialization, robot, camera, original two-tree
assets, physics settings, rendering quality and capture cadence within each pair.
Only the tracker preprocessing and named lighting preset vary. There are no
additional seed sweeps, alternate targets or branch-placement experiments in this
batch. The serial array limits this new batch to one running GPU task at a time;
it does not change or cancel existing jobs.

Evaluate complete sequences using the independent task grader, including its 17
checks for approach, release, measured fall and return. Preserve capture checks
separately: complete images and scheduler exit status do not prove task success.
Record release time, first failure reason, applied command count, tracking-loss
time, post-release drop and final home error. Retain failed captures and the
method/configuration identity with every report.

The primary comparison is whether the candidate preserves the source-light
behavior and improves task completion or tracking continuity under a changed
lighting preset while all unchanged gates remain satisfied. A blackout/dropout
control that permits an invalid approach or release is a regression. A candidate
that holds more often may be more conservative but is not automatically more
useful. Six single-target trials provide a pilot comparison, not a population
success rate or statistical evidence of superiority.

## Research basis and limits

- **Ahmed et al., April 9, 2025; revised June 5, 2025:**
  [An Integrated Visual Servoing Framework for Precise Robotic Pruning Operations
  in Modern Commercial Orchard](https://arxiv.org/abs/2504.07309).
  A UR5e/RGB-D/CoTracker3 controller motivates a future learned point-tracking
  comparator. Its reported validation is in Gazebo; real-world implementation
  and force-sensed cutting are future work. It does not establish field success
  or support assuming that CLAHE helps this repository.
- **Karaev et al., October 15, 2024; ICCV 2025:**
  [CoTracker3: Simpler and Better Point Tracking by Pseudo-Labelling Real
  Videos](https://arxiv.org/abs/2410.11831).
  Online and offline learned trackers provide a useful later comparison for
  occlusion and appearance changes. Offline future-frame access must remain
  distinct from causal control; predicted occluded locations alone must not
  authorize cutting.
- **Strohbehn and Grimm, Robotics and Autonomous Systems 203, September 2026:**
  [Branch pruning alignment using small form factor monocular time-of-flight
  sensors](https://doi.org/10.1016/j.robot.2026.105513).
  Physical lab/orchard experiments motivate complementary close-range sensing
  when the pruner blocks camera views and identify adverse lighting as a
  limitation. Their dual single-beam hardware differs from this simulator's
  dual 8×8 ray casters; their results do not calibrate this sensor model.
- **Gebrayel et al., accepted January 31, 2026:**
  [Point Cloud Based Visual Planning and Servoing for Autonomous Vine
  Pruning](https://tomobrien.vercel.app/papers/Vine_Pruning_MPC.pdf).
  Visibility-aware waypoint planning and ICP visual servoing address motion and
  viewpoint disturbances, including outdoor wind experiments. This motivates
  later dynamic-scene and viewpoint studies; this batch implements neither their
  planner nor a wind model.
- **STMicroelectronics, DS14161 revision 12, July 22, 2025:**
  [VL53L8CX datasheet](https://www.st.com/resource/en/datasheet/vl53l8cx.pdf).
  Range performance depends on ambient illumination, reflectance, resolution and
  zone. The datasheet's full-field target conditions do not establish a
  thin-branch Gaussian noise model. RTX sun changes do not model ambient-IR
  interference in ray-cast ToF: this batch evaluates rendered RGB lighting,
  not calibrated ToF sunlight robustness.

All links above were checked September 19, 2026. Research motivates the problem
and comparisons; it does not substitute for the queued results.

The camera remains simulation-defined, depth is RTX optical-Z ground truth, and
branch identity/axis/radius come from metadata. Lighting presets are artistic,
not radiometrically or astronomically calibrated. The task uses a visual jaw
surrogate and discrete rigid-piece release, not physical blade mechanics or wood
fracture. This experiment does not validate learned recognition, PPO training,
executed CuRobo planning, calibrated physical sensors or collision avoidance.

## Decision on the CLAHE candidate

Completed September 23, 2026, against the primary comparison fixed above:
*does the candidate preserve the source-light behaviour and improve task
completion or tracking continuity under a changed lighting preset, with all
unchanged gates still satisfied?*

| Criterion | Raw baseline | CLAHE candidate | Verdict |
|---|---|---|---|
| Source-light behaviour preserved | 17/17, 68 commands | 17/17, 68 commands | Preserved |
| Task completion under changed light | 2/2 sequences pass | 2/2 sequences pass | No improvement |
| Tracking continuity under changed light | 78 and 78 pre-release frames | 78 and 77 | No improvement |
| Blackout/dropout controls | No invalid approach or release | No invalid approach or release | No regression |

**CLAHE is dropped as a default and recorded as a rejected diagnostic.** The
`photometric_normalization` option stays in the tracker configuration so the
comparison remains reproducible; nothing selects it. Six single-target trials
cannot establish superiority in either direction, and this is a negative result
published on the same terms a positive one would have been.

The CPU replay agrees: across all eight paired saved-capture conditions the two
modes accept identical frame counts, so there is no measured tracking-continuity
gain. CLAHE's median centroid error is 0.3–0.7 mm lower, which is a measurement
difference on surviving frames, not a task outcome.

### What did move, and why it is secondary

The lighting presets are not cosmetic. Over pre-release tracking frames the
minimum surviving feature count was 18 (source), **4 (morning)** and 13 (evening)
for the raw baseline. The tracker's own `min_features` is 4, tested as
`count < 4`, so **morning raw sat exactly on its accept floor**: one fewer
surviving feature would have rejected the frame. Minimum confidence in that run
was 0.1735 against a `min_confidence` of 0.15.

CLAHE raised the minimum feature count in all three presets (18→21, 4→7, 13→15).
It raised the minimum confidence under source (0.581→0.621) and morning
(0.1735→0.495) but **lowered it under evening** (0.500→0.403). The effect is
therefore not a uniform margin improvement, and margin is not the pre-registered
metric. It is reported because it shows the pilot was a real stress and that
source light is not the hard case, which is why morning light belongs in any
broader target sweep.

## Follow-up after this batch

Use the completed reports to decide whether CLAHE merits further evaluation or
should remain a rejected diagnostic. Broaden targets and initial views before
estimating task success rates. A separate depth-consistency study should inspect
surviving-feature surface membership, accepted-position drift and the mismatch
between heuristic confidence and metric error. A later CoTracker3 replay baseline
should include runtime/latency and causal-input checks before live integration.
Sensor-noise and ToF ambient-light work require separately documented assumptions
and, eventually, physical calibration data.

## Execution record

The [CPU aggregate](evidence/visual_robustness_replay_2026-09-19.json)
records 16 paired conditions, 3,200 attempted saved-image updates, full input/code
hashes, and OpenCV 4.11.0 / NumPy 2.2.6 / Python 3.10.19. Both modes use the
**current** demo configuration, including feature maintenance, for both captures.
Historical capture configurations are recorded separately; the CLI option
`--config-policy recorded` is a separate historical diagnostic.

| Saved capture / stress | Raw | CLAHE | Interpretation |
|---|---|---|---|
| Successful sequence, unmodified | 78/78 pre-release frames | 78/78 | Both stop tracking the moving released piece afterward |
| Closure-failure sequence, unmodified | 200/200 frames | 200/200 | 123 frames occur after the recorded controller stopped; no rescued task claim |
| Abrupt exposure transition, either capture | Lost at frame index 10 | Lost at index 10 | Neither method handles the gain-0.35 transition |
| RGB blackout, either capture | Lost at index 10 | Lost at index 10 | Missing texture rejects target tracking |
| Depth dropout, successful capture | 69/78 pre-release frames | 69/78 | Missing depth rejects targets; tracker can recover |
| Depth dropout, failed capture | 179/200 frames | 179/200 | Identical accepted-frame coverage |

There is **no measured tracking-continuity improvement** from CLAHE in this
replay. The full lighting comparison remains a bounded test of different rendered
illumination, not a promotion of the candidate. Shared-CPU wall times are stored
for diagnostics and do not establish deployment latency.

Reproduce replay with the same local captures (write to a new output path):

```bash
python tools/benchmark_visual_robustness.py \
  --capture-dir artifacts/isaac_render/job_21328323 \
  --capture-dir artifacts/isaac_render/job_21317409 \
  --output /tmp/pruning-visual-robustness-new.json
```

Independent September 19 regrading also confirms the source success (17/17)
and both three-second daylight probes (11/11 capture checks each, no release).
Local verification: **514 passed, 9 skipped, 1 simulator-only deselected**;
Ruff lint and format checks pass across pruning source/tests/tools and the
renderer (117 files). The nine skips require USD support absent from the CPU
interpreter; GPU integration remains pending. Shell syntax and submission dry
run pass. External asset inventory verifies 25 files, including six scene
textures and all imported robot outputs. No new GPU outcome is established yet.


Array **`21360571_[0-5%1]`** was accepted September 19 at 16:12 PDT; all six
tasks were pending at the 16:13 PDT check (reason `(None)` at that snapshot).
[Submission evidence](evidence/vision_pilot_submission_2026-09-19.json) records
resources, conditions and hashes. Frozen code revision is
`b4b4250fae14df0c1f39c70f60cf21960bae90ba`; later documentation commits do not
change queued code. The batch lives at
`artifacts/vision_robustness/lighting-20260919/`; inspect its per-run grades after
Slurm executes it. No existing job was modified. The portable CPU demo also
reproduced its success, sensor-loss and blocked-geometry outcomes. Work stopped
at queueing and pushing this checkpoint, without waiting for GPU completion.

## Authorized research execution (2026-09-20)

Implementation and jobs are now in progress; the preceding audit-only snapshot is historical. Frozen DA2 offline evaluation job **21370005** submitted, no dependencies. The one-frame CPU accuracy gate failed; learned control remains conditional. Checkpoint-specific leakage corrections, current job/result states and storage are tracked in the [execution record](RESEARCH_EXECUTION_2026-09-20.md) and `docs/evidence/research_execution_2026-09-20.json`.
