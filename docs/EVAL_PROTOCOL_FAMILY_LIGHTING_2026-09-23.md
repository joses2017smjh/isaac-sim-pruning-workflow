# Learned depth across tree families and lighting — protocol, September 23, 2026

This protocol is registered **before** the matrix is submitted. It fixes the
trees, the lighting presets, the models, the metric and the gates, so the result
cannot be shaped after it arrives. It also records what is already known from
the September 20 pilots, regenerated from disk by a committed tool, so the
matrix is judged against a stated expectation rather than a blank slate.

## The question, split into the two things "the model" means

The controller's branch tracker is classical pyramidal Lucas–Kanade. It is not
learned and it does not appear here.

The learned models are the metric-depth networks from the depth-estimation
project: **DA2** (Depth Anything V2 metric ViT-L, fine-tuned on Envy renders)
and **DINO** (a six-view DINOv2 refiner that takes DA2 depth as input). The
question this protocol answers is whether *those* carry over

1. from Envy, the family they were fine-tuned on, to UFO, and
2. from the source lighting to morning, noon and evening presets,

measured offline on rendered frames against simulator ground-truth depth.
Whether the closed-loop controller works on Envy or UFO trees is a separate,
larger question; see the last section.

## What is already known, and regenerable

Two September 20 runs were never written up. Their numbers below are produced
by `tools/aggregate_family_eval.py` through `sql/family/*.sql`, not copied from
a notebook.

**Stage A — DA2 on Isaac RTX frames of the original orchard tree**
([evidence](evidence/stage_a_depth_2026-09-23.json), job `21370005`, 600 frames,
234 with a tracked target). Every pre-registered gate failed on every frame:

| Light | Target MAE (median) | Signed error (median, full frame) | Latency p95 |
|---|---|---|---|
| source | 0.575 m | +0.410 m | 0.248 s |
| morning | 0.605 m | +0.428 m | 0.249 s |
| evening | 0.866 m | +0.620 m | 0.248 s |

The target was **0 of 234** frames within 20 mm, and 0 of 234 within 10%
relative error. The error is a systematic overestimate, not noise: the model
predicts the branch too far in every frame. This does **not** isolate a
Blender-to-Isaac renderer gap, because four things change at once against the
training renders: the working distance (0.16–0.39 m here; training trees sit at
roughly 1–2.6 m), the camera intrinsics and resolution, the tree asset and bark,
and the lighting model.

**Pilot — DA2 and DINO on Blender Cycles renders, one tree per family**
([evidence](evidence/family_pilot_depth_2026-09-23.json), job `21370039`,
48 frames). Tree-mask MAE, metres, N = 1 tree × 6 views per cell:

| Model | Family | source | morning | noon | evening | evening ÷ source |
|---|---|---|---|---|---|---|
| DA2 | Envy 00000 | 0.143 | 0.196 | 0.159 | **0.864** | **6.1×** |
| DA2 | UFO 00000 | 0.184 | 0.302 | 0.246 | **1.449** | **7.9×** |
| DINO | Envy 00000 | 0.145 | 0.214 | 0.164 | 0.877 | 6.0× |
| DINO | UFO 00000 | 0.194 | 0.321 | 0.260 | 1.292 | 6.6× |

Three things are visible even at N = 1. Almost all of the error is a constant
under-estimate (signed median ≈ −MAE in every cell); with each frame's own
offset removed the residual is 0.03–0.06 m under source, morning and noon.
Evening breaks both models. And DINO adds nothing over DA2 (worse in 7 of 8
cells). DA2's inference p95 was 0.29 s against the 0.10 s gate; DINO's six-view
total was 2.2 s. Neither model passes any gate on any frame.

**What N = 1 cannot say.** Whether UFO is worse than Envy, or whether the tree
happens to be harder. Whether the evening effect is a property of the family or
of one geometry. That is what the matrix is for.

**Two problems in the pilot's target metrics, and their fix.** The target
window was 3×3 pixels unmasked on a spur about one pixel wide at 512×288, so it
mostly scored background; and the pilot's visibility annotations carried a
half-pixel error (8 of 48 marked visible; 44 after correction). The evaluation
now also records a **mask-gated target error** that scores tree pixels only, and
`prepare_family_evaluation` applies the half-pixel correction when it builds the
plan. The unmasked metric is kept so September 20 results stay comparable. The
matrix registers the masked metric as its target-level measure **before
running**; the thresholds it is judged against are unchanged.

## Registered design

**Trees.** Fixed on September 20 by the provenance audit, not chosen after
seeing outcomes:

| Family | Trees | Exposure |
|---|---|---|
| Envy | 00000, 00003, 00008, 00012 | DA2 **validation** trees. Every one of the 100 Envy trees was in DA2 training or validation, so **Envy is not an unseen-tree test**. |
| UFO | 00000, 00001, 00002, 00003 | Absent from the task fine-tuning manifests. Backbone pretraining exposure is unknown. |

The two 00000 trees were rendered by the pilot and are reused; the other six
are rendered by this batch. A tree whose render fails stays registered and is
reported as incomplete. **N never shrinks.**

**Conditions.** Four Blender lighting presets (source, morning, noon, evening)
× six camera views (three rigs at z = 0.85, 1.00, 1.15 m, left and right at
±0.12 m) per tree; Blender 4.2.19 Cycles, 512×288, 16 samples, seed 1729,
Filmic; `orchard_template.blend` with `bark_brown_02`. Views and seed are
identical across lights, so the lighting comparison is paired. 8 trees × 24 =
**192 frames per model**. Every asset and script is hashed into the plan; the
source is frozen with `git archive` of the committed tree.

**Models.** DA2 checkpoint `5ecc5182…`, DINO checkpoint `2750c4d8…`, both
verified by hash at run time. No prediction is scaled, shifted or aligned to
ground truth.

**Primary metric.** Tree-mask MAE in metres per model × family × lighting cell,
with **n_trees** (4 per family) as the population count; views of one tree are
re-renders of one geometry and are not independent. Target-level: the
mask-gated target error. Both come from `sql/family/*.sql`.

**Gates.** The September 20 thresholds, unchanged: target p95 ≤ 20 mm, mean
relative error ≤ 10 %, coverage ≥ 99 %, inference p95 ≤ 100 ms. **Expected
outcome: fail.** The pilot failed all of them by a wide margin; the matrix
measures by how much and whether that holds across trees.

**Registered readings.**

- *Lighting effect replicated* if evening mask MAE is at least 3× source in at
  least 3 of the 4 trees of a family, for that family.
- *Family gap* is reported as the per-tree mean UFO − Envy difference under each
  light with its sign across trees. With four trees per family only a large,
  consistent gap is interpretable, and it still cannot separate "UFO" from
  "Envy trees the model was validated on".
- The debiased residual is reported as a diagnostic of the error's structure. It
  uses ground truth per frame and is never an achievable accuracy.

## Budget and resources

Measured on the pilots: 63 s and 55 s per tree render, 60 s to score 48 frames.

| Job | Tasks | Reserved | Expected actual |
|---|---|---|---|
| Render array `%1` | 6 | 6 × 15 min = 90 GPU-min | ~6 min |
| Evaluation (waits on the array, `afterany`) | 1 | 30 GPU-min | ~5 min |
| **Total** | | **120 GPU-min** | **~11 min** |

Storage: about 0.4 GB. One GPU at a time. A storage preflight is written into
the batch and the job scripts refuse to run without it. Submission requires an
explicit OK.

Regenerate the plan without submitting:

```bash
python tools/queue_family_matrix.py
```

## What this will not establish

- **Not a renderer-gap result.** Stage A confounds distance, intrinsics, asset
  and renderer. Two controls would separate them and are not in this batch:
  the original orchard tree rendered in Cycles at the same rigs, and a
  close-range rig at the Isaac working distance.
- **Not darkness versus sun angle.** Evening is one artistic preset. Blender has
  an `overcast` preset (diffuse, no sun) that would separate the two; it is not
  in the registered four, to keep the matrix comparable with the pilot.
- **Not like-for-like with the Isaac lighting pilot.** The Isaac and Blender
  presets are different definitions (evening azimuth 270° vs 315°, different
  intensity units), and the Isaac pilot had no noon.
- **Not an unseen-tree result for Envy**, and not a claim about UFO pretraining
  exposure.
- **Not closed-loop.** No Envy or UFO tree has been targeted by the vision-guided
  controller in Isaac. The renderer (`RenderEnv`) presents the original Blender
  orchard export; Envy and UFO are L-Py cylinder USDs with no spur-selection
  path into it. The USD imports pass (`21370047`), which means the stage opens
  with finite bounds, nothing more. A controller-on-Envy/UFO study needs target
  selection over cylinder metadata (the oracle in `geometry/cut_point.py`
  exists), a spawn path in `RenderEnv`, and its own registration. That is an
  engineering project, and it is not started here.
