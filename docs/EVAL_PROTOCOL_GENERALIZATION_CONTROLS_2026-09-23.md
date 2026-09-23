# Why the learned depth fails away from its training cell — controls, September 23, 2026

This protocol is registered **before** the batch is submitted. The
[matrix](EVAL_PROTOCOL_FAMILY_LIGHTING_2026-09-23.md#result--september-23-2026)
showed the frozen DA2 and DINO models lose 6–8× accuracy under the evening
preset in all eight trees, and the
[anchoring analysis](EVAL_PROTOCOL_DEPTH_ANCHORING_2026-09-23.md#result--september-23-2026)
showed that neither evening nor the Isaac working distance is a scale problem.
Both results say *where* the models fail. Neither says *why*, and the why
decides what to change: the input pipeline, the training data, the camera, or
the depth head. This batch changes one axis at a time from the matrix `source`
cell and predicts each outcome in advance.

## What is already known, from three read-only audits

- **Training.** The checkpoint (`5ecc5182…`) was fine-tuned on 68,400 Envy
  renders under one fixed lighting with **no photometric augmentation** at all
  (resize, crop, horizontal flip only). The metric head was trained from scratch
  behind a sigmoid scaled to 20 m; supervised tree pixels sit at roughly
  1.1–2.6 m. About 91% of the original training frames are gone from disk;
  6,270 remain. Regeneration is possible but not bit-identical.
- **Camera.** Training frames and the matrix share the same projection: 28 mm on
  a 36 mm sensor, HFOV 65.5°, level view. The Isaac wrist camera is 20 mm on a
  30 mm aperture at 480×320 (fx = fy = 320 px, HFOV 73.7°), sits at
  0.16–0.39 m from the spur, and looks **up by 39.7°** in every Stage A frame.
  The metric head takes no camera input; the focal mismatch alone bounds the
  Isaac over-estimate at 1.38×, but the measured ratio is 3.8× and it is not a
  constant factor, so range extrapolation and appearance dominate.
- **Anchoring.** One metric range fixes UFO daylight and not Envy; the all-GT
  affine ceiling is 0.035–0.06 m in Blender daylight, 0.10–0.13 m at evening,
  0.37–0.46 m on Isaac. A relative head plus a many-point metric fit is the
  candidate architecture, untested.

## Design

All eight registered matrix trees (four Envy, four UFO) are re-rendered under
ten conditions, six views each except the sweep, 62 frames per tree, by
`tools/render_generalization_controls.py` with the same geometry, seed, texture,
sampler and Filmic tone mapping as the matrix. Definitions are fixed in
`tools/generalization_controls.py`; the launcher writes them into the frozen
plan and the renderer reads them back.

**Lighting axis** (far rig, training camera; the `source` cell is the baseline):

| Condition | What changes | Corner |
|---|---|---|
| `source` | nothing; determinism check against the matrix | bright, overhead sun |
| `evening` | companion preset; determinism check against the matrix | dark, 12° sun, warm |
| `evening_x2.6` | evening with sun energy and world strength × 2.6 | brighter, 12° sun |
| `overcast` | companion preset: same world as source, no sun | bright, diffuse |
| `overcast_div2.6` | overcast with world strength ÷ 2.6 | darker, diffuse |

The factor 2.6 is the matrix's source-to-evening mean-luma ratio (142.3 / 54.9),
fixed here before rendering. Filmic is not linear, so the achieved luma of each
cell is measured by the planner and recorded in the plan; in the CPU smoke
render `evening_x2.6` and `overcast_div2.6` land within two luma levels of each
other, which makes them a matched-brightness pair that differs only in whether
the light is a low sun with cast shadows or a diffuse sky.

**Test-time photometric normalization** (derived in the evaluation job from
this batch's plain `evening` renders by `tools/photometric_normalization.py`;
the `evening` cell is the baseline): `exposure` (linear gain to the source
mean luma, clipped), `gamma` (power law solved to the same target), `hist_eq`
(global luma equalization) and `clahe` (clip 2.0, 8×8, the setting the tracker
experiment rejected on September 19). Parameters are fixed; nothing is tuned.

**Camera axis** (`source` light):

| Condition | Camera | Poses | Baseline |
|---|---|---|---|
| `source/far/isaac_wrist` | Isaac wrist model (20 mm / 30 mm, 480×320) | matrix rig | `source` |
| `source/close/training_rig` | training camera | 0.16, 0.25, 0.39 m from the spur, level, ±0.02 m | `source` |
| `source/close/isaac_wrist` | Isaac wrist model | same | `source/close/training_rig` |
| `source/close_up40/isaac_wrist` | Isaac wrist model | same distances, optical axis pitched up 39.7°, camera below the spur | `source/close/isaac_wrist` |
| `source/sweep/training_rig` | training camera | one centred pose at 0.16, 0.25, 0.39, 0.6, 0.9, 1.3, 1.8, 2.3 m | `source` |

Depth and mask are rendered once per (pose set, camera model) and shared by the
lighting cells, so a lighting effect can never be a geometry difference.

**Models.** DA2 metric (`5ecc5182…`) on every frame with saved predictions.
Six-view DINO (`2750c4d8…`) only on far-rig training-camera cells, where its
rig assumption holds; the other cells are DA2-only by design and say nothing
about the refiner. The public **relative** DA2 ViT-L (`a7ea19fa…`, the
checkpoint whose backbone the fine-tune started from) runs on the same frames
plus the pinned matrix (192) and Isaac Stage A (600) frames; its disparity is
scored only through a per-frame all-GT affine fit in disparity space, which is
a ceiling and never a result.

**Metric.** Tree-mask MAE in metres per model × family × condition, paired
within tree against the registered baseline (`generalization_controls.BASELINE_OF`,
inserted as a table so the pairing cannot be chosen afterwards); the mask-gated
target error is the target-level metric; the sweep reports predicted and
reference target depth side by side. All numbers come from `sql/controls/*.sql`
through `tools/aggregate_controls_eval.py`. The pre-registered September 20
gates are evaluated per condition, unchanged. N per cell is 8 trees × 6 views
(8 × 8 for the sweep); a missing render or condition stays in the denominator
and is listed in the plan.

## Pre-registered predictions

Ratios are of per-tree tree-mask MAE against the condition's baseline, DA2.

- **P1 (brightness alone does not explain evening).** `evening_x2.6` stays
  above **3× source** in at least 6 of 8 trees, and `overcast_div2.6`, at about
  the same luma, stays under **2× source** in at least 6 of 8 trees: what the
  model cannot read is the low sun's shadow and colour structure, not the
  darkness. *Refuted if* `evening_x2.6` comes under 2× source in ≥ 6 trees, or
  `overcast_div2.6` goes above 3× source in ≥ 6 trees, which would make
  brightness the cause and a training-side exposure fix the lever.
- **P2 (test-time normalization does not rescue evening).** No variant brings
  the evening tree-mask MAE under **2× source** in ≥ 6 of 8 trees, for the same
  reason as P1; `gamma` and `exposure` do better than `hist_eq` and `clahe`.
  *Refuted if* `exposure` or `gamma` reaches under 2× source in ≥ 6 trees,
  which would put an exposure step in front of the model at runtime.
- **P3 (overcast is close to source).** `overcast` stays under **1.5× source**
  in ≥ 6 of 8 trees: removing the overhead sun at similar brightness changes
  little, because the training world was mostly diffuse. *Refuted if* it
  exceeds 2× source in ≥ 6 trees.
- **P4 (the wrist camera at matrix distance is a bounded focal effect).**
  `source/far/isaac_wrist` shifts the tree-mask signed median **more positive**
  than `source` by 0.2–0.9 m (a scale of roughly 1.2–1.5 on the prediction) in
  ≥ 6 trees. *Refuted if* the signed medians agree within 0.1 m (no focal
  effect) or the ratio exceeds 2 (something beyond focal).
- **P5 (close range reproduces the Isaac failure in Cycles).** In
  `source/close/training_rig` the mask-gated target prediction is **≥ 2× the
  reference** at all three distances (predicted 0.6–1.2 m against 0.16–0.39 m),
  and the sweep's target ratio crosses 1 between **0.6 and 1.3 m**. The Isaac
  camera model and the upward pitch each add **≤ 0.15 m** target MAE on top.
  *Refuted if* the close-range target MAE is ≤ 0.10 m at any distance, which
  would clear range and point at the Isaac renderer, assets or tool geometry,
  making the deferred orchard-tree0 replay the next experiment.
- **P6 (the metric head, not the backbone, is the close-range failure).** The
  relative head's all-GT disparity-affine ceiling on Isaac Stage A is **under
  0.15 m** (tree-mask ceiling of the metric head there: 0.37–0.46 m), and on
  the close Cycles cells under 0.10 m; on matrix daylight it is within 0.02 m
  of the metric head's ceiling. *Refuted if* the Isaac ceiling stays above
  0.30 m, which would mean the backbone does not resolve branch structure at
  74° HFOV and 0.3 m and a relative-plus-anchor architecture would not help.
  Evening is registered as a question, not a prediction: a relative ceiling
  under 0.08 m there would place the evening failure in the fine-tuned head.
- **P7 (determinism).** The re-rendered `source` and `evening` cells reproduce
  the matrix per-tree tree-mask MAE within **0.01 m** on every tree; OptiX is
  not bit-stable, so bytes are not compared.

If P1, P2 and P5 hold, the conclusion is: **evening needs rendered lighting
variation in training, not an exposure step; the Isaac gap is the training
range distribution and the metric head's output floor, so the architecture for
the approach is a relative head anchored by the rig's range sensor, and the
final approach below 0.4 m belongs to the range sensor alone.** If P6 fails,
close range needs new training data at that camera and distance, whichever head
is used.

## What this cannot show

- Nothing about the Isaac renderer, assets or tool geometry: every control is a
  Cycles render. Agreement with the Isaac cell is evidence about range and
  camera, not about RTX. The original orchard tree rendered through Cycles at
  the recorded wrist poses is the renderer control; it is deferred because the
  Isaac-to-Blender world mapping has not been derived and verified.
- Nothing about hardware: the wrist camera model is the simulation's, the rig's
  camera is uncalibrated, and no range sensor is simulated here.
- Nothing about closed-loop control, and nothing about DINO outside its rig.
- The relative-head numbers use ground truth to fit and are never achievable.
- Every Envy tree was in the fine-tune's train or validation split; only the
  four UFO trees and the Isaac frames are transfer checks.

## Cost and submission

Render array of 8 tasks at 15 min reserved each (matrix trees took 53–62 s for
24 frames; 62 frames and 38 geometry passes here) and one 60 min evaluation
job: **180 GPU-minutes reserved, about 40 expected, one GPU at a time.** Output
about 1.5 GB on the share against a 2.5 GB preflight estimate. Submitted only by
`tools/queue_generalization_controls.py --submit` from a clean committed tree
after this protocol is committed, with the storage preflight and the user's
approval recorded in the batch directory.

Not queued, and why: a re-fine-tune with photometric jitter on the 6,270
surviving frames (about 8 A40-hours) needs a training wrapper in this
repository, because the companion trainer has no version control and the
launchers treat that tree as read-only; whether it is the right lever depends on
P1 and P2. A regenerated training set with rendered lighting variation is
storage-bound (about 720 GB at the original resolution against 85 GB free).
