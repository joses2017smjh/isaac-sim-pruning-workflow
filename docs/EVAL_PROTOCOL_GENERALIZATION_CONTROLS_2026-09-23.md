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
  0.15 m** (the metric head's all-valid ceiling there, Isaac having no tree
  mask: 0.37–0.46 m), and on
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

---

## Result — September 23, 2026

Render array `21403403` (8 tasks, 2.4 min each) and evaluation `21403404`
(12.5 min) COMPLETED (0:0) on cn-gpu5; 688 of 688 registered frames scored,
no missing condition; about 32 GPU-minutes used of 180. Code revision
`fbc446d`, all three checkpoint hashes matched the plan.
[Evidence](evidence/generalization_controls_2026-09-23.json) ·
[anchoring ceilings per condition](evidence/generalization_controls_anchoring_2026-09-23.json).
Achieved mean luma: source 142, evening 55, `evening_x2.6` 88,
`overcast` 129, `overcast_div2.6` 87.

### Tree-mask MAE, DA2, mean of per-tree values (n = 4 trees per family)

| Condition | Envy | ratio to baseline | UFO | ratio to baseline | Baseline |
|---|---|---|---|---|---|
| `source` | 0.134 | — | 0.188 | — | — |
| `evening` | 0.906 | 6.8× | 1.451 | 7.8× | source |
| `evening_x2.6` (luma 88) | 0.821 | 0.91 | 1.443 | 1.00 | evening |
| `overcast` (luma 129) | 0.170 | 1.27 | 0.247 | 1.32 | source |
| `overcast_div2.6` (luma 87) | 0.333 | 1.99 | 0.577 | 2.34 | overcast |
| `evening/gamma` | 0.637 | 0.70 | 1.290 | 0.89 | evening |
| `evening/exposure` | 0.850 | 0.94 | 1.357 | 0.94 | evening |
| `evening/hist_eq` | 0.785 | 0.87 | 1.271 | 0.88 | evening |
| `evening/clahe` | 0.727 | 0.80 | 1.437 | 0.99 | evening |
| `source/far/isaac_wrist` | 0.155 | 1.17 | 0.484 | 2.65 | source |
| `source/close/training_rig` | 0.527 | 3.95 | 0.385 | 2.10 | source |
| `source/close/isaac_wrist` | 0.529 | 1.03 | 0.410 | 1.07 | close/training_rig |
| `source/close_up40/isaac_wrist` | 0.455 | 0.87 | 0.412 | 1.01 | close/isaac_wrist |

Distance sweep, target predicted ÷ reference (training camera, source light):
Envy 4.4, 3.5, 2.3, 1.14, 1.11, 1.04, 0.96, 0.90 and UFO 4.0, 2.6, 1.8, 1.28,
1.17, 1.12, 1.05, 0.92 at 0.16, 0.25, 0.39, 0.6, 0.9, 1.3, 1.8, 2.3 m. The
prediction sits on a floor of 0.6–0.9 m below 0.4 m in every tree. All
pre-registered gates fail in every condition, as before.

### Verdicts on the registered predictions

- **P1 half supported, half refuted.** `evening_x2.6` stays 6.1–7.8× source
  in 8 of 8 trees: brightening the evening scene by the full luma ratio
  recovers almost nothing (ratio 0.91 Envy, 1.00 UFO to plain evening).
  But `overcast_div2.6` did **not** stay under 2× source: darkening a diffuse
  scene to the same luma costs 2.5× (Envy) and 3.1× (UFO). Darkness alone is a
  2–3× problem; the low sun's shadow and colour structure is the remaining
  factor of about 3 at matched brightness.
- **P2 supported; the ordering half refuted.** No variant brings evening under
  2× source in any tree (best: `gamma`, 4.8× Envy, 6.9× UFO). `gamma` is the
  best variant on Envy and second on UFO and cuts the Envy target error
  0.65 → 0.24 m, but `hist_eq` edges it on UFO and `exposure`, which clips
  highlights, is the weakest on Envy. No test-time curve is a fix.
- **P3 supported.** `overcast` is 1.05–1.46× source in 8 of 8 trees.
- **P4 family-dependent.** Envy behaves as the focal argument predicts: signed
  median −0.12 → +0.15 m, ratio 1.17. UFO does the opposite: −0.19 → −0.50 m,
  ratio 2.65 (1.5–4.4 across trees). The wrist camera at matrix distance is not
  a bounded focal effect on UFO; refuted there.
- **P5 supported with two misses.** The close-range failure reproduces in
  Cycles from range alone: target predictions 0.6–0.9 m against 0.16–0.39 m,
  ratio ≥ 2 at all three distances on Envy and at two of three on UFO (1.8 at
  0.39 m); target MAE 0.38–0.55 m, never near 0.10 m. The Isaac camera model
  adds 0.00–0.03 m and the upward pitch −0.07 to 0.00 m, both within the
  0.15 m bound. The ratio crosses 1 between 1.3 and 1.8 m, later than the
  registered 0.6–1.3 m.
- **P6 refuted at frame level, informative at the target.** The relative
  head's all-valid disparity-affine ceiling on Isaac Stage A is 0.80–1.39 m,
  not under 0.15 m; a whole-frame fit in disparity space is dominated by the
  2–9 m background pixels, where inverting a fitted disparity amplifies
  error, so this metric was a poor choice for Isaac frames and the registered
  reading fails on it. Restricted to the 3×3 target window the relative
  head's ceiling on Isaac is 0.019–0.040 m (metric head: 0.37–0.46 m over
  all valid pixels, target error 0.57–0.87 m), and on the close Cycles cells
  0.005–0.012 m. On the far Blender cells the relative head is worse than the
  metric head (ceiling 0.12–0.14 m against 0.04–0.05 m, correlation ≈ 0.1):
  the public head does not resolve 1-pixel spurs at 2 m. Evening is not
  answered: the relative ceiling there (0.12–0.13 m) equals its ceiling under
  source, so this head is lighting-indifferent but too coarse at that range.
- **P7 supported.** Re-rendered `source` and `evening` reproduce the matrix
  per-tree means within 0.001 m (Envy 0.1335 / 0.9062 against 0.1335 / 0.9059;
  UFO 0.1877 / 1.4506 against 0.1878 / 1.4506).

### A finding that was not predicted

The metric head's own all-GT affine ceiling on the close Cycles cells is
**0.011–0.035 m** with correlation 0.45–0.81: at 0.16–0.39 m the fine-tuned
model gets the *shape* right and only the scale and offset wrong, so a
many-point metric anchor would work there. On Isaac at the same distances the
metric head's ceiling is 0.37–0.46 m. Range therefore explains the Isaac
*magnitude* (the 0.6–0.9 m floor) but not the Isaac *shape collapse*. What
remains different between the two is the renderer, the tree asset and bark,
the tool jaws in view and the sky background. The deferred orchard-tree0
replay in Cycles at the recorded wrist poses is now the necessary experiment,
not an optional one.

### What this means for generalizing better

1. **Evening.** Not an exposure problem and not fixable in front of the model.
   Darkness costs 2–3×, low-sun shading another ~3×; both are absent from the
   training distribution, which had one fixed light and no photometric
   augmentation. The lever is training-side lighting variation, rendered (sun
   angle and colour), not jitter alone. The re-fine-tune is now justified and
   still not queued: it needs a repository-local trainer.
2. **Close range.** Range explains the floor; the metric head extrapolates
   to 0.6–0.9 m below 0.4 m in every tree. Two routes: close-range training
   poses in the generator, or a relative head anchored by the rig's range
   sensor at the target, where both heads' target-window ceilings are under
   0.04 m. Below 0.4 m the range sensor alone remains the metric estimator.
3. **Isaac.** The shape collapse is not range and not camera. The renderer
   control (orchard tree0, Cycles, recorded wrist poses) is next, and needs
   the Isaac-to-Blender world mapping derived and checked first.
4. **UFO.** The wrist camera hurts UFO and not Envy at matrix distance, which
   is the first camera effect that differs by family; it should be part of
   any close-range training set.

