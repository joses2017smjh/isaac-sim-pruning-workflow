# Render gap or lighting? — protocol, September 30, 2026

Registered **before** submission. The
[rendered-lighting result](EVAL_PROTOCOL_LIGHTING_TRAINING_2026-09-27.md#result--september-30-2026)
found two things nobody predicted. The new arms' source-light error on the
evaluation matrix fell 35–45%. Yet their validation RMSE on the companion's
own source-lit frames did not move. The two frame sets come from different
renderer settings:

| | Evaluation matrix | Training and companion frames |
|---|---|---|
| Resolution | 512×288 | 1920×1080 |
| Sampler | 16 fixed samples, OIDN denoised | the template's adaptive sampler, up to 4,096 samples (threshold 0.01), OIDN denoised |
| Cycles seed | 1729 | 0 |

The sky is not a difference. `daylight_presets.apply_preset` rebuilds the
world in both renderers, so the training renderer's world fix leaves no trace
in any frame. Lens, view transform, exposure and depth of field are equal. The
seed changes only the RGB noise pattern, never depth or mask.

Part of the matrix gain may therefore be robustness to the render gap rather
than to light. This batch separates the two.

## Design

The 8 matrix trees are re-rendered at the same camera poses, under the same
four presets and with the same tree placement, using the training renderer's
settings (`tools/render_family_lighting.py --renderer training`): 1920×1080,
with the template sampler asserted unchanged. In the same session, each view's
depth and tree mask are rendered once more on the 512×288 matrix grid with the
matrix sampler (16 samples, seed 1729). This grid pass exists only for the
geometry check below; its RGB is discarded and every changed setting is
restored and checked before the next frame.

Five checkpoints are scored by the frozen `tools/depth_generalization.py` in
two arms:

- **Matched.** The new 1920×1080 RGB is downsampled to 512×288 (area
  interpolation) and scored against the published matrix's own depth and
  mask. It differs from the published matrix only in how the RGB was rendered.
- **Native.** The new 1920×1080 RGB is scored against the new render's own
  1920×1080 depth and mask. It differs from Matched in resolution and in its
  ground truth.

The checkpoints are F (frozen), J and C (the published jitter and control
arms), and A and B (the rendered-lighting arms). Each is hashed in the plan and
must match the hash its own evaluation recorded. The metric, aggregation
(`tools/aggregate_family_eval.py`, `tools/anchor_depth_analysis.py`) and
per-tree means are exactly as in the published result.

## Pre-registered predictions

Ratios are per tree, on the Matched arm unless stated.

- **G0 (same geometry).** A tree enters Matched only if, for every view, its
  grid-pass depth and mask reproduce the published matrix depth and mask: at
  least 99.5% of tree-mask pixels within 1 mm and mask IoU at least 0.995
  (`training_lighting.agreement`, thresholds unchanged). Predicted: all 8 trees
  pass. *Refuted if* any tree fails; it is then excluded from Matched and
  reported. The 1920×1080 geometry point-sampled at matrix pixel centres is
  recorded as a measurement and never gates; 1920 is 3.75 × 512, so those
  samples never fall on matrix pixel centres.
- **G1 (the source gain is render-gap robustness).** On Matched, A's and B's
  source error is at least 0.85 × C's in at least 6 of 8 trees, which would mean
  the 35–45% gain shrinks to 15% or less once the frames are rendered like the
  training data. *Refuted if* A or B is at most 0.70 × C in at least 6 of 8
  trees: the source gain then survives the renderer change.
- **G2 (the low-sun gain is lighting).** On Matched, B's evening error is at
  most 0.3 × C's in at least 3 of 4 Envy trees, and A's UFO evening error is at
  most 0.5 × J's in at least 3 of 4 UFO trees. *Refuted if* B's evening error is
  at least 0.6 × C's in at least 3 of 4 Envy trees.
- **G3 (where the gap lives).** F's source error on Matched is at most 0.85 ×
  its error on the published matrix in at least 6 of 8 trees. *Refuted if* it
  is at least as large in at least 6 of 8.

Native against Matched (resolution and ground truth together) is reported for
every model as a measurement, not predicted.

How to read the outcome:
- **G1 and G2 both hold.** The source gain was renderer robustness, and the
  low-sun gain is lighting.
- **G1 refuted, G2 holds.** Both gains are real, beyond the render gap.
- **G2 refuted.** The lighting claim does not survive training-like renders.

## What this cannot show

- **Scope.** The same 8 trees, one bark, Blender only. The Isaac camera is not
  touched; Stage A stays as published.
- **One render per frame.** The path tracer has sampling noise; this batch
  measures model error, not render repeatability.
- **CPU-only check of the grid pass.** The grid pass was tested only on CPU
  before submission (6/6 views at IoU 1.0 against the published GPU ground
  truth), so G0 is also a check that it holds on GPU.

## Cost and risk

- **Render.** 8 trees × 24 frames plus 6 grid-pass geometry renders per tree,
  one GPU at a time, 15 minutes reserved per tree. The measured training-render
  rate is a mean of 15.4 s per frame (about 6.5 minutes per tree), but 1 of 74
  tasks in the training render stalled to its limit. A stalled tree keeps the
  frames it finished, is recorded, and is excluded from both arms.
- **Evaluation.** One job, 60 minutes reserved: five checkpoints × two arms.
- **Total.** **180 GPU-minutes reserved**, about 60–80 expected. Storage is
  about 11 GiB on the hpc-share, mostly Native predictions.
- **Approval.** The user approved the check on September 30 at an initial
  estimate of about 60 minutes. This reservation is put to the user before
  submission because it exceeds that estimate.

## Result — October 1, 2026

All 8 trees rendered, with no stall, on RTX 8000 in about 70 of the 180
reserved GPU-minutes. Both arms were scored: five checkpoints × 192 frames,
with checkpoint hashes equal to the published ones.

Evidence (code `2b683ed`):
[verdicts](evidence/render_gap_verdicts_2026-10-01.json), the Matched
[aggregate](evidence/render_gap_matched_matrix_2026-10-01.json) and
[anchoring](evidence/render_gap_matched_anchoring_2026-10-01.json), the Native
[aggregate](evidence/render_gap_native_matrix_2026-10-01.json) and
[anchoring](evidence/render_gap_native_anchoring_2026-10-01.json), and the
[frame measurement](evidence/render_gap_frame_stats_2026-10-01.json).

| | Registered | Measured, per tree | Verdict |
|---|---|---|---|
| **G0** | every view: IoU ≥ 0.995 and ≥ 99.5% within 1 mm | 48 of 48 views: IoU 1.000, 100% within 1 mm | supported |
| **G1** | A/C and B/C ≥ 0.85 on Matched source in ≥ 6 of 8 trees; refuted at ≤ 0.70 in ≥ 6 | A/C 0.80–0.90 (4 of 8 ≥ 0.85); B/C 0.81–0.88 (1 of 8); none ≤ 0.70 | partly supported |
| **G2** | B/C Envy evening ≤ 0.3 in ≥ 3 of 4; A/J UFO evening ≤ 0.5 in ≥ 3 of 4 | B/C 0.08–0.11 (4 of 4); A/J 0.13–0.26 (4 of 4) | supported |
| **G3** | F Matched / F published ≤ 0.85 on source in ≥ 6 of 8; refuted at ≥ 1.0 in ≥ 6 | 1.22–1.96 in 8 of 8 | **refuted** |

**Reading.** G2 holds and G1 is neither held nor refuted, so the outcome lies
between the protocol's first two readings.
- **The low-sun gain is lighting.** It survives the renderer change: B's
  evening error is about a tenth of C's on Matched (Envy) and on Native.
- **The source gain is mostly renderer-dependent.** On the published matrix,
  A/C and B/C were 0.49–0.69, a 35–45% gain. On Matched they are 0.80–0.90, a
  10–20% gain. On Native, A and B have no source gain over C:
  - Envy family means: B 0.035 m, C 0.035 m.
  - UFO family means: B 0.057 m, C 0.047 m, so B is worse.

**G3 failed the other way.** The frozen model's source error does not shrink
on training-renderer frames: it grows by 22–96%. Every model does worse on
the Matched frames than on the published matrix frames of the same views:
- source family means rise 1.36–2.69×, the rendered-lighting models most;
- morning and noon also rise for every model;
- evening falls for F, C and J and rises for A and B.

The frames themselves differ in one direction in every view. This is a
measurement, not pre-registered. Against the same depth and mask:
- the tree is darker in 192 of 192 frames (median grey 68 against 78 at
  source light);
- the image is sharper in 192 of 192 frames (Laplacian variance 1.7–2.2×);
- grey correlation on the tree is 0.91–0.94.

The matrix's 512×288, 16-sample denoised frames are therefore not a neutral
stand-in for the training renderer at matrix resolution, and part of the
published source gain belongs to them.

**Native (measurement).** Against the new render's own 1920×1080 depth and
mask, every model is far more accurate. Source family means are 0.035–0.037 m
for Envy and 0.043–0.057 m for UFO, against 0.20–0.27 m on Matched. Native
changes resolution, input filtering and ground truth together, so it does not
say which of them matters.

**Revised scorer (October 2).** An adversarial review of the scorer
suggested integrity checks and fuller tables. These are now in
`tools/score_render_gap.py` (`9b0e062`):
- seven refusals before any prediction: the published pins, the plan, the
  gate thresholds, every evaluation's provenance, the aggregates' links back
  to them, the tree set of each arm, and the Matched ground truth;
- incomplete cells unscore their tree;
- unrounded ratios with rounding-sensitive trees;
- per-tree tables for all three arms;
- affine ceilings, and Matched against published.

On the real inputs every check passes, no tree is rounding-sensitive, and the
verdicts are unchanged
([revised verdicts](evidence/render_gap_verdicts_2026-10-02.json)). The
protocol names no reading for "G1 partly supported, G2 supported", so the
evidence records none.

**Limits.**
- 8 trees, one bark, Blender only, one render per frame.
- The Isaac camera and Stage A are untouched.
- The frame measurement shows how the frames differ, not which difference
  (brightness, sharpness or denoising) moves the error.
