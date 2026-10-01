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
