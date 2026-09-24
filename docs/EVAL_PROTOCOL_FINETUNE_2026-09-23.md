# Re-fine-tune with photometric jitter — protocol, September 23, 2026

Registered **before** submission. The
[controls batch](EVAL_PROTOCOL_GENERALIZATION_CONTROLS_2026-09-23.md#result--september-23-2026)
placed the evening failure in the training distribution: one fixed light, no
photometric augmentation, darkness costing 2–3× and low-sun shading the rest,
and nothing in front of the model fixing it. The cheapest training-side lever
is photometric jitter on the frames that still exist. This batch measures what
that lever buys and what it costs, against a control that trains identically
without it.

## Design

`tools/finetune_da2.py` is a repository-local wrapper. The companion trainer
cannot be used as-is: it has no version control, its `--pretrained-from`
discards the trained head, its `--resume` continues a finished schedule, and
its wandb import no longer loads. The wrapper imports the companion's dataset
and model classes by path (their hashes are pinned in the plan), copies the
loss, the metric and the optimizer groups verbatim, and

- **warm-starts** from the full fine-tuned checkpoint (`5ecc5182…`, model
  weights only, strict) with a fresh AdamW and polynomial schedule;
- trains **6 epochs** (the original plateaued by epoch 5 of 14) at batch 2,
  learning rate 5e-6 (head ×10), FP32, 518×518 random crops, 50% flips, on the
  **6,270 surviving frames** (100 Envy trees, bark_brown_02, box and
  box_cam1–4, left and right views; the companion's own seed-1 80/20 tree
  split, filtered by file existence: about 4,980 train and 1,290 val rows,
  exact counts and hashes in the batch plan);
- in the **jitter arm** applies, with probability 0.8 per sample after the
  resize and before ImageNet normalization: a brightness gain in [0.35, 1.3]
  (the evening preset is 2.6× darker), a gamma in [0.6, 1.6], and per-channel
  gains in [0.8, 1.2] for the warm cast; the **control arm** applies nothing.

Selection is by the companion validation split's RMSE, as in the original.
The 192 family-matrix frames are logged as an extra validation set every epoch
and never selected on. Each arm's `best.pth` is then scored by the frozen
`depth_generalization.py` on the three plans the frozen model was scored on:
the family matrix (192), the controls (688) and Isaac Stage A (600), so every
comparison is on identical bytes. The fine-tuned checkpoint's hash is unknown
until it exists; the evaluation job records it.

## Pre-registered predictions

Tree-mask MAE, DA2, per-tree means against the frozen model's cells.

- **P1 (evening).** The jitter arm cuts the matrix evening MAE by **at least
  40%** in 6 of 8 trees (Envy 0.906 → ≤ 0.54, UFO 1.451 → ≤ 0.87) and the
  control arm by less than 20%: the gain comes from the jitter, not from more
  epochs on box data. *Refuted if* the jitter arm improves evening by less
  than 20% in most trees, which would mean the shading structure needs rendered
  lighting variation, not a curve family.
- **P2 (daylight is not paid for).** Neither arm worsens the matrix `source`
  MAE by more than 0.03 m (Envy 0.134, UFO 0.188) in more than 2 trees.
  *Refuted if* the jitter arm's source MAE rises by 0.05 m or more in most trees.
- **P3 (evening after jitter stays above the control-batch ceiling).** Even
  the jitter arm's evening MAE stays above 0.20 m in both families, because the
  cast shadows of a 12° sun are not a photometric transform of the training
  frames. *Refuted if* evening comes under 0.20 m in a family.
- **P4 (range and Isaac are untouched).** On the controls' close-range cells
  and on Isaac Stage A, neither arm changes the target error by more than 20%:
  those failures are not lighting.
- **P5 (the wrapper reproduces the recipe).** The control arm's validation
  RMSE on the companion split stays within 0.01 of the frozen model's 0.054
  after epoch 1: nothing in the wrapper broke the recipe.

## What this cannot show

The surviving frames are the box and box_cam1–4 views of one bark, a narrower
distribution than the original 19 set-ids and two barks; a gain here may not
hold on the original distribution, which is gone. Every Envy tree is in train
or val, so only the UFO and Isaac cells are transfer checks. No rendered
lighting variation is trained on; that arm needs a training-render launcher
and is the next batch if P1 fails.

## Cost

Per arm: one training job on an A40, **8 hours reserved** (2,490 iterations
per epoch at about 0.7 s, six epochs and six validations, about 3.5–4 hours
expected, a 7-hour runtime guard inside), and one evaluation job, 60 minutes
reserved. Two arms: **1,080 GPU-minutes reserved**, one GPU at a time when
chained. Output about 4.5 GB per arm (two model-only checkpoints of 1.34 GB
and 1,480 saved predictions). Submitted only by
`tools/queue_finetune.py --arm <jitter|control> --submit` from a clean
committed tree after this protocol is committed.

---

## Result — September 24, 2026

Jitter arm: training `21404511` COMPLETED in 2 h 57 min (best epoch 5, companion
val RMSE 0.0561, checkpoint `658c3c77…`); control arm: training `21404513`
COMPLETED in 2 h 55 min (best epoch 5, val RMSE 0.0555, `d5691712…`). Each
arm's evaluation job scored the matrix (192) and the controls (688) and then
failed on its Stage A step, because the launcher pointed the scorer at an
evaluation document whose plan sits under a `plan` key (fixed in `0e7750c`);
the Stage A scores of both checkpoints come from the frozen rescoring job
`21405526` on identical bytes. Data as trained: 4,980 train and 1,290 val rows
survived the file filter. The jitter-applied counter in the training log read
zero because it lived in the main process while the DataLoader workers ran the
transform; the worker path was verified on CPU to change 6 of 8 samples with a
maximum normalized change of 1.46 at an identical crop, and later runs record
a startup self-check.
[Matrix](evidence/finetune_family_matrix_2026-09-24.json) ·
[controls](evidence/finetune_controls_2026-09-24.json) ·
[Stage A](evidence/finetune_stage_a_2026-09-24.json) ·
[anchoring ceilings](evidence/finetune_anchoring_2026-09-24.json).

### Tree-mask MAE, mean of per-tree values (n = 4 trees per family)

| Cell | Frozen | Jitter arm | Control arm |
|---|---|---|---|
| Envy source | 0.134 | 0.153 (1.14×) | 0.137 (1.02×) |
| Envy evening | 0.906 | **0.322 (0.36×)** | 0.883 (0.98×) |
| UFO source | 0.188 | 0.196 (1.04×) | 0.185 (0.99×) |
| UFO evening | 1.451 | 1.247 (0.86×) | 1.442 (0.99×) |
| Envy `overcast_div2.6` (dark, diffuse) | 0.333 | **0.163** | 0.347 |
| UFO `overcast_div2.6` | 0.577 | **0.183** | 0.607 |
| Envy `evening_x2.6` (brighter, low sun) | 0.821 | 0.361 | 0.748 |
| UFO `evening_x2.6` | 1.443 | 1.276 | 1.421 |
| Envy close range, target MAE | 0.550 | 0.508 | 0.555 |
| UFO close range, target MAE | 0.384 | 0.322 | 0.383 |
| Isaac Stage A source, target MAE (unmasked) | 0.572 | 0.482 | 0.658 |

Per tree, the jitter arm's evening ratio to frozen is 0.27–0.43 on the four
Envy trees and 0.82–0.93 on the four UFO trees; the control arm's is
0.94–1.00 on all eight. All pre-registered gates still fail for every model.

### Verdicts

- **P1 not met as registered, and family-split.** The jitter arm cuts evening
  by at least 40% in 4 of 8 trees, not 6: every Envy tree (57–73%) and no UFO
  tree (7–18%). The control clause holds (under 2% on every tree), so the
  gain is the jitter's. Every Envy tree was in the fine-tune's own train or
  validation split; the four UFO trees are the transfer check, and there the
  gain is small.
- **P2 supported.** Source MAE rises by 0.005–0.024 m under jitter on every
  tree, under the 0.03 m line. Morning and noon on Envy rise more (+0.05 to
  +0.06 m), which P2 did not cover and is recorded here.
- **P3 supported.** Evening stays above 0.20 m in both families (0.32, 1.25).
  The evening affine ceiling is unchanged (Envy 0.128 → 0.116, UFO 0.101 →
  0.108): the jitter moved the offset, not the shape.
- **P4 supported, narrowly.** Close-range target error changes by −8% (Envy)
  and −16% (UFO); Isaac Stage A target error by −16% (jitter) and +15%
  (control). Range and Isaac are not lighting problems.
- **P5 supported.** The control arm's validation RMSE is 0.059 after epoch 1
  and 0.0555 at best, within 0.01 of the frozen model's 0.054.

### Unpredicted

The jitter arm fixes the **darkness** cell outright in both families
(`overcast_div2.6`: Envy 0.333 → 0.163, UFO 0.577 → 0.183, both near their
source cells) while leaving the low-sun cells far off (`evening_x2.6` UFO
1.276). That is the controls' decomposition confirmed from the training side:
brightness is a curve the jitter teaches; cast shadows and the warm sky are
not. The remaining evening failure needs rendered lighting variation in
training, which is now the justified next batch and needs a training-render
launcher that does not yet exist.

