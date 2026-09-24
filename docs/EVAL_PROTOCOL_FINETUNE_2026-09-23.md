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
