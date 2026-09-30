# Rendered lighting variation in training — protocol, September 27, 2026

Registered **before** submission. The
[fine-tune result](EVAL_PROTOCOL_FINETUNE_2026-09-23.md#result--september-24-2026)
showed that photometric jitter teaches the depth model brightness (the dark,
diffuse cell recovered in both families) but not low-sun shading: UFO evening
improved only 14–18%, and the evening affine ceiling did not move. This batch
tests the lever that result points at: training frames rendered under varied
sun direction, colour and sky, at the exact poses of the surviving training
frames, compared with the published jitter and control arms on identical
evaluation bytes.

## What is already known

- The companion's seed-1 split has 80 Envy train trees and 20 Envy val trees,
  and no UFO tree. All four matrix Envy trees are in val, so rendering the
  train split only keeps every matrix, val and UFO tree out of the new data.
- 4,980 train rows survive on disk: bark_brown_02, sets `box` and
  `box_cam1–4`, 60 frames per tree (90 for six trees). Every row has its
  annotation JSON with the exact camera pose.
- The August render used the template's sampler (1920×1080, 4,096 samples,
  adaptive, CPU OpenImageDenoise, Filmic) at a measured 21.55 s per frame on a
  V100, and its light is exactly the `source` preset.
- A CPU check reproduced three surviving frames' depth at their annotated poses
  to a 2.5 µm median (0.57–2.35 mm maximum) with 100% tree-mask agreement, and
  a CPU smoke run of the repository renderer
  (`tools/render_training_lighting.py`) completed the full code path on three
  frames without writing anything into the companion.

## Design

**Sampler (`tools/training_lighting.py`, `rendered-lighting-v1`).** One draw
per frame, seeded from the frame's identity:

| Parameter | low sun (p = 0.6) | high sun (p = 0.4) |
|---|---|---|
| Sun elevation | U[4, 25]° | U[25, 75]° |
| Sun colour temperature | U[2,500, 4,500] K | U[4,500, 6,500] K |
| World mix toward the dusk world | U[0.5, 1] | U[0, 0.5] |
| World strength | log-U[0.07, 0.35] | log-U[0.25, 0.80] |

Both regimes: sun azimuth U[0, 360)° in the world frame, sun energy
log-U[1, 6], sun angle 8°. World colour mixes (0.70, 0.75, 0.82) toward
(0.10, 0.13, 0.20); sun colour follows a declared Planckian formula (Kang et al.
2002 to linear Rec.709). **Hold-out:** a draw whose sun lies within 15° of any
registered preset sun (source at the zenith, morning 60°/18°, noon 160°/65°,
evening and its ×2.6 variant 315°/12°), in the world frame or rotated with the
pose's yaw, is rejected and redrawn. The overcast presets have no sun and are
outside the support. This tests interpolation from at least 15° away, not
extrapolation; the Isaac presets belong to another renderer and are not held
out.

**Render.** Each of the 80 train trees is re-rendered at all of its surviving
poses with the template's own sampler, asserted and never changed; the camera
box of the `box_cam` sets is placed at the centre pose, as the generator did.
Only the RGB is kept. Depth and the tree mask are rendered to node-local
scratch and compared with the surviving files the training row will reuse; a
frame enters training only if at least 99.5% of tree-mask pixels agree within
1 mm and the mask IoU is at least 0.995. A one-tree **pilot** (lpy_envy_00001,
90 frames, plus two extra frames under `source` compared with the surviving RGB)
runs first; the full render reuses its tree.

**Fine-tune arms.** Warm start from the published checkpoint (`5ecc5182…`),
the same wrapper and recipe as the published arms, on 9,960 rows (the 4,980
surviving source-lit frames plus up to 4,980 verified renders), **3 epochs**,
the same 14,940 optimizer steps as the published arms' 6 epochs on 4,980 rows.
Arm A (`rendered_jitter`) keeps the photometric jitter; arm B
(`rendered_control`) has none. Each arm's training manifest is built inside its
job, after the renders, by `tools/build_training_lighting_manifest.py`; zero
verified rendered rows is a failure, never a silent fall-back. Selection on the
companion's source-lit validation split; the matrix is logged, never selected
on. Each arm's best checkpoint is scored by the frozen
`depth_generalization.py` on the matrix (192), controls (688) and Isaac Stage A
(600) plans, identical bytes to the frozen and published arms.

## Pre-registered predictions

Pilot and render:

- **L1.** All 90 pilot frames pass the geometry check. *Refuted if* any fails:
  depth reuse is then unsafe on the GPU renderer and the full render's rows are
  judged by their own checks.
- **L2.** Pilot render time is at most 40 s per frame. *Refuted if* higher (the
  full array's 50 s-per-frame reservation would then be at risk).
- **L3.** The two `source` frames match the surviving RGB within 2/255 mean on
  the tree mask. *Refuted if* not; the material or texture setup then differs
  from August.
- **L4.** At least 99% of the full render passes the geometry check, and at
  least 30% of kept frames have mean luma at or below 90.

Fine-tune, per-family means of per-tree tree-mask MAE (F frozen, J published
jitter arm, C published control arm):

- **R1 (rendered light alone fixes low-sun shading).** B's `evening_x2.6`
  error is at most 0.6 × C per tree in at least 3 of 4 Envy trees (mean
  ≤ 0.45 m against C's 0.748). *Refuted if* B improves on C by less than 20% in
  at least 3 of 4 trees.
- **R2 (shape, not offset).** A's evening affine ceiling is at most 0.8 × J in
  both families (Envy ≤ 0.092, UFO ≤ 0.086). *Refuted if* within ±10% of J in
  both families.
- **R3.** A's Envy evening error is at most 0.20 m in at least 3 of 4 trees.
  *Refuted if* at least 0.29 m in 3 of 4.
- **R4 (transfer to the unseen family).** A's UFO evening error is at most
  0.75 m in at least 3 of 4 trees (J: 1.247). *Refuted if* above 1.0 m in 3 of 4.
- **R5 (daylight not paid for).** A's and B's source error is at most F + 0.03 m
  in at least 6 of 8 trees, and A's Envy morning and noon are no worse than J.
  *Refuted if* source rises by 0.05 m or more in most trees.
- **R6.** Close-range and Stage A target errors stay within ±20% of J (arm A)
  and C (arm B): range and Isaac are not lighting problems.
- **R7.** Each arm's best validation RMSE is within 0.01 of 0.054.

R1 and R2 together would show that rendered lighting fixes the shading failure;
R4 decides whether it transfers to UFO.

## What this cannot show

Only one bark and only Envy trees are rendered; UFO and Isaac are the only
transfer checks. The evaluation frames are 512×288 at 16 samples, the training
frames 1920×1080 at the adaptive sampler; this batch does not change that
mismatch. Surviving frames are seen 3 times instead of 6. The hold-out tests
interpolation. Renders derive from the companion's licensed meshes and textures
and stay under `artifacts/`; only aggregates are published.

## Cost

Pilot: one task, 90 minutes reserved. Full render: the other 79 trees, 4,890
frames, in two arrays (74 trees of 60 frames at 60 minutes and 5 of 90 frames at
80 minutes reserved per task, from 50 s per frame, 2.3× the V100 measurement),
4,840 GPU-minutes reserved, about 30–40 hours of wall time one task at a time. Fine-tune: 2 arms × (8 h training + 1 h
evaluation) = 1,080 GPU-minutes reserved, about 3 h each expected. About 15 GB of
renders and 9 GB of checkpoints and predictions on the hpc-share.

## Result — September 30, 2026

All 11 predictions are supported, most by wide margins
([verdicts](evidence/lighting_training_verdicts_2026-09-30.json), code
`52d90c1`). The four aggregate files hold all five models side by side
([matrix](evidence/finetune_rendered_family_matrix_2026-09-30.json),
[anchoring](evidence/finetune_rendered_anchoring_2026-09-30.json),
[controls](evidence/finetune_rendered_controls_2026-09-30.json),
[Stage A](evidence/finetune_rendered_stage_a_2026-09-30.json)). The frozen
model and the published jitter and control arms reproduce the September 24
evidence exactly on every row.

**Render.**
- **L1.** 90 of 90 pilot frames pass the geometry check.
- **L2.** The pilot renders at 14.7 s per frame on average.
- **L3.** The two source frames differ from the surviving RGB by 0.14/255 on
  the tree mask.
- **L4.** The full render passes 4,890 of 4,890 frames, and 50.2% have mean
  luma at or below 90.
- **Hold-out.** Recomputed from all 4,980 recorded draws, the closest training
  sun to any preset sun is 15.01–15.06°. The evaluation frames were rendered
  with exactly those preset suns, so the test is interpolation, as registered.
- **Resume.** One tree timed out and was resumed inside its own batch (see
  `SLURM_JOBS.md`); every frame entered training.

**Training.** Both arms ran 3 epochs, 14,939 optimizer steps (14,940
registered; the September 24 arms show the same off-by-one), on 9,960 rows.
The best validation RMSE is 0.0568 in both arms, at epoch 2 (R7).

Per-family mean tree-mask error in metres. F is the frozen model, J and C the
published jitter and control arms, A (rendered + jitter) and B (rendered, no
jitter) the new arms.

| | F | J | C | A | B |
|---|---|---|---|---|---|
| Envy source | 0.134 | 0.153 | 0.137 | 0.086 | 0.074 |
| Envy evening | 0.906 | 0.322 | 0.884 | **0.048** | **0.048** |
| Envy evening ×2.6 | 0.821 | 0.361 | 0.748 | 0.050 | **0.047** |
| UFO source | 0.188 | 0.196 | 0.186 | 0.117 | 0.100 |
| UFO evening | 1.451 | 1.247 | 1.442 | **0.095** | 0.086 |
| Evening affine ceiling, Envy / UFO | 0.128 / 0.101 | 0.116 / 0.108 | 0.127 / 0.103 | **0.043 / 0.047** | 0.043 / 0.049 |
| Stage A target, source / evening | 0.572 / 0.850 | 0.482 / 0.720 | 0.658 / 0.911 | 0.511 / 0.563 | 0.548 / 0.600 |

- **R1.** B's evening ×2.6 error is 0.035–0.053 m per Envy tree, 0.05–0.07 of
  C.
- **R2.** A's evening affine ceiling is 0.38× (Envy) and 0.43× (UFO) that of J.
  The shape itself improved, not only the offset.
- **R3.** A's Envy evening error is 0.039–0.052 m in all four trees.
- **R4.** A's UFO evening error is 0.068–0.139 m in all four trees, where J is
  1.19–1.34 m. The unseen family transfers.
- **R5.** Source error falls in every tree for both arms, and A's Envy morning
  and noon (0.098 and 0.093) beat J (0.223 and 0.211).
- **R6.** Close-range and Stage A source target errors are within 20% of J and
  C.

**Two findings nobody predicted, and what they limit.**
- **Source error fell 35–45% as well** (Envy 0.134 → 0.086 and 0.074), and so
  did the overcast cells, which are outside the sampler's support. R5 only
  bounded harm. So the gain is not specific to low sun: rendered lighting
  variation (sun, colour, sky and world) improves every Blender cell. The
  low-sun gain (about 0.8 m at evening) is still about fifteen times the
  general one (about 0.05 m), so R1–R4 stand as written.
- **The companion's own validation RMSE did not move.** It is 0.0568, against
  0.0561 (J) and 0.0555 (C), on 1,290 source-lit 1920×1080 frames of the
  validation trees. The gain appears on this repository's 512×288, 16-sample
  evaluation renders of the same trees. Part of it may be robustness to that
  render gap, which this design cannot separate from lighting.

**Limits.**
- **Isaac is barely helped.** Stage A target error at source is unchanged
  within 20%. At evening it falls from 0.72/0.91 (J/C) to 0.56/0.60 m, but
  every Isaac error is still over half a metre, so the controller's own camera
  is not fixed. Every large gain is measured on Blender renders of one bark.
- **Close-range UFO means cover three trees.** The target of `lpy_ufo_00003`
  is never visible.
- **Stale plan description.** The new plans' `scope` string still describes
  the published arms' 6,270 frames; `data_description` is correct.
