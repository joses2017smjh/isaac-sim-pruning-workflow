# The renderer control for the Isaac depth failure — protocol, September 23, 2026

Registered **before** submission. The
[controls batch](EVAL_PROTOCOL_GENERALIZATION_CONTROLS_2026-09-23.md#result--september-23-2026)
showed that at 0.16–0.39 m the fine-tuned depth model's *shape* is right in
Cycles (all-GT affine ceiling 0.01–0.04 m) and wrong on the Isaac Stage A
frames (0.37–0.46 m). Range and camera are therefore not the Isaac failure.
What remains different is the renderer, the tree asset and its bark, the tool
jaws in view and the lighting model. This batch removes the renderer and the
bark one at a time while keeping everything else the Stage A wrist recorded.

## Design

`tools/render_tree0_replay.py` opens the companion's `orchard_template.blend`,
the file the Isaac tree0 was exported from, and places the Blender camera at
the recorded Isaac wrist pose of every third frame (0–75, 26 per run) of the
three Stage A runs (`run_00_source_raw`, `run_02_morning_raw`,
`run_04_evening_raw`, pinned by report hash), using the Isaac camera model
(20 mm on 30 mm, 480×320, fx = fy = 320). The Isaac-to-Blender chain (export
translation, 150° yaw, ROS optical basis to the Blender camera) was derived
from the export manifest and the run reports and verified on September 23: the
recorded target projects within 0.2 px at frame 0, and the recorded RTX depth
is reproduced to 1e-6 m by a ray cast and 4e-6 m by a Cycles tile. Depth and
the tree0 mask are rendered once per pose.

Two bark textures per pose, from the template's own material slots:

| Condition | Bark | What it isolates |
|---|---|---|
| `isaac_pose/palm/<light>` | palm bark, the slot every tree0 face carries and what the Isaac export rendered | renderer alone (RTX against Cycles) |
| `isaac_pose/bark_brown_02/<light>` | bark_brown_02, the texture of every training render | renderer plus bark |

Displacement links are muted because the Isaac preview surface had none, so
geometry is identical. Lighting is the companion preset the run recorded,
applied through the matrix code path; the Isaac dome light, the UR5e and the
orange tool jaws are absent, and that is stated as a limit, not hidden.

**Models.** DA2 metric (`5ecc5182…`) with saved predictions; the public
relative head (`a7ea19fa…`) through the all-GT disparity fit. **Metric.**
Tree-mask MAE and the metric head's all-GT affine ceiling per condition,
compared with the Stage A Isaac cells of the same runs and frames; the target
pixel is the Stage A tracked pixel of the same frame when the Cycles depth
agrees with the target's optical Z within 3 cm. N = 26 frames × 3 runs per
condition, 156 frames in all; a missing frame stays in the denominator.

## Pre-registered predictions

- **P1 (renderer).** Under palm bark in Cycles the metric head's affine
  ceiling stays **above 0.25 m** and the tree-mask MAE above 0.35 m, close to
  the Isaac cells (0.37–0.46 m ceiling, ~0.5 m MAE): the shape collapse is
  not RTX. *Refuted if* the palm ceiling drops under 0.10 m, which would put
  the failure in the Isaac renderer or its lighting model.
- **P2 (bark).** Under bark_brown_02 at the same poses the ceiling drops
  **under 0.06 m** and the tree-mask MAE under 0.20 m: the model's shape
  depends on the bark it was trained on, and the palm bark is the Isaac
  failure. *Refuted if* the bark_brown_02 ceiling stays above 0.15 m, which
  would leave the pose distribution (the upward view into sky) and the scene
  content as the remaining causes.
- **P3 (relative head).** The relative head's target-window ceiling is under
  0.04 m under both barks, as on the Isaac frames (0.019–0.040 m): the public
  backbone does not depend on the bark.
- **P4 (lighting).** The three runs' lights order the same way as in Isaac
  (evening worst) under both barks; the Isaac evening cell was not darkness
  (luma 157), so a large evening penalty in Cycles would be a preset effect,
  and a small one confirms the Isaac evening effect was the dome-plus-sun
  combination.

If P1 and P2 hold, the Isaac failure is the tree's bark, and the fix is either
bark variety in training or a bark-matched export; the renderer is cleared. If
P2 fails as well, the next control is the pose distribution: the same poses on
the L-Py trees, which the controls renderer can produce.

## What this cannot show

Nothing about the tool jaws, the robot or the dome light, all absent here; the
Cycles frames show sky where Isaac showed orange jaws. Nothing about hardware.
Palm bark under Cycles is not a training-distribution sample either: the model
never saw palm bark under any renderer.

## Cost

One render job, 20 minutes reserved (156 frames plus 78 geometry passes, about
6 minutes expected), and one evaluation job, 30 minutes reserved (312 model
frames, about 4 minutes expected): **50 GPU-minutes reserved, one GPU at a
time.** Output about 250 MB. Submitted only by
`tools/queue_tree0_replay.py --submit` from a clean committed tree after this
protocol is committed.

---

## Result — September 23, 2026

Render `21404508` (12 min) and evaluation `21404509` (2 min) COMPLETED (0:0);
156 of 156 registered frames scored, code revision `1c8ef03`, checkpoint
hashes matched. [Evidence](evidence/tree0_replay_2026-09-23.json) ·
[anchoring ceilings](evidence/tree0_replay_anchoring_2026-09-23.json).

### DA2 metric head, 26 frames per cell (frames 0–75 of each Stage A run)

| Cell | Tree-mask MAE | Signed median | Target MAE | Affine ceiling | Corr. | Zone-fit target |
|---|---|---|---|---|---|---|
| Cycles, palm, source | 0.416 | +0.41 | 0.457 | 0.182 | 0.83 | 0.069 |
| Cycles, bark_brown_02, source | 0.411 | +0.41 | 0.447 | 0.182 | 0.83 | 0.054 |
| Cycles, palm, morning | 0.392 | +0.39 | 0.423 | 0.170 | 0.85 | 0.031 |
| Cycles, bark_brown_02, morning | 0.391 | +0.39 | 0.424 | 0.166 | 0.85 | 0.037 |
| Cycles, palm, evening | 0.601 | +0.60 | 0.689 | 0.289 | −0.08 | 0.362 |
| Cycles, bark_brown_02, evening | 0.630 | +0.65 | 0.730 | 0.295 | −0.03 | 0.373 |
| Isaac Stage A, source (all valid pixels, no mask) | 0.614 | +0.41 | 0.572 | 0.464 | 0.77 | 0.161 |
| Isaac Stage A, evening | 0.665 | +0.62 | 0.850 | 0.456 | 0.78 | 0.405 |

The Isaac rows are full-frame numbers (no tree mask exists there), so they
are not the same metric as the Cycles tree-mask rows; the signed medians and
the target errors are the like-for-like columns.

### Verdicts

- **P1 not supported.** The renderer is not cleared, but it is not the main
  cause either. Cycles at the recorded poses reproduces the Isaac
  over-estimate almost exactly: signed +0.41 m against Isaac's +0.41 m at
  source, target error 0.45 m against 0.57 m, tree-mask MAE 0.42 m (above the
  registered 0.35 m). The metric head's affine ceiling, however, is 0.18 m in
  Cycles against 0.46 m on Isaac: below the registered 0.25 m, above the
  0.10 m that would have placed the failure in the renderer. The renderer
  and its lighting account for the difference in *shape* (0.18 → 0.46 m
  ceiling) and for none of the *magnitude*.
- **P2 refuted.** Bark has no effect: palm and bark_brown_02 agree within
  0.01 m on every column in every light. The Isaac failure is not the palm bark.
- **P3 supported.** The relative head's target-window ceiling is 0.015–0.042 m
  under both barks and all three lights.
- **P4 supported.** Evening is worst under both barks (0.60–0.63 m against
  0.39–0.42 m) with the correlation collapsing to about zero, as on the L-Py
  trees; the Cycles evening preset is the dark one, so this is the matrix's
  evening effect, not the Isaac dome-plus-sun effect.

### What this settles

The Isaac magnitude (a flat 0.6–0.9 m prediction for targets at 0.16–0.39 m)
comes from the pose and tree distribution: the same tree at the same poses
fails the same way in the training renderer, and the controls batch showed
the L-Py trees at the same distances fail the same way too. The Isaac *shape*
collapse has a renderer or lighting component of about 0.28 m of ceiling that
Cycles does not show. Bark is irrelevant. The many-zone fit brings the
Cycles target error to 0.03–0.07 m in daylight and the Isaac target error to
0.16 m; the remaining Isaac gap is the RTX shape, which no anchor fixes.
Ordered by what they buy: close-range and upward-view poses in training,
then the RTX-versus-Cycles appearance gap (dome light, tone mapping, tool
jaws), then nothing about bark.

