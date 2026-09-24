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
