# Can one metric range fix the learned depth? — protocol, September 23, 2026

Registered before the analysis is run. It uses no GPU: every input already
exists on disk as saved predictions, ground-truth depth and masks from the
[matrix](EVAL_PROTOCOL_FAMILY_LIGHTING_2026-09-23.md) and
[Stage A](evidence/stage_a_depth_2026-09-23.json).

## Why this is the first question to ask

The matrix measured the learned model's error as **almost entirely a constant
per-frame under-estimate** in daylight (signed median ≈ −MAE; 0.03–0.06 m left
once each frame's own offset is removed), a much larger constant at evening
(with prediction–truth correlation collapsing), and on Isaac a **flat
over-estimate** that ignores the target's true range (predicted ~0.85 m for a
target at 0.16–0.39 m, correlation −0.5).

A pruning rig is not a monocular camera. It has two 8×8 time-of-flight sensors
and, in this simulator, RTX depth at the tracked pixel — the number the
controller already uses. So before asking how to train the model to know
metric scale from appearance alone, ask whether it needs to: **if one metric
range at the target removes the error, the right architecture is relative
depth plus a metric anchor, and "generalizing better" means anchoring, not
retraining.** If it does not, the model's structure is wrong and no anchor will
save it.

## Method

For every frame with a saved prediction, a valid target pixel and ground truth:

| Variant | Formula | What it uses |
|---|---|---|
| raw | `pred` | nothing |
| **shift** | `pred − pred[t] + r[t]` | one metric range `r[t]` at the target |
| **scale** | `pred · r[t] / pred[t]` | the same one range |
| affine ceiling | least-squares `s·pred + b` against the whole ground-truth frame | **all of ground truth — a diagnostic ceiling, never a result** |

`pred[t]` and `r[t]` are medians over the mask-gated 3×3 window at the target.
On Isaac frames `r[t]` is the RTX optical-Z at the tracked pixel, exactly the
controller's own metric depth. On Blender frames there is no ToF, so `r[t]` is
the ground-truth depth at the target pixel, **standing in for one ToF zone**;
that is labelled as a stand-in in the evidence and is not a runtime result.

The metric is tree-mask MAE (Blender) or all-valid-pixel MAE (Isaac) after each
variant, per model × family × lighting, with `n_trees` and the fraction of
frames that could be anchored. Frames without a visible target are kept in the
table, unanchored. Aggregation runs through `sql/anchoring/*.sql`.

## Pre-registered predictions

- **P1 (daylight, Blender).** Shift anchoring brings DA2's tree-mask MAE under
  source, morning and noon to **≤ 0.08 m** in both families, because the error
  there is a constant offset. *Refuted if* the shift-anchored MAE stays above
  0.10 m in either family under any daylight preset.
- **P2 (evening, Blender).** Anchoring does **not** rescue evening: the
  shift-anchored MAE stays **≥ 0.15 m**, because the prediction's structure has
  collapsed, not just its offset. *Refuted if* shift anchoring brings evening
  under 0.10 m.
- **P3 (Isaac).** Scale anchoring beats shift anchoring, because the Isaac
  failure is range compression, but **neither** brings the all-valid MAE under
  **0.20 m**, because the prediction is nearly flat across the frame. *Refuted
  if* either variant brings Isaac under 0.15 m.
- **P4 (structure).** The affine ceiling — the best any per-frame scale-and-shift
  could do — is under 0.06 m in daylight Blender frames and over 0.10 m at
  evening and on Isaac. This separates "wrong scale" from "wrong shape".

If P1 holds and P2–P3 hold, the conclusion is: **anchor at runtime for daylight;
evening and the Isaac working distance need a different model input, not a
different scale.** If P1 fails, the daylight error is not a single offset and
the anchoring architecture is not enough on its own.

## What this cannot show

- Nothing about a real VL53L8CX: the anchor is simulated (RTX) or a stand-in
  (Blender ground truth). Range noise, multipath and zone-to-pixel registration
  are not modelled here; the repository's range-fusion estimator and the
  recorded 8×8 grids are the next step, not this one.
- Nothing about closed-loop control.
- The affine ceiling uses ground truth and is never an achievable number.

No GPU is used. The analysis is one command:

```bash
PYTHONPATH=/nfs/hpc/share/sanchej7/sil-pydeps python tools/anchor_depth_analysis.py \
  --evaluation da2=artifacts/generalization/family-matrix-20260923/eval/da2/evaluation.json \
  --evaluation dino=artifacts/generalization/family-matrix-20260923/eval/dino/evaluation.json \
  --evaluation da2_isaac=artifacts/generalization/phase-20260920/stage_a_job_21370005/evaluation.json \
  --output docs/evidence/depth_anchoring_2026-09-23.json
```

---

## Result — September 23, 2026

Run on 192 + 192 Blender frames (DA2, DINO) and 600 Isaac frames, no GPU.
Every number is from
[`docs/evidence/depth_anchoring_2026-09-23.json`](evidence/depth_anchoring_2026-09-23.json)
through `sql/anchoring/*.sql`.

### Tree-mask MAE after anchoring, DA2 (n = 4 trees per family)

| Family | Light | raw | **shift** | scale | affine ceiling | anchored frames |
|---|---|---|---|---|---|---|
| UFO | source | 0.188 | **0.057** | 0.059 | 0.039 | 11 / 24 |
| UFO | morning | 0.269 | **0.060** | 0.064 | 0.048 | 11 / 24 |
| UFO | noon | 0.235 | **0.057** | 0.068 | 0.035 | 11 / 24 |
| UFO | evening | 1.451 | 0.193 | 0.387 | 0.101 | 11 / 24 |
| Envy | source | 0.134 | **0.176** | 0.191 | 0.053 | 18 / 24 |
| Envy | morning | 0.177 | 0.136 | 0.142 | 0.060 | 18 / 24 |
| Envy | noon | 0.152 | 0.132 | 0.137 | 0.052 | 18 / 24 |
| Envy | evening | 0.906 | 0.284 | 0.382 | 0.128 | 18 / 24 |
| Isaac tree0 | source | 0.614 | 0.473 | 0.485 | 0.464 | 78 / 200 |
| Isaac tree0 | morning | 0.567 | 0.426 | 0.474 | 0.371 | 78 / 200 |
| Isaac tree0 | evening | 0.665 | 0.504 | 0.491 | 0.456 | 78 / 200 |

DINO behaves the same as DA2 in every cell.

### Verdicts on the registered predictions

- **P1 — refuted.** Shift anchoring takes UFO daylight to 0.057–0.060 m, near
  its ceiling, but leaves Envy at 0.13–0.18 m and makes Envy *source* worse
  (0.134 → 0.176). The prediction assumed the error was one constant per frame.
  It is, for UFO; it is not, for Envy.
- **P2 — supported.** Evening stays at 0.19–0.28 m after the best single anchor,
  and even the affine ceiling is 0.10–0.13 m. The prediction's structure is
  wrong at evening (correlation ≈ 0 on Envy), not just its offset.
- **P3 — half refuted.** On Isaac, scale did not beat shift, and neither helped
  much: 0.43–0.50 m against a raw 0.57–0.67 m. The affine ceiling itself is
  **0.37–0.46 m**. The Isaac failure is not a scale problem at all; the model
  does not see the structure at that working distance and field of view.
- **P4 — supported.** Ceiling < 0.06 m in daylight, > 0.10 m at evening and on
  Isaac.

### Why one anchor works on UFO and not on Envy

The mechanism is visible in the per-tree numbers. At the 3×3 mask-gated target
window, the prediction's local error minus the frame's median error is
0.01–0.07 m on UFO trees, and 0.06–0.32 m on Envy trees, with inconsistent
sign across trees (`lpy_envy_00008` source: −0.42 m at the spur against −0.10 m
over the tree; `lpy_envy_00012`: −0.05 against −0.12). On Envy the error is
**spatially non-uniform**: the thin spur is predicted with a different offset
than the tree as a whole, so a one-pixel anchor transfers the wrong correction.
The affine ceiling of 0.04–0.06 m on the same frames shows that a fit over many
points would work on both families.

### What this means for generalizing better

1. **In daylight the model's shape is right and its scale is wrong, and the rig
   can fix that at runtime — but not from one pixel.** Anchor with many ranges
   per frame: the two 8×8 time-of-flight grids give up to 128 zones, of which
   roughly a third are valid in the recorded runs. A per-frame scale-and-shift
   fit to those zones is the architecture; the ceiling says it reaches
   ~4–6 cm on both families without touching the model.
2. **Evening is not a scale problem.** No anchor fixes a collapsed prediction.
   That needs a model-input change: lighting-robust training or test-time
   photometric normalization, tested next.
3. **Isaac is not a scale problem either.** The ceiling of 0.4 m means the
   prediction is nearly flat. The working distance (0.16–0.39 m) and field of
   view (~74°) are both outside anything the model was trained on; those are
   the next controls to render.

The next experiments follow from this and are registered separately; this
analysis used no GPU and changed no model.

---

## Follow-up — many-zone fit, September 23, 2026 (exploratory)

Run after the controls batch and **not registered here before running**; the
only prior statement is the September 23 anchoring audit's expectation that a
zone fit would come within 0.02 m of the affine ceiling in Blender daylight
and stay above 0.35 m on Isaac. The variant, in `tools/anchor_depth_analysis.py`:
an 8×8 grid with a 65° diagonal (the simulator's VL53L8CX model), co-located
with the camera and aligned to its axis, one range per zone (median ground
truth over the zone's tree pixels, 4 or more, within 3.4 m), a per-frame
least-squares scale and shift over the zones with one robust trim, scored on
the tree mask and at the 3×3 target window. The sensor's baseline, cone
footprint, noise and multipath are not modelled; the ranges are ground truth.
[Evidence](evidence/generalization_controls_anchoring_2026-09-23.json).

| Cell (DA2) | raw MAE | one anchor (shift) | zone fit | affine ceiling | zones | target err raw → zone |
|---|---|---|---|---|---|---|
| Envy source | 0.134 | 0.176 | 0.053 | 0.053 | 36 | 0.25 → 0.18 |
| UFO source | 0.188 | 0.057 | 0.041 | 0.039 | 19 | 0.20 → 0.05 |
| Envy evening | 0.906 | 0.285 | 0.132 | 0.128 | 36 | 0.65 → 0.13 |
| UFO evening | 1.451 | 0.193 | 0.118 | 0.101 | 19 | 1.35 → 0.46 |
| Envy close, training camera | 0.527 | 0.069 | 0.045 | 0.028 | 16 | 0.55 → 0.020 |
| UFO close, training camera | 0.385 | 0.045 | 0.011 | 0.011 | 17 | 0.38 → 0.006 |
| Envy close, Isaac camera | 0.529 | 0.082 | 0.039 | 0.026 | 17 | 0.53 → 0.016 |
| UFO close, Isaac camera | 0.410 | 0.050 | 0.015 | 0.014 | 18 | 0.40 → 0.006 |
| Isaac Stage A, source | 0.614 | 0.473 | 0.520 | 0.464 | 55 | 0.57 → 0.16 |
| Isaac Stage A, evening | 0.665 | 0.504 | 0.469 | 0.456 | 55 | 0.85 → 0.40 |

The zone fit reaches the affine ceiling in every Blender cell, which the single
anchor did not on Envy (0.176 against 0.053 at source). At the close-range
cells it turns a 0.4–0.55 m target error into 0.006–0.02 m with 16–18 zones:
where the model's shape is right, many ranges fix its scale. On Isaac it does
not (target 0.57 → 0.16 m, tree mask unchanged at ~0.5 m), because the ceiling
there is the shape. At evening the target error after the fit stays 0.13 m
(Envy) and 0.46 m (UFO), the structure problem again. The audit's daylight
expectation held; its Isaac expectation held.

