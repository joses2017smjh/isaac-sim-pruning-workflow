# Tracker drift, camera side and the tree1 population — protocol, September 26, 2026

Registered **before** submission. The
[strategy sweep](EVAL_PROTOCOL_STRATEGIES_2026-09-23.md#result--september-24-2026)
showed that changing the mouth's path changed no tree0 outcome, and that two of
seven listed tree1 spurs pass. This protocol goes after the perception failures
and measures the tree1 population properly. No gate, threshold or grader check
changes; `tools/validate_vision_sequence.py` is the only judge.

## What the recorded frames show

A read-only diagnosis replayed the live tracker offline on the recorded wrist
frames. The replay reproduces all 46 recorded runs exactly (state, feature
count and pixel, 0.0000 px maximum difference), so a tracker variant can be
checked on the recorded camera motion up to each recorded stop. The three
tree0 `vision_invalid` targets fail in three different ways:

- **590, drift into an edge.** The tracked pixel starts on the spur's centre
  line and slides 1.8–2.8 px sideways until two of the nine depth-window pixels
  land on a surface 7.3 cm behind the spur; the spread gate then fails exactly
  as designed (one outlier pixel passes, two fail). The drift comes from the
  motion model: the pixel moves by the median flow of features that sit
  5.3–8.3 px to one side while the image scales by 12–26% during the approach.
- **8353, an end-on view.** All nine window pixels are on the spur, and the
  pixel is centred. The camera sees the spur 10.4° off its own axis, so three
  pixels span 36 mm of the spur's own depth, over the 25 mm limit. The camera
  sits on a fixed side of the jaw gap (`blender_demo_scene.py`); the mirrored
  side of the same gap sees this spur at 51.8°. Tree1 spur 19264, refused at
  startup as not visible, is the same geometry (10.2°; its mirrored side sees it
  at 32.4°).
- **12142, a real occlusion.** An orchard trellis wire passes 3–5 cm in front
  of the camera; the world-jump and flow checks catch it. No perception change
  addresses it.

The tree1 refusal of unlisted spurs existed only because the component resolver
was hard-coded to tree0; the resolver now takes a tree index, verifies
`tree1.usdc` against the export hash, and reproduces all listed tree1
candidates exactly (vertex IDs identical, centres within 1e-7 m). The spur
14884 "no drop" is a rigid-piece artefact: its hull is wedged between its
neighbours in the spur chain at release and it moved 1.1 mm; physics repeats
bit for bit, so it will repeat.

## Variants (labelled rows in `tools/queue_vision_robustness.py`, `STRATEGIES`)

| Name | Path | Tracker motion model | Camera side | Touches |
|---|---|---|---|---|
| `baseline` | straight, 4 mm | translation (median flow) | fixed | — |
| `similarity_tracker` | straight, 4 mm | **similarity** (scale, rotation, translation fitted with `cv2.estimateAffinePartial2D` over the features that already passed every unchanged check; median flow as fallback) | fixed | none |
| `mount_side` | straight, 4 mm | translation | **mirrored when the default side sees the spur within 20° of end-on**, decided once at startup | none; reads the known target axis (a known-map choice, like the seed pixel) |
| `similarity_mount` | straight, 4 mm | similarity | mirror if end-on | as both |

Offline evidence for the similarity model (replay of all 46 recorded runs):
590's final sideways drift falls from 0.41–0.55 to 0.03–0.09 of the spur's
half-width and all five 590 runs pass their recorded failing frame with the
world target within 5 mm; no passing run changes; one earlier loss (8353
fine-step, frame 2 instead of 6, with 6 features). Two variants were
considered and not registered: re-centring the depth window and the
single-pixel window both "survive" 590 while its tracked pixel keeps drifting
and 8353's target sits 15–22 mm off; both change what the spread gate measures.

Offline evidence for the mount rule: over the 14 recorded registered targets
it reproduces every recorded default mount exactly and mirrors exactly two
(8353, 19264). A ray caster over the scene meshes, which reproduces recorded
RTX depth windows to 0.1 mm, shows that on a straight approach with centred
tracking 8353's default side fails the spread gate at 6 of 20 path samples and
loses the spur behind other wood at 10 of 20; the mirrored side 0 and 0. Robot
self-occlusion on the mirrored side is not modelled.

## Batches and populations

Every population is reported separately and never pooled.

| Batch | Variant | Register | Trials |
|---|---|---|---|
| `perc-similarity-tree0-20260926` | `similarity_tracker` | [tree0 ten](evidence/eval_targets_tree0_2026-09-23.json) | 10 |
| `perc-similarity-tree1-20260926` | `similarity_tracker` | [tree1 listed seven](evidence/eval_targets_tree1_listed_2026-09-23.json) | 7 |
| `perc-mount-flip-20260926` | `mount_side` | [the two targets the rule mirrors](evidence/eval_targets_mount_flip_2026-09-26.json) | 2 |
| `perc-both-tree0-20260926` | `similarity_mount` | tree0 ten | 10 |
| `perc-both-tree1-20260926` | `similarity_mount` | tree1 listed seven | 7 |
| `tree1-listed-repeat-r1/r2/r3-20260926` | `baseline` | tree1 listed seven, three separate batches | 21 |
| `tree1-listed-morning-20260926`, `-evening-` | `baseline` | tree1 listed seven, morning and evening light | 14 |
| `tree1-seeded-ten-20260926` | `baseline` | [the tree1 half of the Sept 23 register](evidence/eval_targets_tree1_seeded_2026-09-26.json), presented for the first time | 10 |
| `tree1-seeded30-20260926` | `baseline` | [a new seeded draw of 30](evidence/eval_targets_tree1_seeded30_2026-09-26.json) over the 259 screened tree1 spurs (seed 20260926; overlaps earlier registers only at 14944) | 30 |

The mount rule changes only the two targets it mirrors, so `mount_side` runs
only on those; every other target's camera is byte-identical to the baseline
by construction. The repeats are separate batches because the launcher refuses
a duplicated target inside one plan; each repeat is its own row, never merged.
Canonical placement moves every target to one world point with the tree at
150°, which puts the robot's home pose inside the canopy for most interior
tree1 spurs: the seeded tree1 rates will be dominated by layout refusals by
design, and they are reported with that stated. The per-target CPU
presentability predictions (known-map, never a filter) are
[committed before any run](evidence/tree1_presentability_predictions_2026-09-26.json):
7 of the seeded ten and 22 of the seeded thirty are predicted to overlap the
robot at home.

## Pre-registered predictions

- **P1 (similarity, tree0).** Spur 590 does not stop with `mixed_surfaces`
  before the cut phase reaches `align`. 8353 still stops `vision_invalid`
  within its first 10 frames and 12142 still stops at frames 27–33 with a world
  jump, flow or appearance loss. Every hazard-contact, ToF and layout-refusal
  target keeps its class. *Refuted if* 590 stops `mixed_surfaces` before align,
  or if 8353 or 12142 changes class (the change would be less specific than
  claimed).
- **P2 (similarity, tree1).** 14944 and 15004 pass; every other listed spur
  keeps its class. *Refuted if* either pass fails any check.
- **P3 (mount side).** 19264 initializes (its seed is visible). 8353's first
  non-tracking frame comes later than frame 7; if it then stops with
  `mixed_surfaces`, the window holds pixels off the spur (a drift like 590),
  not nine on the spur. *Refuted if* 19264 is again refused as not visible, or
  8353 stops within 7 frames or with a nine-on-spur `mixed_surfaces` window.
- **P4 (both, registered populations).** On the tree0 ten, `vision_invalid`
  falls from three targets to at most one (12142). Hazard-contact, ToF and
  layout classes are unchanged. The tree1 listed seven pass at least 2 of 7,
  with 14944 and 15004 still passing. *Refuted if* 590 or 8353 stops
  `vision_invalid`, or if 14944 or 15004 fails.
- **P5 (repeats).** 14944 and 15004 pass in 3 of 3 repeats each. 14884 fails
  only the drop check with a drop under 2 mm. 19145 is refused at 60.554 N,
  bit-identical. 19264 stops as not visible. 19384 and 19444 stop at ToF
  clearance within one frame of 45 and 44. *Refuted by* any change of outcome
  class.
- **P6 (light).** Every listed spur keeps its source-light outcome class under
  morning and evening. *Refuted by* any class change; a pass turning into
  `vision_invalid` under evening would be the first light-dependent outcome on
  a registered target.
- **P7 (the seeded ten, first presentation).** 15599, 37023 and 37796 are
  presented (no refusal, visible seed). At least 4 of the other 7 are refused
  at startup. At most 3 of 10 pass. *Refuted if* any of the three is refused or
  not visible, or fewer than 4 of the seven are refused.
- **P8 (seeded thirty).** Layout refusals plus not-visible exceed 40% of the
  30. No pass comes from a target predicted to overlap at home. *Refuted if*
  they are under 30%, or a predicted-overlap target passes.

## What this cannot show

Nothing about 12142 or the contact and ToF targets, which belong to the
approach-planning experiment. The similarity model's replay covers genuine
camera motion only up to each recorded stop (590: 13–42 of the ~67 frames an
approach needs); the closed-loop run is the test. The mount rule's analysis
excludes the robot and jaw proxy from the ray caster. Nothing here is a field
success rate.

## Cost

111 trials of 200 frames on `ampere` A40, 25 minutes reserved per task (15
measured on a full recorded run, under 1 on a refusal): **2,775 GPU-minutes
reserved**, one task at a time within each batch. About 50 GB of captures
(450 MB per recorded run); each batch runs the launcher's storage preflight
against the 1.7 TB line. Every recorded run is composed afterwards into a
labelled GIF, MP4 and poster on CPU, pass or fail.
