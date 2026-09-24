# Approach strategies against the recorded failure classes — protocol, September 23, 2026

Registered **before** submission. The
[September 23 sweep](EVAL_PROTOCOL_2026-09-23.md#result--september-23-2026)
passed 0 of 40 and stopped every recorded run at a gate. This protocol does
not touch a gate, a threshold or the grader (`tools/validate_vision_sequence.py`
is unchanged and remains the only judge). It changes one thing the gates
watch: how the mouth travels to the tracked point. Each strategy is a labelled
row in the frozen plan, forwarded to the renderer by three environment
variables and checked against the capture's own report before grading.

## What the recorded failures say

Of the 20 recorded tree0 runs (10 targets × source and morning), every stop was
in the approach phase and every target failed the same way under both lights:

| Class | Targets | What the recording shows |
|---|---|---|
| Layout refused at startup | 10001, 10061, 23167 | Robot in contact with the scene before motion (226 N to 1.6 MN); no frames. Base, yaw and target pose are fixed by the sweep protocol; no approach strategy can change this. |
| Hazard contact | 530, 7524, 22988 | Tracking healthy (confidence 0.92–0.99); the `mock_pruner__base` (530, 7524) or the `ur5e__upper_arm_link` (22988) touched wood on the straight, standoff-free path. |
| Vision invalid | 590, 8353, 12142 | Depth spread over the 3×3 window at the spur edge (`mixed_surfaces`), or a 0.24 m world jump; the cut gate latches any non-tracking frame. |
| ToF clearance | 18669 | Tracking healthy; a ToF zone saw neighbouring wood under 0.06 m at 155 mm from the target. |

## Strategies (`tools/queue_vision_robustness.py`, `STRATEGIES`)

| Name | Path | Step | Frames | Addresses |
|---|---|---|---|---|
| `baseline` | straight line from the mouth to the live target | 4 mm | 200 | the September 23 rows, re-used, not re-run on tree0 |
| `tool_axis_standoff` | first to a point 60 mm short of the target along the tool's forward axis (frozen at the first tracking frame), then along that axis | 4 mm | 200 | body contact: the pruner body stays behind the mouth |
| `horizontal_standoff` | first to a point 80 mm short along the horizontal mouth-to-target direction (frozen at the first tracking frame), then along it | 4 mm | 200 | contact and ToF: a level final approach |
| `fine_step` | straight line | 2 mm | 400 | per-frame image motion at the spur edge |

The approach axis is never recomputed from scene metadata; it is recorded in
the run evidence together with the phase (`standoff`, `final`) of every
command. `ApproachStrategy` refuses a standoff without a direction and a step
over 10 mm; a test asserts that every strategy leaves the tracker and cut
configurations identical to the baseline. The single-pixel depth window was
considered and rejected: it makes the depth-spread gate vacuous, which is a
gate change in disguise.

## Populations

- **Tree0**, the 10 registered September 23 targets
  ([register](evidence/eval_targets_tree0_2026-09-23.json), the tree0 subset of
  the original file, unchanged), under the `source` light only: the September
  23 result showed light did not change any outcome. 10 targets × 3 new
  strategies = 30 trials.
- **Tree1**, a **different population**: the 7 spurs the export manifest lists
  as candidates and that pass the 12 mm jaw screen
  ([register](evidence/eval_targets_tree1_listed_2026-09-23.json), built by
  `tools/enumerate_spur_targets.py --listed-only`, no sampling). Baseline
  strategy, source light: 7 trials. This is the first recorded tree1 data.

N never shrinks: a refused layout, a crash and a timeout are all rows. The
grader's 17 checks decide pass/fail; the SQL failure taxonomy of the sweep is
reused per strategy.

## Pre-registered predictions

- **P1 (hazard contact).** `tool_axis_standoff` removes the `mock_pruner__base`
  contact on targets 530 and 7524: neither stops with `hazard_contact` before
  reaching the standoff point. Target 22988 (upper arm) stops the same way under
  every strategy. *Refuted if* 530 or 7524 stops with `hazard_contact` during
  the standoff phase under `tool_axis_standoff`.
- **P2 (vision invalid).** `fine_step` changes the outcome class of at least one
  of 590, 8353, 12142 (a later stop, a different reason, or a pass); the two
  standoff strategies leave all three unchanged, since the tracker is the same.
  *Refuted if* all three stop with `vision_invalid` at the same frame ±2 under
  `fine_step`.
- **P3 (ToF clearance).** 18669 stops at `tof_minimum_clearance` under
  `tool_axis_standoff` and `fine_step`; `horizontal_standoff` changes its stop
  reason or frame. *Refuted if* the horizontal path stops identically.
- **P4 (layout refusals).** 10001, 10061 and 23167 are refused at startup under
  every strategy, identically.
- **P5 (rate).** No strategy passes more than 3 of 10 tree0 targets; any pass at
  all is the first recorded pass on a registered target. The tree1 listed
  register records frames for at least 5 of 7 and passes at most 2.

## Cost

Per trial 25 minutes reserved (measured 16.5 on the sweep); the fine step's
400-frame episode gets 50. Reserved: tree0 `tool_axis_standoff` 170 min,
`horizontal_standoff` 170 min, `fine_step` 340 min, tree1 baseline 119 min,
**799 GPU-minutes in total, one A40 at a time, arrays chained with `afterany`**
(about 13 hours of wall time; expected use roughly 60%). Each recorded run
writes about 0.42 GB (0.84 GB for the fine step), about 20 GB in all. Every run
is turned into a labelled GIF, MP4 and poster afterwards on CPU by
`tools/compose_isaac_workflow.py`, pass or fail alike.
