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

---

## Result — September 24, 2026

Arrays `21404500` (tool-axis standoff), `21404502` (horizontal standoff),
`21404503` (fine step) and `21404504` (tree1 listed baseline): 37 of 37
planned trials produced a graded row; no trial timed out (the 400-frame fine
step took 20–30 min against its 50-min limit). Code revision `1c8ef03`. The
four arrays ran independently rather than chained (a wrong result key in the
submission loop, disclosed in the job ledger); each still ran one task at a
time. Every recorded run has a GIF, MP4 and poster from
`tools/compose_batch_media.py`.
[Strategies evidence](evidence/strategies_2026-09-24.json) ·
[tree1 evidence](evidence/tree1_listed_2026-09-24.json).

### Tree0: 0 of 40 under every strategy

| Target | Baseline (Sept 23) | `tool_axis_standoff` | `horizontal_standoff` | `fine_step` |
|---|---|---|---|---|
| 530 | hazard contact @60 | hazard contact @61, after the standoff point (final phase from 55) | hazard contact @59, after the standoff point (from 50) | **ToF clearance** @131 |
| 7524 | hazard contact @10 | hazard contact @11, before the standoff point | hazard contact @11 | hazard contact @51 (same place at half speed) |
| 22988 | hazard contact @23 | hazard contact @23 | hazard contact @22 | hazard contact @45 |
| 590 | vision invalid @25 | vision invalid @21 | vision invalid @17 | vision invalid @42 |
| 8353 | vision invalid @2 | vision invalid @3 | vision invalid @1 | vision invalid @7 |
| 12142 | vision invalid @31 | vision invalid @30 | vision invalid @31 | vision invalid @60 |
| 18669 | ToF clearance @29 | ToF clearance @26 | **hazard contact** @22 | ToF clearance @56 |
| 10001, 10061, 23167 | layout refused | layout refused | layout refused | layout refused |

Rate 0/40, Wilson 95% interval 0–0.088. Failure taxonomy over the 40:
12 layout refusals, 12 hazard contacts, 12 vision invalid, 4 ToF clearance.

### Tree1 listed population: 2 of 7 pass

| Spur | Outcome | Checks |
|---|---|---|
| 14944 | **pass** | 17/17: zero contact force, 68 applied vision commands, piece dropped 0.81 m, returned home within 1.4 µm |
| 15004 | **pass** | 17/17: as above, minimum tracker confidence 0.51 |
| 14884 | closure reached, release recorded, piece did not fall (1 mm) | 16/17, `piece_dropped_after_release` |
| 19384 | ToF clearance stop @45 | 9/17 |
| 19444 | ToF clearance stop @44 | 9/17 |
| 19264 | initial target not visible from the approach pose | 8/17 |
| 19145 | layout refused at startup | 0/0 |

Rate 2/7, Wilson 95% interval 0.08–0.64. These are the first recorded passes
on any registered target. They are on the export manifest's own listed
candidates under the baseline strategy and the source light, a population
chosen by the renderer's acceptance list rather than by the seeded draw, and
seven spurs is what that list holds; the interval says how little the rate
is worth. Spur 14884's release is a rigid-piece artefact: the piece was
detached and stayed where it was.

### Verdicts

- **P1 half supported.** On 530 both standoff strategies reach the standoff
  point and the contact comes in the final approach, so the prediction holds
  there; on 7524 the `mock_pruner__base` touches wood at frame 10–11 with the
  mouth still 0.31 m from the target, before any standoff logic acts, so the
  prediction is refuted there. 22988 stops identically under every strategy,
  as registered. The contacts on 7524 and 22988 happen on the way in from the
  home pose, which no mouth-path rule changes; that is a planning problem.
- **P2 formally not refuted, in substance refuted.** Under `fine_step` the
  three vision-invalid targets stop at frames 42, 7 and 60 instead of 25, 2
  and 31, which the registered wording counts as "a later stop"; but the
  frame ratios (1.7×, 3.5×, 1.9×) are what half speed produces at the same
  spatial point, and the reason is unchanged. The standoff strategies leave
  all three unchanged (frames within 4), as registered. Tracker depth at the
  spur edge does not depend on the step.
- **P3 supported.** 18669 stops at ToF clearance under `tool_axis_standoff`
  and `fine_step`; under `horizontal_standoff` the stop reason changes, to a
  hazard contact at frame 22.
- **P4 supported.** The three refused layouts are refused identically 12 times.
- **P5 supported.** No strategy passes any tree0 target; the tree1 listed
  register records frames for 6 of 7 and passes 2.

### What this settles

The mouth's path is not what fails on the registered tree0 targets. Two
targets are hit by the pruner body or the upper arm on the way in, three lose
the tracker's depth gate at the spur edge within the first three seconds, one
sees neighbouring wood in a ToF zone, three cannot be set up without contact.
Changing the step size moves the same failures later in time and one of them
(530) from a body contact into a ToF stop. What passed, twice, was a listed
tree1 spur on a straight path: thin (6.7 mm), unobstructed, tracked at 0.5–0.6
confidence, with the piece falling after release. The next levers are
collision-aware approach planning for the body and arm, and the tracker's
depth window at spur edges, each its own registered experiment.

