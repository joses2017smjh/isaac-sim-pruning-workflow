# A jaw-presence input for the safety gate — protocol, November 25, 2026

Registered **before** the replay. The safety gate takes the target's position
from the controller it gates, so a tracker that is confidently wrong passes
([ROS2_SIL.md](ROS2_SIL.md), November 22 addendum). An exploratory study on
November 25 compared three ways to give the gate evidence of its own about the
jaw mouth
([study](ROS2_SIL.md#november-25-what-could-witness-the-target-in-the-mouth-exploratory)).
The user chose the beam grid, built offline first, CPU only.

This experiment tests the gate rule on recorded cycles. It does not test a
sensor: nothing in the simulation publishes the reading yet.

## What changed, and what did not

- **The reading.** `isaaclab_pruning.sensors.jaw_beam_grid` defines nine beams
  across the jaw gap, parallel to the closing axis: three positions along the
  jaw (−8, 0, +8 mm) at three heights along the tool axis (−10, 0, +10 mm from
  the mouth). A beam is blocked when anything crosses it.
- **The rule**, in the Python reference and the C++ gate. With
  `jaw_presence_min_beams` above 0, closure is authorized only while at least
  that many beams are blocked; a closure that continues with fewer latches
  `jaw_empty_during_closure`; a missing reading latches `jaw_presence_missing`.
  After detachment the rule does not apply, as for the other cut checks. This
  experiment uses **1 beam**.
- **Off by default.** At 0 the node does not subscribe to the reading and every
  verdict is what it was. `SafetyDecision` is unchanged; the reading is echoed
  in `decision_json` only.
- **Not changed:** the simulated tool, the live loop, the controller, and the
  controller's own cut check (`evaluate_cut_gate`).

## Where the readings come from

`tools/cast_jaw_beams.py` casts the nine beams offline at every recorded cycle's
tool pose and drawn jaw closure, against the exported orchard: the trees and the
ground relief. It is the caster the study validated against the recorded ToF
grids (all 29,350 comparable zones within 1 mm) and the recorded depth images
(all eleven runs, once the ground was in the mesh set). Each reading is joined
to its cycle on the cycle index, the stamp and the tool pose.

**Excluded scenes.** A capture in which the tool point is beneath the ground
relief in any cycle is not a physical scene. It is listed and kept out of J2 to
J5. The study found two such captures, both on target 530.

## What has already been seen

- The study's eleven runs, nine of them usable. In those nine a beam was
  blocked in all 54 closing frames, and with the tool 20 mm short of the ready
  pose no beam was blocked in any of the nine. The grid's spacing and the
  one-beam threshold were chosen after seeing that.
- Counts taken from the recordings without casting anything: 237 captures
  rebuild into 48,223 cycles, and 95 of them reach closing (40 on target 14944,
  22 on 15004, 11 on 8235, 8 on 530, 8 on 14884 and one each on six others).
- The unit tests of the rule, and one build of the ROS packages (CPU job
  22150805: 41 tests, 0 failures).

Not seen: a beam reading for any other capture, the rule replayed on any
recorded cycle, or the node running with the rule on.

## Procedure

Everything runs from a clean clone at the registration commit, on CPU.

1. `tools/export_gate_parity.py` rebuilds today's cycles.
2. `tools/cast_jaw_beams.py` writes the readings.
3. `tools/validate_safety_gate.py`, unchanged invocation, on the **November 5**
   cycles file (`8475d280…`), whose stored verdicts came from the gate before
   this change: the parity sweep and the 23 injected faults, jaw presence off.
4. `tools/validate_safety_gate.py --jaw-beams` on today's cycles: every capture
   that reaches closing with the tool above the ground runs through the C++ node
   three times with jaw presence on. *Recorded*: the reading at the recorded
   pose. *All clear*: every beam reported clear. *Tool short*: the reading cast
   with the whole tool 20 mm short of its recorded pose along its own axis.
   Then three input faults on one capture.
5. `tools/score_jaw_presence.py` scores the predictions below.

The node is compared with `tools/gate_replay.py`: the Python reference, plus
the one rule the node adds (the jaw output advances only on a passing cycle that
authorizes closure).

## Pre-registered predictions

- **J1 — off changes nothing.** On the November 5 cycles the node matches the
  stored verdict on every cycle, with the same number of holds as on November
  5; all 23 faults and the one documented non-stop give the same outcome, first
  hold and reason as on November 5; and the new Python reference with the rule
  off reproduces every stored verdict. *Refuted* by any difference.
- **J2 — on changes no recorded cut.** With the recorded-pose reading and the
  rule on, no capture with the tool above the ground differs from the rule off
  in any cycle's verdict, reason, latch or jaw output. Through the node, every
  such capture that reaches closing matches the reference on every field of
  every cycle, and its jaw output reaches the recorded maximum. *Refuted* by one
  changed capture.
- **J3 — an empty jaw cannot close.** With every beam reported clear, in every
  such capture the node's jaw output stays at 0 for the whole run, the node
  holds wherever the recorded jaw advanced, and its first hold is at the cycle
  and for the reason the reference gives. *Refuted* by one capture whose jaw
  output advances.
- **J4 — a jaw that stopped 20 mm short cannot close.** With the tool-short
  reading, no beam is blocked in any cycle whose certificate was ready or whose
  phase was closing, and the node refuses every closure. *Refuted* by one
  accepted closure. A blocked beam that does not lead to an accepted closure is
  reported and also refutes the first half.
- **J6 — a bad reading holds.** A dropped reading holds with
  `input_missing:jaw_presence`; a reading with the wrong number of flags, or in
  the wrong frame, holds with `invalid_jaw_presence`; each from the first
  faulted cycle to the end. *Refuted* by any of the three.

Reported without a prediction:

- **J5.** At each usable capture's first ready cycle, the study's 36
  displacements: how often a beam is blocked without the target between the
  jaws, and by what.
- **Ground audit.** Every capture with the tool beneath the ground, by target.

## What a result would and would not show

- It is a replay. The controller's messages are the recorded ones, so a hold
  does not change what the controller does next. "The jaw did not move" is read
  on the gate's jaw output.
- The reading is geometry: exact, noise-free, with beams as lines, and with the
  jaws in no mesh set. It says something crosses the gap, not that it is the
  right spur.
- J3 and J4 inject the case the recordings do not contain. They are not
  recorded failures of the tracker.
- A live confirmation needs the reading published by the simulated tool and GPU
  time. Neither is part of this experiment, and neither is approved.

## Not in this change

Counting blind ToF cycles per sensor was listed with this milestone. It touches
the ToF path and is deferred to its own change.

## Result — November 25, 2026

Run from a clean clone at the registration commit `72493fb`, on CPU: the cycles
and the readings by job `22150835`, the default harness by `22150836`, the jaw
harness by `22150837`, and the scorer by `22150888`. Verdicts:
[jaw_presence_verdicts_2026-11-25.json](evidence/jaw_presence_verdicts_2026-11-25.json),
copied byte for byte, with the
[beam summary](evidence/jaw_beams_summary_2026-11-25.json), the
[jaw harness](evidence/jaw_presence_harness_2026-11-25.json) and the
[default harness](evidence/cpp_safety_gate_parity_2026-11-25.json).

237 captures rebuilt into 48,223 cycles. 233 have a reading. In 28 the
tool is beneath the ground, which leaves **205 usable captures**, 87 of
which reach closing, on 10 distinct targets.

| Prediction | Verdict | Measured |
|---|---|---|
| J1 — off changes nothing | **supported** | The node matched the stored verdict on 43,623 of 43,623 cycles (217 captures), with 24,817 holds as on November 5. 23 of 23 faults held with the same first hold and reason. The new reference reproduced every stored verdict. |
| J2 — on changes no recorded cut | **supported** | No usable capture changed in any cycle (205 captures, 42,001 cycles). A beam was blocked in all 1,079 cycles whose certificate was ready, all 516 closing cycles and all 511 cycles in which the jaw advanced. Through the node: 87 captures, 17,911 cycles, every field matched and the jaw output reached the recorded maximum. |
| J3 — an empty jaw cannot close | **supported** | In 87 of 87 captures the jaw output stayed at 0. The node held with `jaw_empty_during_closure` one cycle after the controller announced closing, in every capture, where the reference said it would. |
| J4 — a jaw 20 mm short cannot close | **refuted** | The gate refused 82 of 87 closures and **accepted 5**. All five are on target 8235 (5 of its 11 closure captures): with the tool 20 mm short, other wood blocked one or two beams in every ready or closing cycle. The target was between the jaws in none of the 940 cycles. |
| J6 — a bad reading holds | **supported** | A dropped reading, a reading with four flags and a reading in the wrong frame each held from the faulted cycle to the end, with the registered reason. |

**J5, counts only.** At the first ready cycle of 95 usable captures, over the
study's 36 displacements (3,420 poses): the target was between the jaws in
2,080, and a beam was blocked in every one of those. In 178 of the 1,340
poses without the target a beam was blocked anyway, always by other wood. The
worst displacement was 50 mm along the jaw (64 of 94).

**Ground audit.** In 28 of the 233 cast captures the tool point is beneath
the orchard's ground relief: target 530 in 15, 590 in 7, 3721 in 3, and 3433,
3493 and 3673 in one each. In 26 of them it is beneath the ground in every
cycle, the start pose included; in two captures on 530 it is for part of the
run (30 and 192 of 200 cycles). No capture that was left uncast is on 530, so these 15 are every recorded capture on it. The cast was checked against recorded depth images on eleven runs, two of them on 530; the other 26 captures rest on the same cast. 8 of the 28 reach closing, all on 530, and
4 detach. The batches: `hold-gen-a-r1-20261004`, `hold-gen-a-r2-20261004`, `hold-gen-a-r3-20261004`, `hold-gen-ctl-20261004`, `hold-sweep-20261006`, `hold-sweep-ctl-20261006`, `jaw-hold-a-20261001`, `jaw-hold-b-20261001`, `jaw-hold-c-20261001`, `perc-both-tree0-20260926`, `perc-similarity-tree0-20260926`, `planned-pose-gpu-r1-20260929`, `planned-pose-gpu-r2-20260929`, `planned-pose-gpu-r3-20260929`, `strategy-fine-step-20260923`, `strategy-horizontal-20260923`, `strategy-tool-axis-20260923`, `targets-morning-20260923`, `targets-source-20260923`, `tree1-seeded30-20260926`.

### What this shows

- The rule does what it was built for on recorded cycles. Off, nothing
  changes. On, no recorded cut changes, and a closure with nothing reported
  between the jaws stops before the jaw moves.
- **One blocked beam is not evidence of the target.** It says something crosses
  the gap. Near target 8235 that something is neighbouring wood, 20 mm from
  where the spur should be, and the gate let the closure through. J5 says the
  same more broadly: about one displaced pose in eight without the target still
  blocks a beam.
- The four supported predictions are about the rule. J4 is about the reading,
  and it is the one that failed.

### After the result, not registered

Counting blocked beams does not fix J4 on these recordings. At the recorded
pose the fewest blocked beams in any ready or closing cycle was 2 in 8 captures,
3 in 68, 5 in 7 and 7 in 4. With the tool short the most was 0 in 82 captures,
1 in 2 and 2 in 3. A threshold of two keeps all 87 recorded closures and still
accepts 3 of the short ones; a threshold of three refuses all 87 short ones and
would have stopped 8 recorded closures. Which beams are blocked, or a reading
that says how far along the gap the obstruction is, may separate them; neither
was tested.

### Departures

- The protocol expected the excluded scenes to be on target 530. The audit
  found six targets and 28 captures.
- 4 early `isaac_render` captures carry no orchard placement and have no
  reading. None reaches closing. They are in neither J2's 205 nor the 28.
- The cycles rebuilt today include captures recorded after November 5, so a
  regenerated C++ fixture differs from the committed one. The committed fixture
  was left in place and its gtest passes.
