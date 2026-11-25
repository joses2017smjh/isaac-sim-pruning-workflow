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
