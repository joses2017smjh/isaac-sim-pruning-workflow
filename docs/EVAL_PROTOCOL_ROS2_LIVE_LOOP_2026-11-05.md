# Live ROS 2 loop with modelled message delay, and the operator Hold — protocol, November 5, 2026

Registered **before** submission. Until now ROS 2 in this repository replayed
recordings: the arm never moved on a ROS 2 command
([ROS 2 SIL](ROS2_SIL.md)). This experiment closes the loop. Isaac Sim publishes
the wrist camera, depth, time-of-flight, contact and joint state over ROS 2;
the unchanged controller and the C++ safety gate run as ROS 2 nodes; the
simulated UR5e moves only on the gate's command. It then asks what message
delay does to the result, and whether the operator panel's Hold button stops
the arm.

**User confirmation.** On November 5, 2026 the user asked for the three
additions and approved about **500 GPU-minutes**, with commits pushed to
`develop`. This batch reserves at most 455 GPU-minutes (13 tasks × 35 min); the
exploratory smoke run reserved 35. The batch is submitted only after the smoke
run has met R1 below and this protocol and its scorer are committed.

## The known target

`tree1_SPUR_component_14944`, source daylight, raw tracker input, the unchanged
**baseline** controller (straight approach, 4 mm steps), no labelled arm. Under
exactly this condition the in-process controller passed all 17 sequence checks
in **5 of 5** recorded runs: `tree1-listed-baseline-20260923`,
`tree1-listed-repeat-r{1,2,3}-20260926` and `tree1-seeded30-20260926` run 15
(it also passed 1 of 1 in morning light and 0 of 1 in evening light, which are
other conditions). It is the target with the most repeat evidence under the
unchanged controller.

The résumé's selected run, job `21328323` (Sept 14, tree 0 component 530),
cannot be re-run as it was: the jaw proxy's closing axis was corrected after
September 14, and under current code the same baseline approach on 530 stops on
hazard contact (`targets-source-20260923`). Target 530 passes today only with
the labelled jaw-hold arm, which also failed its October 6 control. Choosing
14944 avoids testing ROS 2 on a condition that already fails in-process.

## The loop

- **Isaac Sim** (`hpc/inner/render_pruning_workflow.py`, `PRUNING_ROS2_LOOP=1`)
  renders, steps and records as before. `RosControllerProxy` replaces the
  in-process controller: each frame's bundle is published through Isaac Sim's
  bundled ROS 2 Humble `rclpy`, and the simulator waits for the gated answer.
  Its in-process ToF guard is off; the C++ gate checks ToF.
- **Controller** (`pruning_sil.live_controller`, SIL container): the unchanged
  `VisionPruningDemo`, constructed from the simulator's published config. It
  learns each verdict before observing the next bundle and acknowledges or
  stops as the recorder does.
- **Safety gate** (`pruning_sil_cpp safety_gate_node`, C++17): the ToF clearance
  and cut checks, matched to the Python on 43,623 recorded cycles
  ([evidence](evidence/cpp_safety_gate_parity_2026-11-05.json)). The arm's
  command is `/pruning/gated/command_pose`; the jaw follows `/pruning/gated/jaw`.

**Lockstep and delay.** The simulator pauses while it waits, so ROS 2 transport
time never becomes simulated time; wall-clock round trips are recorded but are
not the delay under test. Delay is modelled in simulated time: in each 0.1 s
frame the first `d` of the six 1/60 s control steps still apply the previous
gated command. `d` = 0, 2, 4, 6 gives **0, 33.3, 66.7 and 100 ms**. The delay
applies to arm commands; the jaw surrogate and detachment follow the gated jaw
at capture time.

## Runs

Frozen batch `ros2-loop-delay-20261105`, one GPU at a time, 35 minutes per task:

| Index | Delay | Purpose |
|---|---|---|
| 0, 4, 8 | 0 ms | same result through ROS 2 |
| 1, 5, 9 | 33.3 ms | delay |
| 2, 6, 10 | 66.7 ms | delay |
| 3, 7, 11 | 100 ms | delay |
| 12 | 0 ms + panel | the operator panel clicks its own Hold button after 60 gate decisions |

Delays interleave so no delay runs at a different queue position. The smoke run
(`ros2-loop-smoke-20261105`, 0 ms) is exploratory, reported separately and never
pooled with these.

## Pre-registered predictions

Scored by `tools/score_ros2_live_loop.py`, committed with this protocol. A run
counts as a pass when its experiment result is `ok`: capture complete, all 17
sequence checks, and the configuration (including the delay) matches the plan.

- **R1 — ROS 2 drove the arm.** In every one of the 13 runs: the report says the
  loop was on with the planned delay and the in-process ToF guard off; every
  frame records the gate's decision and gated command for its cycle; no cycle
  timed out; and every applied arm command equals the gated command for its
  cycle (root frame + recorded root offset, within 1e-5 m), except where the
  simulator itself stopped (tracking error), which is listed.
- **R2 — same result at 0 ms.** All 3 zero-delay runs (0, 4, 8) pass. Refuted if
  any fails.
- **R3 — what delay does (descriptive, the headline).** Pass counts at each
  delay, and the **first failing delay**: the smallest tested delay at which
  fewer than 3 of 3 runs pass, or "none up to 100 ms". Each failure is reported
  with its failed checks and first stop reason. No numerical prediction of the
  first failing delay is registered. Three runs per delay is coarse; a single
  failure at one delay is reported as such, not as a threshold.
- **R4 — Hold stops the arm (run 12).** (a) The panel's record shows the
  synthesized click and the published hold; (b) the gate's first held cycle
  carries `operator_hold` and reason `operator_hold`; (c) no gate decision
  published after the press passes, at most one passing command (the one
  already in flight) is applied after the press, and every command from the
  first held cycle to the end of the run equals the measured pose. Reported:
  the press-to-applied-hold latency in milliseconds (wall clock, same node) and
  in control cycles. Registered: **at most one control cycle**. Run 12 is
  expected to fail its sequence grade (it never cuts) and is not scored for R2.
- **R5 — the panel keeps up (run 12).** Every gate decision logged while the
  panel ran is in the panel's record; every pass/hold change is followed by a
  repaint showing that decision before the next decision arrives; the panel
  receives every wrist frame the simulator published. Reported: median and
  95th-percentile receive and repaint latency.

## What this cannot show

- Simulation only. No sensor, controller or arm hardware is involved, and the
  ROS 2 graph runs on one node over loopback.
- The delay is a modelled transport delay inside a lockstep loop. It does not
  reproduce middleware jitter, dropped messages under load or a real control
  clock, and it is not a latency guarantee.
- One target, one light, three runs per delay. A pass count is not a rate.
- The panel runs offscreen; its screenshots and records are the evidence that
  it displayed what it received, not a usability claim.

## Cost

At most 13 × 35 = 455 GPU-minutes reserved (time limits); recent 200-frame runs
took 20–30 minutes. Storage about 450 MB per run. The share is above its soft
quota (1.537 TiB used of 1.5 TiB, grace ending around November 12); the
preflight's refusal line is the 2 TiB hard limit, as set on September 27.

## Smoke run and submission — November 5, 2026

The exploratory smoke run (`ros2-loop-smoke-20261105`, frozen at `37b92d5`, GPU
job 22039692, 16.8 GPU-minutes) **passed**: capture complete, 17/17 sequence
checks, configuration including the ROS 2 loop confirmed, outcome
`vision_guided_simulated_detachment_and_retreat`. The piece dropped 0.809 m
and the arm returned within 0.25 µm of home. Against R1, all 200 frames
carried a gate decision, none timed out, the in-process ToF guard was off, and
frames 1–199 applied exactly the gated command. Frame 0 could not be checked:
that code did not yet record the t = 0 cycle's gated answer, which `da918c5`
adds. The gate held no cycle. The median sensor-to-gated-command round trip was
1.9 ms (wall clock, lockstep; not simulated time). The smoke's first workspace
build exited 1 after building, on a hash of the absent panel; its trial was
resubmitted once without the dependency (`resubmission.json` in the batch). It
is reported here and never pooled with the registered runs.

The registered batch is submitted from the commit that adds this section.
