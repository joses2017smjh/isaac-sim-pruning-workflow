# Longer episode at 66.7 and 100 ms command delay — protocol, November 25, 2026

Registered **before** submission. On November 5 the live ROS 2 loop made the cut
at every command delay, but at 66.7 and 100 ms all six runs failed one check,
`returned_home_and_final_phase_complete`: the fixed 20 s episode ended while the
arm was still retreating
([result](EVAL_PROTOCOL_ROS2_LIVE_LOOP_2026-11-05.md#result--november-5-2026)).
An explanation was fitted afterwards, not registered
([addendum](EVAL_PROTOCOL_ROS2_LIVE_LOOP_2026-11-05.md#addendum--november-22-2026-after-the-result-not-registered)):
each command is the measured pose plus a step of at most 4 mm, so a late
command overlaps the next and the arm simply moves slower. If that is the whole
story, the same runs pass with a longer episode. This experiment tests it before
the explanation is repeated anywhere.

**User confirmation.** On November 24, 2026 the user approved about **240
GPU-minutes** for this experiment.

## The one change

The episode is **300 frames at 10 fps (30 s)** instead of 200 frames (20 s).
Everything else is the November 5 condition, frozen at the registration commit:
target `tree1_SPUR_component_14944`, source daylight, raw tracker input, the
unchanged baseline controller (straight approach, 4 mm steps), the live
lockstep ROS 2 loop with the C++ safety gate, the same SIL container image, and
the same modelled delay (the first 4 or 6 of the six 1/60 s control steps in
each frame still apply the previous command).

The recorder's scripted schedule depends on the episode length, but in vision
mode the controller sets every command, so nothing before frame 200 should
change. D3 checks that.

## Runs

Frozen batch `delay-episode-20261125`, design `long_episode` in
`tools/queue_ros2_live_loop.py`, one GPU at a time, 40 minutes per task, on
**A40** nodes only. All thirteen November 5 runs ran on A40s, so this keeps the
comparison like-for-like; an RTX 8000 renders about 1.8 times slower and a
300-frame run would not fit the limit there.

| Index | Delay |
|---|---|
| 0, 2, 4 | 66.7 ms (4 steps) |
| 1, 3, 5 | 100 ms (6 steps) |

Delays interleave so neither runs at a different queue position. No panel run.

## Pre-registered predictions

Scored by `tools/score_delay_episode.py`, committed with this protocol. A run
is **scored** when it was graded with all 17 checks, its configuration
(including 300 frames and the planned delay) matches the plan, it recorded 300
frames in order, and D0 holds for it. An unscored run is listed and keeps a
prediction from being supported without refuting it.

- **D0 — ROS 2 drove the arm.** The November 5 scorer's R1, unchanged, in every
  run: the loop was on with the planned delay and the in-process ToF guard off,
  every frame carries a gate decision, no cycle timed out, and every applied arm
  command equals the gated command.
- **D1 — all six pass.** Every scored run passes all 17 checks of the unchanged
  grader. *Refuted* if any scored run fails a check.
- **D2 — the arm is home when the model says.** The first frame in phase
  `complete` is within **10 frames of 233** at 66.7 ms and **of 277** at 100 ms.
  These are November 5's measured approach (112 and 134 frames) plus align (3)
  and closing (6), plus a retreat as long as the approach. *Refuted* if any
  scored run reaches home outside that window or never does.
- **D3 — the first 200 frames repeat November 5.** The cut (detach event) comes
  within **2 frames of 120** at 66.7 ms and **of 142** at 100 ms, as in all six
  November 5 runs. *Refuted* otherwise.

**Reading the result.** If D1–D3 hold, the November 5 failures were the episode
length, as the model says, and the 66.7 ms figure is a time budget, not a
latency limit. If D1 fails, the explanation is wrong or incomplete, and the
failed checks say how. If only D2 fails, the arm got home but not when the model
predicted, so the model's timing is off even if its conclusion holds.

**Infrastructure failures.** A run that hits its time limit, loses its node or
records no capture is unscored. Each such index may be resubmitted **once**,
with the same frozen code, only while the GPU-minutes used stay within 240. A
resubmission is recorded in the batch (`resubmission.json`) and in the result.

**Exposures.** The runner exits 0 only on a 17/17 pass with a matching
configuration, so Slurm states reveal pass or fail as runs finish. The scorer is
committed before submission, so nothing seen afterwards can shape it.

## What this cannot show

- Simulation only, on one target under one light, with every process on one
  node over loopback.
- The delay is a constant command delay inside a lockstep loop. Images still
  reach the controller undelayed, so the 0.25 s freshness check still cannot
  fire; sensor delay is a separate experiment.
- The November 5 repeats were nearly identical, so three runs per delay are one
  observation repeated, not a rate.
- It is not a latency guarantee for anything.

## Cost

At most 6 × 40 = **240 GPU-minutes** reserved (time limits), plus one CPU job
that builds the ROS 2 workspace. The November 5 runs took 16.6 minutes for 200
frames on A40s, so these should take about 25 minutes each, about 150
GPU-minutes in all. Storage about 0.7 GB per run, about 4 GB in all. The share held 1.72 TiB on
November 24, under the 2 TiB hard limit; it is over the 1.5 TiB soft limit, with
grace to about December 19, and these runs finish long before that.

## Submission — November 25, 2026

Submitted from the registration commit `bfd086a`, pushed to `develop` before
submission, with the scorer above already committed. The batch froze that code
in `artifacts/vision_robustness/delay-episode-20261125/code`.

| Job | What |
|---|---|
| `22149695` | CPU: build the ROS 2 workspace for the batch |
| `22149696` | GPU array, tasks 0–5, one at a time (`%1`), A40 only, 40 minutes each, `afterok` the build |

**Scoring, once the array has finished.** Each run writes its own
`sequence_grade.json`, so no aggregation step is needed. From a clean clone at
the registration commit:

```bash
cd /nfs/hpc/share/sanchej7/pruning_work/grading
sbatch score.sbatch bfd086a tools/score_delay_episode.py \
  /nfs/hpc/share/sanchej7/pruning_work/grading/delay_episode_verdicts_bfd086a.json \
  --batch artifacts/vision_robustness/delay-episode-20261125
```

Then copy the verdict file into `docs/evidence/` byte for byte and publish the
result here, including any failed run's failed checks.
