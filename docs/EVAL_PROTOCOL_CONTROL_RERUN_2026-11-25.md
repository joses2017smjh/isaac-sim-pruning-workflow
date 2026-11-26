# Control rerun on a reachable spur — protocol, November 25, 2026

Registered **before** submission. The jaw self-mask and closure-hold arm had
one control that engages the hold: target 530. An audit found the robot beneath
the orchard's ground in every recording of it
([ground audit](GROUND_AUDIT_2026-11-25.md)), including the October 2 "3 of 3".
On October 6 the same arm passed once each on four spurs the robot reaches from
above the ground, so single passes exist. A repeated control in a physical
scene does not.

**User confirmation.** On November 25, 2026 the user approved three repeats on
one spur, capped at **135 GPU-minutes**, and one run with the ToF ground switch
on, capped at **45 GPU-minutes**.

## The one change

The target. The arm is the jaw-in-view arm, unchanged, under the name
`planned_pose_jaw_hold_ctl`; a test pins that the name is the only difference.
The mask, the hold, the four checks waived on held frames, every threshold, the
grader, the light (source), the 200 frames and the scene are as in the
[jaw-in-view protocol](EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md).

**Label.** Every result is *known-map plan; jaw self-mask; closure hold with
freshness and frame-reuse checks waived on held frames*. It is reported apart
from every unchanged-gate result and never pooled.

The fourth run adds one thing more: `planned_pose_jaw_hold_ctl_tof_ground` sets
`PRUNING_TOF_GROUND=1`, which adds the ground relief to the ToF sensors' mesh
list. It changes no gate, tracker or controller setting. It is the first
simulator run of that switch, and the three repeats are the first simulator
runs since the recorder was edited to carry it (`65c307d`).

## Targets, chosen by rule

- **Repeats: tree0 21386.** Of the four October 6 sweep targets that passed
  17/17 with nothing beneath the ground (8175, 21386, 28610, 36786), take those
  whose recorded closure the centred-spur jaw reading accepts as one spur and
  nothing else in the jaw, then the lowest component number. 8175 drops out:
  another spur crosses its jaw and its spur sits up to 6.5 mm off centre
  ([register](evidence/eval_targets_control_rerun_2026-11-25.json)).
- **Switch run: tree1 14944.** Of the reachable spurs this arm has passed on,
  take the one whose October 6 capture gains the most ToF zones from the ground
  in the offline replay: 14944, with 4,638 zones over 200 frames. For 21386 the
  ground is in no ToF zone, so a run there would not exercise the switch
  ([register](evidence/eval_targets_control_rerun_tof_ground_2026-11-25.json)).

## Runs

The launcher refuses a repeated target inside one plan, so each run is its own
batch. Four batches of one task each, `afterany`-chained, on `gpu,ampere` with
the constraint `a40|rtx8000`, in 45-minute tasks, all frozen from the
registration commit. Each run is labelled with its GPU node.

| Batch | Target | Arm |
|---|---|---|
| `control-rerun-a-20261125`, `-b`, `-c` | 21386 | `planned_pose_jaw_hold_ctl` |
| `control-rerun-tof-ground-20261125` | 14944 | `planned_pose_jaw_hold_ctl_tof_ground` |

## What has already been seen

- The October 6 captures of both targets under the same arm. 21386 passed
  17/17: closure started at frame 71, the hold covered 73 to 77 and the piece
  detached at 77. 14944 passed 17/17 and never held.
- The ground audit of those captures (nothing beneath the ground in any frame)
  and the offline ToF replay of both.

## Pre-registered predictions

Scored by `tools/score_control_rerun.py`, committed with this protocol. A run is
**scored** when its configuration matches the plan, it recorded 200 frames in
order on the registered target, and R0 holds for it. An unscored run is listed
and keeps a prediction from being supported without refuting it.

- **R0 — the scenes are physical.** In every frame of every run nothing is
  beneath the ground: the tool point, the jaw mouth, the wrist camera and every
  link hull, by `tools/audit_ground.py`, with forward kinematics reproducing the
  recorded tool point within 2 mm.
- **R1 — three of three.** Each repeat on 21386 passes all 17 checks of the
  unchanged grader. *Refuted* if a scored repeat fails a check.
- **R2 — the hold engages as before.** In each repeat the hold covers closure
  start + 2 to closure start + 6 and the piece detaches at closure start + 6.
  *Refuted* by a scored repeat with any other pattern.
- **R3 — with the switch off the recorder is unchanged.** Each repeat lists the
  collision meshes of the October 6 capture of 21386 and no ground mesh, and
  the offline trees-only cast reproduces its ToF grids up to the cut (at least
  98% of zones agree on validity and at least 95% of the zones valid in both
  agree within 1 mm). *Refuted* by a scored repeat that differs.
- **R4 — with the switch on the sensors see the ground.** The run on 14944
  lists the ground mesh among the ToF targets, the offline cast with the ground
  reproduces its ToF grids by the same criterion, and the trees-only cast does
  not. *Refuted* if any of the three fails.
- **R5 — the switch changes no outcome above the ground.** The run on 14944
  passes all 17 checks, never holds and has no ToF stop, as on October 6.
  *Refuted* otherwise.

## What this cannot show

- One spur for the repeats and one for the switch. Three repeats say how often
  render noise changes the outcome on that spur, not on others.
- It is the labelled arm: a known-map plan, the jaw self-mask and a hold that
  waives two checks on held frames. It says nothing about the unchanged gate.
- Jaws are visual only, release is a kinematic detach, depth is exact.
- R4 and R5 are one run. The switch stays off by default afterwards; making it
  the default is a separate decision.
- Nothing here is hardware.

## Cost

Four tasks at a 45-minute limit reserve 180 GPU-minutes. The October runs of
this arm took 23 to 30 minutes each on an RTX 8000, so about 95 to 120 are
expected. Storage is about 0.7 GB per run. Scoring is one CPU job.

## Submission — November 25, 2026

All four batches were submitted from the registration commit `d351016` and
froze identical code. CI was green there (Foundation, ROS 2 SIL and Pages).

| Batch | Target | Array job | Reserved |
|---|---|---|---|
| `control-rerun-a-20261125` | 21386 | `22152681` | 45 GPU-minutes |
| `control-rerun-b-20261125` | 21386 | `22152682` (`afterany` `22152681`) | 45 GPU-minutes |
| `control-rerun-c-20261125` | 21386 | `22152683` (`afterany` `22152682`) | 45 GPU-minutes |
| `control-rerun-tof-ground-20261125` | 14944 | `22152684` (`afterany` `22152683`) | 45 GPU-minutes |

The share's project usage was 1.73 TiB at submission (`lfs quota`), over the
1.5 TiB soft limit and under the 2 TiB hard limit; the storage preflight passed.

To score, once all four have ended, from a clean clone on a node with `pxr`:

```
cd /nfs/hpc/share/sanchej7/pruning_work/grading
sbatch --constraint=el9 --mem=20G score.sbatch d351016 tools/score_control_rerun.py \
  /nfs/hpc/share/sanchej7/pruning_work/control_rerun_20261125/control_rerun_verdicts_d351016.json \
  --batches artifacts/vision_robustness/control-rerun-a-20261125 \
            artifacts/vision_robustness/control-rerun-b-20261125 \
            artifacts/vision_robustness/control-rerun-c-20261125 \
            artifacts/vision_robustness/control-rerun-tof-ground-20261125
```

Nothing above this line changes after submission.

## Result — November 25, 2026

All four runs ended `COMPLETED` and nothing was resubmitted. The registered
scorer ran from a clean clone of the registration commit `d351016` on a node
with `pxr` (CPU job `22153680`). Its verdict file is copied byte for byte to
[`evidence/control_rerun_verdicts_2026-11-25.json`](evidence/control_rerun_verdicts_2026-11-25.json).
All four runs were scored and **all six predictions are supported**.

| Prediction | Verdict | Measured |
|---|---|---|
| **R0** the scenes are physical | supported | Nothing beneath the ground in any of the 800 frames; forward kinematics reproduces the recorded tool point within 0.007 mm |
| **R1** three of three | supported | Each repeat on 21386 passed 17/17 |
| **R2** the hold engages as before | supported | In each repeat closure started at frame 71, the hold covered 73 to 77 and the piece detached at 77 |
| **R3** switch off, recorder unchanged | supported | Each repeat lists the October 6 collision meshes and no ground mesh, and the trees-only cast reproduces all 9,984 compared ToF zones |
| **R4** switch on, the sensors see the ground | supported | The run on 14944 lists `/World/Orchard/ground/ground_mesh`; the cast with the ground reproduces all 9,984 zones; the trees-only cast agrees on validity in 82.7% of them, under the 98% bar |
| **R5** the switch changes no outcome above the ground | supported | The run on 14944 passed 17/17, never held and had no ToF stop |

| Batch | Target | Array job | Node (GPU) | Elapsed | Grader | Closure start | Held frames | Detached |
|---|---|---|---|---|---|---|---|---|
| `control-rerun-a-20261125` | 21386 | `22152681` | `cn-gpu5` (RTX 8000) | 27:18 | 17/17 | 71 | 73–77 | 77 |
| `control-rerun-b-20261125` | 21386 | `22152682` | `cn-gpu7` (RTX 8000) | 24:03 | 17/17 | 71 | 73–77 | 77 |
| `control-rerun-c-20261125` | 21386 | `22152683` | `cn-gpu7` (RTX 8000) | 23:49 | 17/17 | 71 | 73–77 | 77 |
| `control-rerun-tof-ground-20261125` | 14944 | `22152684` | `cn-gpu7` (RTX 8000) | 27:11 | 17/17 | 71 | none | 77 |

In the verdict file the switch run carries `as_registered: false`. That field
tests the repeats' hold pattern; R5 predicted no hold on 14944, so `false` is
the expected value there.

### How to read it

- **It is 3 of 3 on one spur under the labelled arm.** *Known-map plan; jaw
  self-mask; closure hold with freshness and frame-reuse checks waived on held
  frames.* It is not pooled with the October 6 pass on the same spur or with
  any other result.
- **It is not a success rate.** 21386 was chosen by rule from spurs this arm had
  already passed on. Three repeats say the outcome on that spur did not change
  from run to run. They say nothing about other spurs.
- **It does not restore the 530 control.** The October 2 "3 of 3" on 530 stays
  as recorded, with its dated note. This is a different spur, and the first
  repeated control of this arm in a scene with nothing beneath the ground.
- **R2 says the hold engaged, not that it was needed.** The scorer checks which
  frames were held and when the piece detached. It does not test what the cut
  would have done without the hold.
- **R3 rests on the mesh list.** On 21386 the ground is in no ToF zone, so the
  cast with the ground reproduces the repeats' grids just as well as the
  trees-only cast does. What shows the switch was off is the recorded mesh
  list, which is the October 6 one with no ground mesh.
- **The ToF comparison covers frames 0 to 77**, up to the cut: 78 frames, two
  sensors, 64 zones each, 9,984 zones per run. Later frames were not compared.
- **R4 and R5 are one run, and the switch never had to stop anything.** The
  ground was in view and far away. The offline replay's finding that the switch
  would hold every run on 530 at frame 0 is still a replay; no simulator run
  has put the robot beneath the ground with the switch on.
- **The switch stays off by default.** Making it the default is a separate
  decision that has not been made.

### Observations outside the registered scoring

Computed after scoring, from the working tree, by ad hoc scripts. They are not
part of any verdict.

- **The repeats are not copies of each other.** Between repeats the recorded
  tool position differs by up to 2.2 mm and a joint position by up to
  0.004 rad. The controller phase is the same in every frame of all three
  repeats and of the October 6 capture.
- **The switch added about as many ToF returns as the replay expected.** Over
  200 frames the switch run has 10,039 valid zone readings of 25,600; the
  October 6 capture of the same spur has 5,403. The difference is 4,636. The
  offline replay had counted 4,638 zones gained from the ground at the October 6
  poses.
- **The clearance check was never approached.** The nearest valid ToF return is
  83 mm in the switch run and in the October 6 capture, against the 60 mm limit.

### Cost

One GPU per task (`sacct`, `gres/gpu=1`).

| | Used | Cap |
|---|---|---|
| Three repeats | 75 min 10 s | 135 GPU-minutes |
| Switch run | 27 min 11 s | 45 GPU-minutes |

The unused part of each cap is not carried over to anything else. Storage is
about 0.34 GB per run, 1.4 GB in all, against the 0.7 GB per run estimated.
