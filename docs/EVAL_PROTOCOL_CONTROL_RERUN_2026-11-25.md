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
