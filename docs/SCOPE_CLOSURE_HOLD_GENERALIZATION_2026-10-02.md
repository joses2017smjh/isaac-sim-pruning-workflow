# Closure hold on an unseen contact target — scope, October 2, 2026

**Status: scoping only.** Not registered. The user approved this follow-up in
principle on October 2. Its first step rebuilds a lost CPU tool (below). That
needs the user's go-ahead, because it is several hours of work. The GPU budget
is asked for only after the candidate search. Independent verifiers checked
the factual claims here against the committed evidence. Their corrections are
applied.

## Question

Contact target 530 passed in 3 of 3 runs with the jaw self-mask and the
closure hold
([result](EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md#result--october-2-2026)).
The hold was designed on 530's own recordings, so those runs confirm it in
closed loop but do not test it elsewhere. Does it carry an unseen contact
target through closure?

## Why there is no candidate yet

- **No other recorded run gives the hold anything to do.** With the mask and
  the hold on, the replay of all 129 recorded runs held frames only in the
  three planned-pose 530 runs, at 76–79 through the recorded stop. That is
  check `hold_only_530` of the
  [P0 recheck](evidence/jaw_in_view_p0_replay_recheck_2026-10-02.json). In
  the three planned-pose 19444 runs, the mask alone moved the stop to frame
  67, and no frame was held.
- **The diagnosed targets are used up.** Seven targets were diagnosed: six
  contact or time-of-flight stops, and 12142, where a trellis wire crossed
  the camera's line of sight. Only 530 and 19444 had a clear re-oriented
  known-map path ([diagnosis](evidence/contact_diagnosis_2026-09-27.json)).
  19444 stopped on the final leg, at frames 67–70, in all 5 of its
  planned-pose runs, when the open jaw crossed its tracked patch. The hold
  never applies there.
- **So candidates must be new.** They have to come from targets that were
  never run with a re-oriented plan.

## The missing tool

The known-map model that found 530's and 19444's paths was never committed.
Only its outputs are:
[contact_diagnosis_2026-09-27.json](evidence/contact_diagnosis_2026-09-27.json)
and
[tree1_swept_path_predictions_2026-09-27.json](evidence/tree1_swept_path_predictions_2026-09-27.json).
Git history and the remaining scratch space hold no copy. Those outputs
describe what it did:
- **Kinematics.** URDF kinematics from the recorded settled home, not the
  commanded one (shoulder lift −0.528 against −0.586).
- **Collision.** The robot's convex link hulls, as PhysX uses them, against
  the exported orchard meshes.
- **Sensors.** A re-cast of the two 8×8 time-of-flight sensors, and a camera
  line-of-sight check.
- **Motion.** A frame-by-frame replay of the baseline motion. Each frame
  takes the measured tool pose plus a step of at most 4 mm toward the
  target, solved by damped-least-squares IK (damping 0.05, a 0.05 rad joint
  step limit, 12 substeps per frame). Orientation drift is not modelled.
- **Search.** A final-pose search over 6,016 orientations.

## Plan

1. **Rebuild the model as a committed CPU tool, with tests.** Its acceptance
   test is fixed now, before any code is written. It checks passes as well as
   stops, so a tool that over-predicts blockage fails it. Every item is
   required:
   - **Recorded stops.** Reproduce every stop on the same link and object,
     0–4 frames early:
     - the contacts on 530, 7524 and 22988;
     - 18669's forearm contact at home and its time-of-flight stop;
     - the time-of-flight stops on 19384 and 19444;
     - the wire's line-of-sight block on 12142.
   - **Recorded passes and refusals.** The four recorded runs that reached
     closure (14944, 15004, 14884 and Stage A 8235) are event-free. The four
     refused layouts (10001, 10061, 23167 and 19145) overlap orchard geometry
     at the settled home.
   - **The re-orientation result.** Among the seven diagnosed targets, it
     finds clear re-oriented paths for exactly 530 (55.5°) and 19444 (83.6°).
   - **The committed predictions.** It reproduces the 40 per-target calls in
     the tree1 swept-path predictions. The exceptions are 35837, 35957 and
     36017: each is marginal at home, within 0.1 mm of a trellis wire, and
     may take either class.

   A rebuild that misses any item selects nothing.
2. **Search for candidates.** Search the jaw-fit-screened pool (444 spurs: 185
   on tree0 and 259 on tree1), excluding 530 and 19444. The selection rule and
   its projection tool are committed with the rebuilt model, before the search
   runs. A target qualifies only if all three conditions hold:
   - **(a) A clear path.** The model finds a clear re-oriented path to a
     gate-clear final pose.
   - **(b) The approach keeps the patch.** The open jaw leaves at least 140 of
     the 169 patch elements at every predicted pose of the approach. This is
     the check that 19444 fails and 530 passes.
   - **(c) The closure loses the patch.** At the planned final pose, with the
     jaw at closure progress 1/6 (the first frame rendered closing), the jaw
     leaves fewer than 140. That is the `jaw_mask_occluded` loss that starts
     the hold.
   - **Go/no-go.** With no qualifying target there is no GPU request, and the
     finding is that 530's case is rare. With one, the experiment tests one
     unseen target, and its result says so.
3. **Register the experiment.** It runs `planned_pose_jaw_hold` on the
   selected targets, with controls. It uses P1 as amended on October 2, with
   the 0.01 px outline band. Draft predictions, carried over from P4, P5, P6
   and P8 of the jaw-in-view protocol:
   - the first closure loss is mask-attributable and comes at the frame step 2
     predicts;
   - the hold covers every frame from that loss to the deadline, and detach
     comes on time;
   - at least one run per target passes all 17 checks;
   - the piece and the true mouth-to-spur distance stay still.

   It also carries:
   - **A P0 replay gate**, which checks the code only. With every flag off, the
     code reproduces the 141 earlier recordings through each recorded stop.
     With the mask and the hold on, it reproduces the 6 live jaw-in-view
     recordings, holds included.
   - **A scorer**, reviewed blind and committed before submission. Until it is
     committed, no Slurm state or elapsed time is read, because they reveal
     outcomes.
   - **The label** on every result: *known-map plan; jaw self-mask; closure
     hold with freshness and frame-reuse checks waived on held frames*.

## Cost

- **CPU.** The rebuild and its acceptance test are several hours of work,
  mostly agent time, plus share CPU. The search is about 1–2 CPU-hours.
- **GPU.** The figure depends on how many candidates qualify, and is stated
  when the experiment is registered. As a guide: 4 targets × 3 repeats + 2
  controls = 14 runs × 45 minutes = 630 GPU-minutes reserved.
- **Storage.** About 0.47 GB of captures per run, so about 6.6 GB for 14 runs.
  The share's project quota held 1.504 TiB on October 2: above the 1.5 TiB
  soft quota, below the 2 TiB hard limit, with 3 weeks 4 days of grace left.

## Risks

- **The rebuild may fail its acceptance test.** Then the tool selects nothing.
- **There may be no qualifying target.** The orchard may hold no unseen target
  that meets (a)–(c). Then the hold cannot be tested on another target here.
- **A selected target may stop before closure**, on contact or on a
  perception loss. The hold is then untested in that run. Condition (b)
  screens the open-jaw loss that stopped 19444, but not every other cause.
- **Tracking through the re-orientation is shown on two targets only.** The
  tracker held through 530's 55.5° in 6 of 6 runs and 19444's 83.6° in 5 of 5.
  A new target's view through its own re-orientation is untested.
