# Pending work and stretch goals

Updated November 25, 2026. Job details are in [SLURM_JOBS.md](../SLURM_JOBS.md)
and results in the [roadmap](ROADMAP.md).

## Queued

Nothing is queued.

Finished and published:
- [longer episode at 66.7 and 100 ms delay](EVAL_PROTOCOL_DELAY_EPISODE_2026-11-25.md#result--november-25-2026),
  November 25: all six 30 s runs passed 17/17 with the arm home at the
  registered frames (D0–D3 supported), so the November 5 delay failures were
  the 20 s episode running out.
- [closure-hold sweep](EVAL_PROTOCOL_HOLD_SWEEP_2026-10-06.md#result--november-22-2026),
  scored November 22: 4 of 28 targets passed 17/17; the hold carried all 6
  closures to detachment; S1, S3–S6 supported, S2 untested, S7 refuted
  (control 530 failed on tracker drift during closure).
- [closure hold on unseen targets](EVAL_PROTOCOL_HOLD_GENERALIZATION_2026-10-04.md#result--october-5-2026):
  a negative result. 0 of 9 runs reached closure, so the hold was not
  exercised; the controls passed.
- [agreement arm in closed loop](EVAL_PROTOCOL_AGREEMENT_ARM_CLOSED_LOOP_2026-10-03.md#result--october-4-2026):
  evening 14944 passed 17/17 in 4 of 4 runs; A1–A5 supported.
- [jaw in the camera's view](EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md#result--october-2-2026):
  530 passed 17/17 in 3 of 3 runs, the first pass of a contact target.
  - P2–P6 and P8 supported, P7 partly supported.
  - P1 refuted, by one pixel on the jaw's outline.
- [depth-aware appearance check in closed loop](EVAL_PROTOCOL_DEPTH_APPEARANCE_CLOSED_LOOP_2026-10-01.md#result--october-2-2026):
  C1–C6 supported. 4 of 4 low-sun 15004 runs pass.
- [depth-aware appearance held-out replay](EVAL_PROTOCOL_DEPTH_APPEARANCE_HELDOUT_2026-10-01.md#result--october-1-2026):
  H1–H4 supported.
- [render-gap check](EVAL_PROTOCOL_RENDER_GAP_2026-09-30.md#result--october-1-2026):
  G0 and G2 supported, G1 partly supported, G3 refuted.
- [jaw-shadow counterfactual](EVAL_PROTOCOL_JAW_SHADOW_2026-09-30.md#result--october-1-2026):
  5 of 5 predictions supported.
- [rendered-lighting training](EVAL_PROTOCOL_LIGHTING_TRAINING_2026-09-27.md#result--september-30-2026).
- [known-map approach](EVAL_PROTOCOL_PLANNED_APPROACH_2026-09-27.md#result--september-30-2026).

## Needs a decision (no GPU)

- **Storage.** On November 22 the share held 1.71 TiB against the 1.5 TiB soft
  quota, with 3 weeks 6 days of grace left (to about December 19). This
  repository's directory is about 139 GiB of that.
- **An independent witness for the jaw mouth (studied November 25).** The gate
  takes the target from the controller. An exploratory CPU study
  ([write-up](ROS2_SIL.md#november-25-what-could-witness-the-target-in-the-mouth-exploratory))
  found the existing ToF sensors see the target at closure but lose it when it is
  centred, a beam grid across the jaw gap follows the geometry, and one camera
  ray is fragile. Which channel to build is undecided. The same study found that
  the cut closes with the spur 7.5 to 12.8 mm ahead of the mouth centre, near the
  jaw tips; whether to change the approach so it closes on the centre is a
  separate open question.
- **Pixel-centre convention (measured November 22).** The jaw self-mask and
  the depth-aware check treat pixel index u as continuous u + 0.5, which the
  renderer's camera and every mask-fidelity check support. The point tracker
  uses OpenCV centres and back-projects with no offset, so its 3D target sits on
  a ray half a pixel away: 0.26 to 0.59 mm per axis over 95% of 17,261 tracking
  frames (0.61 mm at most), against the 8 mm mouth tolerance
  ([evidence](evidence/pixel_convention_2026-11-22.json)). Moving the tracker to
  u + 0.5 would change every baseline replay, so it needs a labelled,
  registered arm, not a silent fix.
- **Repository housekeeping.** Decide what to do with the uncommitted
  `studio/src` prototype and add repository topics.

## Approved follow-ups (October 2): CPU scoping first, GPU budget asked later

- **P1 with a 0.01 px outline band**, for future runs. Registered in the
  [jaw-in-view protocol](EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md#registered-for-future-runs--october-2-2026).
  The October 2 verdict stays refuted.
- **Perception-side target selection**: tried on the CPU, October 5–6. Only
  the home visibility check is reliable, and it is registered as S1 of the
  closure-hold sweep. The tracking-prediction study is unfinished (partial
  work in `pruning_work/perception_select_20261005/tracking/`).
- **Evening 14944: the registered agreement arm in closed loop**
  ([scope](SCOPE_AGREEMENT_ARM_CLOSED_LOOP_2026-10-02.md)): done October 4,
  4 of 4 passes (see Finished).

## Stretch goals

- **Approach planning from sensors.** The known-map experiment used the
  scene map, so it is an upper bound. The next step is a planner that sees the
  arm-side obstacles from its own sensors.
- **Appearance check robust to cast shadows.** Done as a labelled variant
  (D_strict + J, October 2) on simulator depth. Next, pair it with real jaw
  geometry and a real depth sensor's noise in place of the visual surrogate
  and ground-truth depth.
- **Repeatable renders.** The RGB render differs run to run, while depth and
  pose do not. Seeding the path tracer would make single-run comparisons
  meaningful; until then, a class change near a tracker floor counts only if
  it repeats.
- **Close the Isaac depth gap.** The rendered-lighting model fixes Blender
  evening but leaves the Isaac wrist camera above 0.5 m of target error.
- **Envy and UFO trees in Isaac.** The vision controller on the learned-depth
  families, and held-out Envy `00042` / `00065` and UFO rollouts. There is no
  spawn or target-selection path for these trees yet.
- **Motion planning.** Execute a CuRobo plan to the pre-cut standoff; this is
  blocked on a CUDA-extension build outside the pinned stack.
- **Policy learning.** Close the live observation feeds, validate the PPO
  trainer, then train variants A–D × 5 seeds.
- **Physical realism.** Blade actuation and wood severing, a calibrated camera
  model, the 30 cm box compared between Isaac and Blender, and PyBullet sim2sim
  numbers.
- **Hardware.** A ROS 2 hardware-in-the-loop demo, pending a physical rig.
