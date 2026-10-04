# Pending work and stretch goals

Updated October 4, 2026. Job details are in [SLURM_JOBS.md](../SLURM_JOBS.md)
and results in the [roadmap](ROADMAP.md).

## Queued

Nothing is queued or running.

Finished and published:
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

- **Storage.** The share is over its 1.5 TiB soft quota and the grace clock is
  running (about four weeks left).
- **Repository housekeeping.** Decide what to do with the uncommitted
  `studio/src` prototype and add repository topics.

## Approved follow-ups (October 2): CPU scoping first, GPU budget asked later

- **P1 with a 0.01 px outline band**, for future runs. Registered in the
  [jaw-in-view protocol](EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md#registered-for-future-runs--october-2-2026).
  The October 2 verdict stays refuted.
- **Generalize the closure hold**
  ([scope](SCOPE_CLOSURE_HOLD_GENERALIZATION_2026-10-02.md#steps-1-and-2--october-4-2026)).
  - The rebuilt planner passed its acceptance test, and 34 targets qualify.
  - 31 of the 34 also have a clear plan that keeps the jaw out of view, so
    there the occlusion comes from the chosen pose.
  - 3 targets (tree0 3721, tree1 18143, tree1 36196) have no checked
    jaw-free plan.
  - Needs the user's choice of target set and a GPU budget.
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
