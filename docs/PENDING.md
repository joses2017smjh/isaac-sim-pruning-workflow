# Pending work and stretch goals

Updated October 1, 2026. Job details are in [SLURM_JOBS.md](../SLURM_JOBS.md)
and results in the [roadmap](ROADMAP.md).

## Queued

- **Render-gap check** (approved, 180 GPU-min): render array `21499598` →
  evaluation `21499599`, G0–G3 in its
  [protocol](EVAL_PROTOCOL_RENDER_GAP_2026-09-30.md). Held until the October 1
  maintenance window ends.
- **Jaw in the camera's view** (approved with the gate waiver, 270 GPU-min):
  [protocol](EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md) registered. It is
  submitted once its pre-submission replay check (P0) passes on the committed
  code.

Finished and published: the
[jaw-shadow counterfactual](EVAL_PROTOCOL_JAW_SHADOW_2026-09-30.md#result--october-1-2026)
(5 of 5 predictions supported), the
[rendered-lighting training](EVAL_PROTOCOL_LIGHTING_TRAINING_2026-09-27.md#result--september-30-2026)
and the
[known-map approach](EVAL_PROTOCOL_PLANNED_APPROACH_2026-09-27.md#result--september-30-2026).

## Next (CPU, no approval needed)

- **Depth-aware appearance check, held-out replay** (the user's suggestion; the
  user chose this path): replay the strict variant plus the jaw guard on the 12
  counterfactual recordings. Arm A has new appearance events, and arm B is a
  negative control where it must not act. A closed-loop GPU test is proposed
  only if this holds.

## Needs a decision (no GPU)

- **Storage.** The share is over its 1.5 TiB soft quota and the grace clock is
  running (about four weeks left).
- **Repository housekeeping.** Decide what to do with the uncommitted
  `studio/src` prototype and add repository topics.

## Stretch goals

- **Approach planning from sensors.** The known-map experiment used the
  scene map, so it is an upper bound. The next step is a planner that sees the
  arm-side obstacles from its own sensors.
- **Appearance check robust to cast shadows.** A labelled tracker variant, only
  after the counterfactual confirms the cause. Pair it with real jaw geometry
  in place of the visual surrogate.
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
