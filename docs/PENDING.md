# Pending work and stretch goals

Updated September 30, 2026. Job details are in [SLURM_JOBS.md](../SLURM_JOBS.md)
and results in the [roadmap](ROADMAP.md).

## Queued

Nothing is queued. The rendered-lighting training and the known-map approach
finished on September 30 and are published:
[lighting result](EVAL_PROTOCOL_LIGHTING_TRAINING_2026-09-27.md#result--september-30-2026) ·
[approach result](EVAL_PROTOCOL_PLANNED_APPROACH_2026-09-27.md#result--september-30-2026).

## Needs a decision (GPU)

- **Jaw in the camera's view.** Both remaining approach stops are the robot's
  own jaw: during closure the tracker follows it (530), and on the final leg
  it covers the tracked patch (19444). The candidates are a jaw self-mask, an
  ego-motion consistency filter, the mirrored camera side, and holding the
  aligned target through closure. The last one changes what the cut gate sees,
  so it needs registration. About 270 GPU-min for one variant on both targets.
- **Jaw-shadow counterfactual.** Rerun the low-sun stops with the jaw casting
  no shadow: 12 runs, 540 GPU-min reserved. It tests the
  [low-sun diagnosis](EVAL_PROTOCOL_PERCEPTION_2026-09-26.md#corrections-and-diagnosis-of-the-light-dependent-stops--september-28-2026-post-hoc).
- **Render gap vs lighting.** Score the rendered-lighting arms on
  companion-resolution renders of the matrix trees. This separates robustness
  to the evaluation renderer from robustness to light.

## Needs a decision (no GPU)

- **Storage.** The share is over its 1.5 TiB soft quota and the grace clock is
  running (about four weeks left).
- **Repository housekeeping.** Decide what to do with the uncommitted
  `studio/src` prototype and add repository topics.

## Stretch goals

- **Approach planning from sensors.** The queued experiment uses the known
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
