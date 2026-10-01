# Pending work and stretch goals

Updated September 30, 2026. Job details are in [SLURM_JOBS.md](../SLURM_JOBS.md)
and results in the [roadmap](ROADMAP.md).

## Queued

- **Jaw-shadow counterfactual** (approved, 540 GPU-min): 8 chained batches,
  `21491132` → … → `21491149`. J1–J5 in its
  [protocol](EVAL_PROTOCOL_JAW_SHADOW_2026-09-30.md).

The rendered-lighting training and the known-map approach finished on
September 30 and are published:
[lighting result](EVAL_PROTOCOL_LIGHTING_TRAINING_2026-09-27.md#result--september-30-2026) ·
[approach result](EVAL_PROTOCOL_PLANNED_APPROACH_2026-09-27.md#result--september-30-2026).

## In preparation (approved September 30)

- **Jaw in the camera's view** (270 GPU-min): offline replay of a jaw
  self-mask on the recorded 530 and 19444 frames decides the design, then a
  protocol, then submission.
- **Render-gap check**: re-render the matrix trees with the training
  renderer's settings and score all five depth models on them. The exact
  reservation is stated in its protocol before submission.
- **Depth-aware appearance check** (the user's suggestion): an offline replay
  of every recorded run with an appearance decision that also uses depth.
  CPU only.

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
