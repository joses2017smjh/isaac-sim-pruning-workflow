# Pending work and stretch goals

Updated September 29, 2026. Job details are in [SLURM_JOBS.md](../SLURM_JOBS.md)
and results in the [roadmap](ROADMAP.md).

## Queued (no action needed)

The `ampere` A40 nodes are drained until October 1, 16:00, so start times
depend on the `gpu` and `dgxh` nodes.

| Work | Jobs | Registered predictions |
|---|---|---|
| Rendered-lighting training render: the last 6 of 74 trees, then 5 trees of 90 frames | `21442470` (up to 4 at once) → `21442471` | L4 ([protocol](EVAL_PROTOCOL_LIGHTING_TRAINING_2026-09-27.md)) |
| Fine-tunes on the rendered frames, jitter and control arms (H100 or A40) | `21442472` → `21442473`; `21442474` → `21442475` | R1–R7 (same protocol) |
| Known-map re-oriented approach for 530 and 19444, with controls 14944 and 15004, 3 repeats (RTX 8000 or A40) | `21464467` → `21464468` → `21464469` | Q1–Q3 ([protocol](EVAL_PROTOCOL_PLANNED_APPROACH_2026-09-27.md)) |

The timed-out tree, `lpy_envy_00014`, has been resumed and is complete: 60 of
60 frames pass the geometry check.

## Next, once results land (CPU only)

- Grade the approach runs, score Q1–Q3, and label each run by GPU model.
  Compose a GIF for every recorded run. Publish.
- Score both fine-tune arms against R1–R7 and the render against L4. Publish.

## Needs a decision

- **Jaw-shadow counterfactual.** Rerun evening 14944 and 15004 and morning
  15004, with and without a shadow from the visual jaw surrogate. 12 runs, 540
  GPU-minutes reserved. It needs a recorded no-shadow scene option and its own
  protocol, and would test the
  [low-sun diagnosis](EVAL_PROTOCOL_PERCEPTION_2026-09-26.md#corrections-and-diagnosis-of-the-light-dependent-stops--september-28-2026-post-hoc).
- **Storage.** The share is over its 1.5 TiB soft quota and the grace clock is
  running (about four weeks left). Free space, or accept the risk.
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
