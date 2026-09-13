# Reviewer audit and remaining work

Reviewed source, tests, configuration, package metadata, CI, and tracked
evidence through the September 13 review. The repository now contains a
[six-second Blender-orchard visual approach](ISAAC_RENDER.md), the earlier
14-second Isaac robot inspection, and a runnable CPU pruning demo. Job
`21247873` applied 60 live RGB-D vision commands and moved the tool 230.08 mm.
It ended before closure: `vision_approach_incomplete`, not a completed pruning
task. There is no learned policy or completed autonomous pruning system.

## What a reviewer will notice

- **No root LICENSE.** This is intentional: inherited code has unresolved
  repository-level rights and inconsistent license metadata. [NOTICE](../NOTICE.md)
  records the issue. Author permission is needed before choosing a blanket license.
- **GPU setup is cluster-specific.** The BHL container, v60 environment,
  robot meshes, and generated USD assets are external. The 200-tree manifest
  records a conversion; the clone does not contain those 200 assets.
- **The training task is incomplete.** It still uses a fixed smoke target.
  Live tree-oracle selection, reset-time target assignment, and nearby-wood
  tensors are not integrated. The CPU demo executes these geometry checks on
  its procedural scene. The original Isaac inspection used a procedural
  finite-cylinder fixture; the newer renderer imports the original orchard
  `.blend` meshes, UVs, bark/soil maps, posts, and wires through USD. Neither
  path loads PLY files or evaluates held-out Envy/UFO orchard assets. The
  Blender demo selects one known spur from mesh metadata, not a learned detector.
- **Short motion/ToF smoke passes; broader validation is missing.** Job
  `21208215` passes the unchanged 5 mm hold gate on a raised fixture, records
  0.360 mm final error after a 5 mm command, and instruments all 20 rigid bodies.
  The hold covers six steps, not a long-duration test. The inspection records
  no contact; that is not proof of collision avoidance. The earlier 20.12 mm
  hold failure and approximately 180 N floor contact remain in the evidence.
- **Live policy perception feeds are incomplete.** Flow is zero and metric
  depth is constant in the training environment. The recording has real RTX
  wrist RGB and simulator depth. In job `21247873`, seeded pyramidal Lucas–Kanade
  tracking and live RTX optical-Z depth drive bounded approach commands.
  Branch identity, axis, and radius still come from mesh metadata; the depth is
  simulator ground truth, not a trained estimator. Farneback flow remains a
  separately labelled offline diagnostic. The physical camera model and optical
  calibration remain unknown; the rendered mount is simulation-defined.
  CPU-demo metric estimates are synthetic.
- **No physical cut is demonstrated.** The original six-joint UR5e, reviewed
  mock pruner, and dual-ToF offsets remain unchanged. The Blender integration
  adds a visual jaw surrogate and gated rigid-piece detachment, not actuated
  blade CAD or wood-fracture mechanics. Job `21247873` recorded no closure or
  detachment; its final tracked-mouth distance was 35.01 mm. Full-duration job
  `21298152` was cancelled after 6m05s for capture throughput, with partial
  files preserved. Retry `21300015` lost tracking at 5.5 seconds when only three
  features passed the round-trip check, below the unchanged minimum four.
  It latched `vision_invalid`, with no closure or release, and was cancelled
  after 10m53s. Its 71-frame partial recording and incomplete report are
  preserved. CPU diagnosis is underway; no further GPU retry is submitted at
  this checkpoint. The full sequence remains unvalidated. The
  ToF stop gate is diagnostic, not a hardware safety controller; noise is disabled.
- **CuRobo and training are scaffolds.** The baseline runner reports CuRobo
  readiness without executing a plan. `tools/train.py` always exits. There are
  no PPO checkpoints, held-out rollouts, sim2sim measurements, or hardware demo.
- **Legacy code remains.** `source/isaaclab_sensor_learning` preserves the
  inherited harness, including older PhysX and sensor paths. The pruning CPU
  CI covers `source/isaaclab_pruning`, `tests`, and `tools`; it is not a claim
  that every inherited script runs on v60.
- **No exhaustive history-secret audit.** A bounded tracked-file scan found
  no obvious common tokens or private keys. This is not proof that Git history
  is secret-free.

## Reproduction gaps fixed in this update

The Isaac path now has a pinned stack/container lock, source and asset
fingerprints, bounded camera warmup, detached Slurm capture, and an independent
postflight that checks frame counts, real RGB variation and finite depth.
Job `21208215` recorded all 140 frames and completed approach/retreat; the
earlier `21201622` also recorded 140 frames but failed the task. Both outcomes
are preserved rather than equating scheduler completion with success.

The older published MP4 contains the complete 14-second recording. Its GIF samples
the whole timeline to stay below 1.9 MB. A labelled 3×3 median filter affects
RGB display only; CV and measurements use raw inputs. Raw camera arrays and
large capture directories stay outside Git. The recording guide explains
exactly which assets and cluster installation are required; the CPU quickstart
does not claim to reproduce Isaac on a clean CPU-only machine.

The newer [Blender approach](demo/isaac_blender_live_approach.gif) contains all
60 recorded frames in its six-second MP4. Path tracing and OptiX denoising run
inside RTX; no display-only median filter was applied to this capture. All ten
[recording checks](evidence/render_21247873.json) passed, including changing
ToF streams and frozen physics during image accumulation. Recording completeness
does not imply successful cutting: the cut gate never authorized closure,
closure stayed at zero, and no piece dropped or retreat occurred. The earlier contact and
occluded-tracker failures remain in the job ledger.

The portable demo has a CLI, GIF, poster, offline sensor replay, and measured
JSON for all three episodes. At the September 5 published checkpoint, the
clone passed 130 tests, three robot-USD tests skipped because generated assets
were external, and one simulator test was deselected. With local USD assets,
that checkpoint passed 133 CPU tests. These are historical numbers; use the
[README](../README.md#results) and current CI run for the current suite.
[GitHub Actions](https://github.com/joses2017smjh/isaac-sim-pruning-workflow/actions/runs/33980080780)
independently passed that checkpoint's lint, formatting, tests and demo
generation. CPU CI now installs `dev`, `demo` and `render` extras; it tests
capture/compositor helpers but does not run Isaac on a GitHub-hosted runner.
Ruff is pinned to the pre-commit version. Broken hooks that referenced absent
license templates were removed without changing inherited notices. The missing
bark texture is documented as optional rather than described as packed.

At commit `a149a42`, GitHub CI passed with **356 tests passed, eight skipped,
and one deselected**; the local environment with its external assets passed
**364 tests, with one deselected**. These are checkpoint-specific counts, not
GPU task-success rates.

Functional fixes cover lateral servo direction, rotation about the approach
axis, invalid-depth fusion, v60 articulation initialization, contact reporting,
link-origin Jacobians, and explicit Lab `xyzw`/core `wxyz` pose boundaries.
The contact fix traverses the imported URDF's nested rigid bodies instead of
silently monitoring only one body. Bounded SVD damped IK and measured gravity
compensation retain the 800/40 arm gains and gravity. The passing fixture moves
the robot base and dedicated smoke wall up 0.70 m; it does not weaken the 5 mm
gate or relabel the failed floor-level fixture. Live simulator acceptance is
scoped to the [recorded job evidence](../SLURM_JOBS.md).

## GitHub sidebar

Description:

> UR5e visual approach in a Blender orchard, simulated in Isaac with live RGB-D tracking, dual-ToF sensing, and a reproducible CPU pruning demo.

Topics:

`robotics`, `agricultural-robotics`, `isaac-sim`, `isaac-lab`, `ur5e`,
`blender`, `time-of-flight`, `computer-vision`, `visual-servoing`, `optical-flow`,
`rgb-d`, `sensor-fusion`, `simulation`, `python`, `pytorch`

## README decisions

All six requested sections carry information for this repository. The demo
appears immediately after the one-line description, without an extra heading.
The first demo is now the actual Isaac robot recording; the four-command
quickstart explicitly runs the separate CPU demo. The old image inventory,
stack debugging narrative, and scheduler commands sit behind links. Results
separate the incomplete live-vision approach, completed inspection episode,
short control smoke, CPU scenarios, and known failures. No section claims
learned perception, validated cutting, or a task-success rate from those recordings.
