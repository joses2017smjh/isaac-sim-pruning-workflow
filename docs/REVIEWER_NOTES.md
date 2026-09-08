# Reviewer audit and remaining work

Reviewed source, tests, configuration, package metadata, CI, and tracked
evidence through the September 7 Isaac capture. The repository now contains
a [14-second Isaac robot inspection](ISAAC_RENDER.md) and a runnable CPU
pruning demo. It does not contain a completed autonomous pruning system or
learned policy.

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
  its procedural scene. The Isaac inspection renderer uses a separate procedural
  finite-cylinder USD fixture and known target; it does not render a Blender
  PLY tree or evaluate the converted Envy/UFO orchard assets.
- **Short motion/ToF smoke passes; broader validation is missing.** Job
  `21208215` passes the unchanged 5 mm hold gate on a raised fixture, records
  0.360 mm final error after a 5 mm command, and instruments all 20 rigid bodies.
  The hold covers six steps, not a long-duration test. The inspection records
  no contact; that is not proof of collision avoidance. The earlier 20.12 mm
  hold failure and approximately 180 N floor contact remain in the evidence.
- **Live policy perception feeds are incomplete.** Flow is zero and metric
  depth is constant in the training environment. The recording has real RTX
  wrist RGB and simulator depth, but its Farneback flow and brown-pixel
  segmentation are computed offline and do not control the robot. The physical
  camera model and optical calibration remain unknown; the rendered camera's
  mount is simulation-defined. CPU-demo metric estimates are synthetic.
- **No physical cut is demonstrated.** The original six-joint UR5e and mock
  pruner approach, inspect and retreat. The video does not actuate a blade,
  sever wood, or validate cutting/contact mechanics. Its ToF stop gate is a
  diagnostic guard, not a hardware safety controller; noise is disabled.
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

The published MP4 contains the complete 14-second recording. Its GIF samples
the whole timeline to stay below 1.9 MB. A labelled 3×3 median filter affects
RGB display only; CV and measurements use raw inputs. Raw camera arrays and
large capture directories stay outside Git. The recording guide explains
exactly which assets and cluster installation are required; the CPU quickstart
does not claim to reproduce Isaac on a clean CPU-only machine.

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

> UR5e branch inspection in Isaac Sim with live dual-ToF sensing, recorded wrist vision, offline CV overlays, and a reproducible CPU pruning demo.

Topics:

`robotics`, `agricultural-robotics`, `isaac-sim`, `isaac-lab`, `ur5e`,
`time-of-flight`, `computer-vision`, `optical-flow`, `sensor-fusion`,
`simulation`, `python`, `pytorch`

## README decisions

All six requested sections carry information for this repository. The demo
appears immediately after the one-line description, without an extra heading.
The first demo is now the actual Isaac robot recording; the four-command
quickstart explicitly runs the separate CPU demo. The old image inventory,
stack debugging narrative, and scheduler commands sit behind links. Results
separate the one inspection episode, short control smoke, CPU scenarios and
known failures. No section claims learned perception, cutting, or a task
success rate from those recordings.
