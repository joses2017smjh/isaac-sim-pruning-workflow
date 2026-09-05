# Reviewer audit and remaining work

Reviewed source, tests, configuration, package metadata, CI, and tracked
evidence on 2026-09-05. The repository contains a runnable CPU pruning demo and
an Isaac research integration; it does not contain a completed learned policy.

## What a reviewer will notice

- **No root LICENSE.** This is intentional: inherited code has unresolved
  repository-level rights and inconsistent license metadata. [NOTICE](../NOTICE.md)
  records the issue. Author permission is needed before choosing a blanket license.
- **GPU setup is cluster-specific.** The BHL container, v60 environment,
  robot meshes, and generated USD assets are external. The 200-tree manifest
  records a conversion; the clone does not contain those 200 assets.
- **The Isaac task is incomplete.** It still uses a fixed smoke target.
  Live tree-oracle selection, reset-time target assignment, and nearby-wood
  tensors are not integrated. The CPU demo executes these geometry checks on
  its procedural scene.
- **Live motion/contact validation remains open.** Job `21186027` measured
  20.12 mm tool-hold drift against a 5 mm limit. It read two complete ToF grids,
  but the contact tensor covered one body; full-arm coverage is unverified.
- **Perception feeds are incomplete.** Flow is zero and metric depth is a
  constant in the Isaac environment. Wrist RGB is disabled; the physical
  camera model and optical calibration are unknown. The demo's metric noise
  model is explicitly synthetic.
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

The demo now has a CLI, GIF, poster, offline sensor replay, and measured JSON.
CPU CI installs its missing Pillow/Matplotlib extras and runs the demo. Ruff is
pinned to the pre-commit version. The unused-variable lint failure and formatter
drift are fixed. Broken hooks that referenced absent license templates were
removed without changing inherited notices. The missing bark texture is
documented as optional rather than described as packed.

Functional fixes cover lateral servo direction, rotation about the approach
axis, invalid-depth fusion, v60 articulation initialization, contact reporting,
link-origin Jacobians, and explicit Lab `xyzw`/core `wxyz` pose boundaries.
Live simulator acceptance still depends on [job evidence](../SLURM_JOBS.md).

## GitHub sidebar

Description:

> Robotic pruning simulation with dual-ToF feedback, scripted tool control, collision checks, and a reproducible CPU demo; Isaac Sim integration in progress.

Topics:

`robotics`, `agricultural-robotics`, `isaac-sim`, `isaac-lab`, `ur5e`,
`time-of-flight`, `sensor-fusion`, `simulation`, `python`, `pytorch`

## README decisions

All six requested sections carry information for this repository. The demo
appears immediately after the one-line description, without an extra heading.
The old image inventory, stack debugging narrative, and scheduler commands
moved behind links so the first screen shows motion and how to reproduce it.
Results separate CPU scenarios, real cluster gates, and known failures.
