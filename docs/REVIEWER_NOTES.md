# Reviewer audit and remaining work

September 23 update. What changed, and what a reviewer should check first:

- **Success rate:** a pre-registered 20-target sweep under source and morning
  light passed **0 of 40** (Wilson 95% 0–0.088). Every recorded run stopped at a
  gate. Half the trials never ran because the tree1 targets were outside the
  renderer's accepted candidates; that error is disclosed in the
  [protocol](EVAL_PROTOCOL_2026-09-23.md#an-error-in-the-target-register), kept in
  the denominator, and now caught on CPU before submission. Only tree0 was
  measured. [Evidence](evidence/eval_2026-09-23.json).
- **Lighting pilot:** CLAHE dropped on its pre-registered criteria; morning raw
  sat on the tracker's own 4-feature floor. [Evidence](evidence/lighting_pilot_results_2026-09-23.json).
- **ROS 2 SIL, not HIL:** 199/199 recorded decision states reproduced on
  `21328323`; negative controls hold 39/39. [Notes](ROS2_SIL.md).
- **C++:** ToF deprojection port with gtest parity against the Python original.
- **CuRobo:** still not executed; blocked on the pinned stack.
  [Probe](evidence/curobo_feasibility_2026-09-23.json).
- Tracked CPU suite: 567 passed, 9 skipped, 1 deselected; ruff clean.
- Unresolved and not published: the older untracked `studio/src` scene viewer,
  whose asset exporter copies mock-pruner CAD and whose untracked test fails
  because the glTF rebase collapses tree1 onto the origin.

September 20 reconciliation: six raw/CLAHE pilot sequences and two earlier
full daylight runs pass fresh 17/17 grading, all on `tree0_SPUR_component_8235`.
The [new audit](RESEARCH_AUDIT_2026-09-20.md) records 100 Envy + 100 UFO local
assets, unresolved checkpoint exposure and the unexecuted learned-depth study.
Tracked CPU scope passes 514 tests (9 skips, 1 deselected); the full working
tree has 545 passes and one failure in an existing untracked studio test.
Historical milestone/test counts below describe earlier checkpoints.

Latest milestone: two-tree job `21328323` passes all 17 independent sequence
checks: 68 applied vision commands, one surrogate release, 809.49 mm measured
piece drop and <0.001 mm final home error. Both original trees remain in the
scene, but only one known spur is selected. Earlier tracking and closure
failures remain visible beside the successful recording.

Local CPU suite: **489 passed, one simulator-only test deselected**. Ruff passes
on the pruning source/tests/tools and renderer (110 files formatted). Running
Ruff across the entire inherited repository still reports 146 lint errors and
14 files needing formatting; the maintained CI scope is narrower.

September 19 recheck: the two short daylight captures exist and pass capture
validation; release/return are absent. A local uncommitted browser prototype
exists and is preserved separately. The [new experiment protocol](RESEARCH_EXPERIMENTS_2026-09-19.md)
records fresh CPU checks, research sources and a six-run lighting comparison.
Historical test counts above refer to their original checkpoint.

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
- **No physical cut is demonstrated.** The full sequence now releases an existing
  rigid piece and measures its fall. This is a visual jaw surrogate plus a
  discrete PhysX state change, not actuated blade CAD or wood fracture. The
  fixed virtual camera is not physically calibrated. ToF noise is disabled;
  the stop gate is diagnostic, not a hardware safety controller. Post-release
  tracking loss is allowed during home-directed return.
- **Broader evaluation is missing.** The completed source/morning/evening pilot
  still exercises only one known original tree0 target with RTX depth.
  It does not establish multi-tree pruning, learned-depth transfer, calibrated
  daylight robustness or a population task-success rate. Earlier failures remain.
- **Browser prerequisites are missing.** There is no cleared browser asset
  bundle, validated MJCF parity model, trained actor or browser performance
  measurement established by this audit. Local uncommitted browser work is
  preserved; this checkpoint evaluates the simulation experiment path.
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

> UR5e pruning experiments in Isaac Sim: two Blender trees, live RGB-D tracking, dual-ToF sensing, and recorded surrogate-release success and failures.

Topics:

`robotics`, `agricultural-robotics`, `isaac-sim`, `isaac-lab`, `ur5e`,
`blender`, `time-of-flight`, `computer-vision`, `visual-servoing`, `optical-flow`,
`rgb-d`, `sensor-fusion`, `simulation`, `python`, `pytorch`, `slurm`

## README decisions

The root README is a recruiter-scannable case study with technical proof under
it. The GIF is immediately after the one-line description. Problem, Solution,
and Result sit above Quickstart so a hiring manager can read the claim in under
a minute. Architecture, the evidence table, and remaining-work caveats stay in
the same file. The four-command quickstart still runs the CPU demo, not Isaac.
No section claims learned perception, physical cutting, or a measured
task-success rate.

## Authorized research execution (2026-09-20)

Implementation and jobs are now in progress; the preceding audit-only snapshot is historical. Frozen DA2 offline evaluation job **21370005** submitted, no dependencies. The one-frame CPU accuracy gate failed; learned control remains conditional. Checkpoint-specific leakage corrections, current job/result states and storage are tracked in the [execution record](RESEARCH_EXECUTION_2026-09-20.md) and `docs/evidence/research_execution_2026-09-20.json`.
