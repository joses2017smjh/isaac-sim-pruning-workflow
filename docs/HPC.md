# HPC bring-up

Current result: job `21247873` completed on A40 `cn-s-2` in 5m40s with
60 frames of online RGB-D approach in the original textured Blender orchard.
It applied 60 vision commands and moved 230.08 mm; closure, release and retreat
were not reached in the six-second capture. All ten recording checks passed.
The [actual GIF and evidence](ISAAC_RENDER.md) distinguish this approach result
from the full sequence. Retry `21300015` was cancelled after 10m53s on `cn-gpu7`
following tracking loss at frame index 54 (5.5 seconds): three round-trip
inliers remained, below the unchanged minimum four. The controller latched
`vision_invalid`; no closure or detachment occurred. Its 71 partial frames and
incomplete report remain preserved. CPU diagnosis is underway; no further GPU
retry is submitted at this checkpoint. Earlier `21298152` was cancelled for
capture throughput; `21300015` skipped unused intermediate RTX renders without
changing physics steps or manual camera quality. Cut/drop/home-return validation
remains pending. See the [current job ledger](../SLURM_JOBS.md).

Earlier job `21208215` passed the short environment smoke and recorded
140 RTX frames on `cn-gpu6` on 2026-09-07. The original UR5e and mock pruner
approach a procedural tree, inspect at standoff, and retreat while both ToF
grids update. See the [video and reproduction guide](ISAAC_RENDER.md),
[smoke evidence](evidence/smoke_21208215.json), and
[render evidence](evidence/render_21208215.json). This is not a trained policy
or a cutting demonstration. Its inspection video remains available.

Inventory queried from Slurm on 2026-08-27, with the Isaac stack fact from
Jose's BHL work (job `21036831`, 2026-08-26, `cn-r-1` A40, driver 595.71.05):

- `cn-gpu5`–`cn-gpu7`: 8× RTX 8000, partitions `gpu` / `gpu-dmv`.
- `cn-gpu10`–`cn-gpu12`: 8× L40S, partition `tiamat`.
- `cn-r-*`, `cn-s-*`, `cn-t-1`: A40, partitions `ampere` / `athena`.
- `cn-w-1`: H100; `cn-w-2`: L40S.
- `dgx2-*`: V100. No RT cores. Do not run Isaac RTX here.

Account association: `eecs`, QOS `normal`, no fixed partition. Seeing a node
in `sinfo` does **not** prove this account can submit there. `eecs2` (RTX 6000)
is not valid for this account.

## Which Isaac stack this cluster can actually render

Do not pull a moving NVIDIA Isaac Sim 5.x tag. On this cluster that path is
already measured:

| Stack | Location | Versions | RTX |
|---|---|---|---|
| **v51** | `Humanoid_Lite/venv` + `container/bhl.sif` | Isaac Sim 5.1.0, Isaac Lab 2.3.2, Python 3.11 | Dead. Segfault in `omni.usd.create_hydra_engine`. Depth is Warp ray-cast only. |
| **v60** | `Humanoid_Lite/venv-isaac60` + same SIF | Isaac Sim 6.0.0.1, Isaac Lab 3.0.0b2, Python 3.12 | Works. Job `21036831`: RGB std 24.4, depth finite. |

The inherited pruning harness is Isaac Lab **2.x** (`DirectRLEnv`). The renderer
that works is Lab **3.0.0b2**. RGB/eval jobs use v60. Do not treat a 5.1 number
and a 6.0 number as the same table.

Jose's ordered path (5.1 segfault → Warp depth → v60 RTX → Lab 3 warp shims →
obs-width trap) is in [`docs/ISAAC_STACK.md`](ISAAC_STACK.md). Copy those
lessons; do not rediscover them.

Kit EULA is already accepted via `Humanoid_Lite/.home`. Launch with `bhl_exec`
from `Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh`. From an Open OnDemand
or interactive allocation, wrap `sbatch` in `slurm_clean` or the new job is
submitted as a step of the current one.

The SIF is Ubuntu 22.04 (~293 MB). Isaac lives in the **venv**, bind-mounted.
There is no separate 20 GB NVIDIA Isaac image to pin for Gate 0.

## Pinned robot and live-ToF implementation (runtime smoke passed)

The current robot description was generated from pinned BDS revision `dfede4c`
and the selected UR5e calibration. Its generation evidence records the source,
calibration, generated-URDF, selected-mesh hashes, and the reviewed
base-to-camera0/tof0/tof1/tool0 transforms:
[`urdf_generation_ur5e_mock_pruner_bdsdfede4c0_ur18e6f603_calib_3941312424972580002_urdf6b02ce9330be.json`](evidence/urdf_generation_ur5e_mock_pruner_bdsdfede4c0_ur18e6f603_calib_3941312424972580002_urdf6b02ce9330be.json).

Import job `21136450` **passed** its application gate on `cn-gpu7`: evidence has
`status: complete`, `ok: true`, `imported: true`, a hashed output inventory,
metres/Z-up, six active UR revolute joints, no slider, and provenance-verified
camera0/ToF/tool transforms. The robot config now selects that exact nested
content-addressed root and records `reimport_required: false`. See the
[`job evidence`](evidence/urdf_import_21136450.json) and the consolidated
[`SLURM ledger`](../SLURM_JOBS.md). The corresponding
`logs/prune-urdf-import-21136450.out` is intentionally local/gitignored.

The environment now constructs and registers two v60
`MultiMeshRayCasterCamera` sensors. Each produces an 8x8
`distance_to_camera` table at the reviewed 15 Hz cadence; the implementation
tracks the rigid `mock_pruner__base` and applies the two verified site offsets
because this v60 beta overwrites an authored non-rigid site's resolved offset.
Range validity is gated to 0.03--3.4 m. CPU tests cover the configuration and
the deterministic, non-colliding smoke wall. Job `21208215` passed the runtime
check: both grids returned 64/64 finite wall ranges and changed after a commanded
5 mm optical-axis motion. Flow and metric-student policy buffers remain explicit
placeholders. Farneback flow and brown-pixel overlays are offline diagnostics.
The separate Blender demonstration now supplies online seeded LK tracking and
RTX-depth backprojection to the motion controller; it is not a trained policy.

## Batched env smoke (passed on the bench-mounted fixture)

Do not split trainer import, env construct, and obs-width into three jobs.

```bash
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
export PRUNING_ROOT=/nfs/hpc/share/$USER/isaac-sim-pruning-workflow
export PRUNING_ASSET_ID=ur5e_mock_pruner_bdsdfede4c0_ur18e6f603_calib_3941312424972580002_urdf6b02ce9330be
export PRUNING_USD="$PRUNING_ROOT/artifacts/usd/$PRUNING_ASSET_ID/${PRUNING_ASSET_ID}_abs/${PRUNING_ASSET_ID}_abs.usda"
export PRUNING_USD_EVIDENCE="$PRUNING_ROOT/docs/evidence/urdf_import_21136450.json"
slurm_clean sbatch hpc/slurm/env_smoke.sbatch
```

The job must write `docs/evidence/smoke_<jobid>.json` with `"ok": true`,
distinct A/B/C last-dims, C==D width with `not allclose` when ToF ≠ metric,
an absolute-pose hold, complete named PhysX contact coverage, both live
8x8 sensor frames and poses, a verified non-colliding smoke target, and a
nonzero range change after a controlled 5 mm motion along the optical axis.
Flow and metric-student are still placeholders, so the smoke validates the
dual-ToF path and A--D interface plumbing rather than claiming all four variants
are live. Asserts, not log greps. Record both RL imports and do not pip-install
in the job.

Job `21208215` passed with the same **5 mm** translation-drift limit. The fixture
now mounts the robot 0.70 m above the floor and raises its non-colliding test
wall by the same amount. Six 60 Hz hold steps measured zero drift at the recorded
precision; this is a short smoke, not a long-duration stability benchmark. The
subsequent 5 mm command finished with **0.360 mm** position error. Median range
changes were **3.961 mm** and **3.800 mm**, with 64 shared valid pixels per ToF.
All **20** rigid articulation bodies have named contact coverage.

The controller uses SVD damped least squares (`damping=0.05`), a 0.05 rad
per-update joint correction bound, and measured dynamics gravity compensation
`+g(q)`. The arm's 800/40 drive gains and gravity remain enabled. The robot and
reviewed sensor offsets are unchanged. [`smoke_21208215.json`](evidence/smoke_21208215.json)
records these settings, the two independent tool-pose paths, and the sensor
response. `rsl_rl` imports; `skrl` remains absent. Baseline and PPO execution
still need their own evidence; this smoke does not establish either result.

### Historical failures and their resolution

Job `21146271` **failed** the application gate at `phase: construct` on
`cn-s-1`. Its JSON has `ok: false`, no observation result, and no traceback;
the old cleanup order swallowed the caught diagnostic. Retry `21153271`
persisted the exact failure before cleanup: this manually built `DirectRLEnv`
passed an unresolved `{ENV_REGEX_NS}/Robot` token to the v60 spawn function,
which requires an absolute path. Robot, contact, ToF, tree, and smoke-target
expressions now use the supported `/World/envs/env_.*/...` form. Job `21153411`
then registered the robot, contact, and both ray-caster backends before exposing
the next v60 lifecycle mismatch: `SceneEntityCfg.resolve()` dereferenced the
articulation before `DirectRLEnv` started physics and created `_root_view`.
Entity resolution is now deferred until after `super().__init__`, matching the
v60 lifecycle, and the robot spawner explicitly enables contact-report APIs.
Job `21153625` then failed on a raw Warp Jacobian without `.clone()`.
The runtime now uses the link-origin Jacobian and explicitly converts between
Lab 3 `xyzw` and core `wxyz` poses. Jobs `21185961` and `21186027` reached live
stepping but failed the 5 mm hold-drift limit. The diagnostic retry measured
**20.12 mm** translation drift and **0.00309 rad** rotation drift. Both ToF
grids reached frame 2 with 64/64 finite returns. Its contact tensor was finite
but covered only one body, and the controlled 5 mm motion test was not reached.
Job `21201622` then instrumented every nested rigid link and measured about
**180 N** of floor contact on `mock_pruner__base`; the independent tool-pose
paths agreed. Raising the fixture removed this contact in `21208215`, which
passed the unchanged hold limit and the motion-response test. The floor-level
failure remains a failure, not a retroactive pass. See
[`smoke_21146271.json`](evidence/smoke_21146271.json),
[`smoke_21153271.json`](evidence/smoke_21153271.json),
[`smoke_21153411.json`](evidence/smoke_21153411.json),
[`smoke_21153625.json`](evidence/smoke_21153625.json),
[`smoke_21186027.json`](evidence/smoke_21186027.json), and the
[`SLURM ledger`](../SLURM_JOBS.md).

Job `21079145` reached an A40 (`cn-r-1`) and then died: Apptainer killed
`squashfuse_ll` after Kit left a background process, so
`docs/evidence/smoke_21079145.json` was never written. That is **not** a pass;
it remains historical alongside the more informative but still failed
`21146271` attempt.

## Record the actual robot workflow

The detached wrapper combines a pinned-stack preflight, the short environment
smoke, RTX capture, and an independent output validator in one allocation:

```bash
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
slurm_clean sbatch hpc/slurm/render_pruning_workflow.sbatch
```

It selects the promoted robot asset from the repository config, requests one
A40/RTX 8000/L40S, and writes a fresh
`artifacts/isaac_render/job_<jobid>/` directory. It refuses to overwrite an
existing capture. The environment lock is [`ENVIRONMENT.lock.json`](ENVIRONMENT.lock.json).
Use a detached job for capture; the earlier interactive step was cancelled
after 61 frames and did not produce a complete recording.

Job `21208215` recorded **14.0 seconds** of measured physics at 10 video fps.
The tool moved **247.37 mm**; target distance went from **342.98 mm** to
**96.07 mm**, then returned to **343.84 mm**. Final return error was
**1.653 mm**. Contact coverage was complete and measured contact forces were
zero throughout this inspection; collision response was not exercised. The
two ToF grids had **40.01% / 42.31%** valid rays across the recording, rather
than the smoke wall's 100% coverage. Noise was disabled.

The RGB cameras and metric depth are RTX outputs. The wrist mount is defined
in simulation, with a fixed target-informed toe-in; it is not calibrated
hardware. The scripted controller uses known geometry and a measured ToF stop
gate, not the offline CV overlay. The portable [CPU demo](DEMO.md) is a separate
analytic demonstration and does not render Isaac Sim. See
[ISAAC_RENDER.md](ISAAC_RENDER.md) for the media composer, artifacts, and limits.

## Gate 0: 1 m cube + 2 m plane (v60, headless RTX)

```bash
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
mkdir -p /nfs/hpc/share/$USER/isaac-sim-pruning-workflow/logs
slurm_clean sbatch hpc/slurm/isaac_headless_smoke.sbatch
```

Pass condition (all must be in the job log / `logs/isaac_smoke.json`):

1. The job lands on an RTX-capable node (`a40`, `rtx8000`, or `l40s`).
2. Vulkan/EGL / Kit start without the 5.1 hydra-engine segfault.
3. 1 m cube, camera at `y = -2`, `distance_to_image_plane` median within 50 mm of 1.5 m.
4. Fronto-parallel plane at `z = -2`, median within 50 mm of 2.0 m and 5–95 spread under 50 mm.
5. Cube RGB is not a flat frame (`std > 1`).
6. Node, driver, Isaac Sim 6.0.0.1, Isaac Lab 3.0.0b2, and venv path are recorded here.

This gate **passed** on job `21077170` (2026-08-28, `cn-r-4` A40, driver
595.71.05, 44 s). Evidence: `docs/evidence/isaac_smoke_21077170.json` and
`logs/prune-isaac-smoke-21077170.out`.

| Check | Result |
|---|---|
| Node / GPU / driver | `cn-r-4`, NVIDIA A40, 595.71.05 |
| Stack | Isaac Sim 6.0.0.1, Isaac Lab 3.0.0b2, Python 3.12 |
| Venv | `/nfs/hpc/share/sanchej7/Humanoid_Lite/venv-isaac60` |
| SIF | `Humanoid_Lite/container/bhl.sif` sha256 `d427d9c32d23134493be3fe9721a790d71f3134f6cad847725537009f9bcac29` |
| Cube `distance_to_image_plane` | **1.5000 m** (expect 1.5, finite 100%, spread ~0) |
| Plane | **2.0000 m** (expect 2.0, spread ~0) |
| Cube RGB | mean 172.8, std 38.0 (not a flat frame) |
| Trees | Envy `00000`–`00009` written as `UsdGeom.Cylinder` USDA |

Job `21076907` on the same stack started Kit and then died serializing a NumPy
`bool_`. That log is not a pass.

## Trees (CPU, no RTX)

```bash
PYTHONPATH=source/isaaclab_pruning python tools/batch_cyl_to_usd.py \
  /nfs/hpc/share/$USER/Computer_Vision/trees/metadata \
  artifacts/trees --include-held-out \
  --manifest docs/evidence/trees_converted_manifest.json
PYTHONPATH=source/isaaclab_pruning python tools/score_blender_trunk.py
PYTHONPATH=source/isaaclab_pruning python tools/score_camera_offset.py
PYTHONPATH=source/isaaclab_pruning python tools/fit_curobo_spheres.py
```

## 30 cm camera-rect and scripted baselines (after a passing env smoke)

```bash
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
slurm_clean sbatch hpc/slurm/camera_rect.sbatch
# Reuse the exact promoted PRUNING_* exports from the env-smoke block above.
slurm_clean sbatch hpc/slurm/baselines.sbatch
```

Pass files: `docs/evidence/camera_rect_<jobid>.json` and
`docs/evidence/baselines_<jobid>.json` with `"ok": true`. Do not tick those
ROADMAP rows from a script that never ran on a GPU.

## URDF import

Job `21136450` (2026-09-03, `cn-gpu7`, Quadro RTX 8000) is the promoted import.
It accepted Isaac Lab 3's nested `_abs` root layout, inventoried and hashed the
output, and passed composed-stage validation for metres, Z-up, six UR joints,
absence of a slider, and all reviewed fixed transforms. `ArticulationCfg`
supplies stiffness/damping from `ur5e_pruner.yaml`, as the importer still warns
that those gains are absent from its generated PhysX drives.

Job `21125352` remains preserved failure evidence: it rejected the converter's
nested root before stage validation because the old wrapper expected a flat
root path. Job `21077217` remains historical proof that the stale BDS snapshot
could load, not evidence for current transforms. Neither supersedes the green
[`21136450` evidence](evidence/urdf_import_21136450.json).

## Cache policy

Jobs prefer `/scratch/$USER` for Omniverse and CUDA caches (`setup_node_cache`).
If that directory is missing, caches fall back to Lustre. Keep Kit caches and
generated USD off the stak home filesystem.
