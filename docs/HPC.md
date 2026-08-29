# HPC bring-up

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

## Batched env smoke (one RTX slot, after Gate 0)

Do not split trainer import, env construct, and obs-width into three jobs.

```bash
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
slurm_clean sbatch hpc/slurm/env_smoke.sbatch
```

The job must write `docs/evidence/smoke_<jobid>.json` with `"ok": true`,
distinct A/B/C last-dims, C==D width with `not allclose` when ToF ≠ metric,
one `env.step(0)`, and a finite PhysX contact tensor. Asserts, not log greps.
v60 may have `rsl_rl` and not `skrl`; record both and do not pip-install in the job.

Job `21079145` reached an A40 (`cn-r-1`) and then died: Apptainer killed
`squashfuse_ll` after Kit left a background process, so
`docs/evidence/smoke_21079145.json` was never written. That is **not** a pass.
The inner script now flushes JSON at each phase; requeue the same sbatch.

## Gate 0: 1 m cube + 2 m plane (v60, headless RTX)

```bash
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
mkdir -p /nfs/hpc/share/$USER/isaac-sim-pruning-workflow/logs
slurm_clean sbatch hpc/slurm/isaac_headless_smoke.sbatch
```

Pass condition (all must be in the job log / `logs/isaac_smoke.json`):

1. The job lands on an RTX-capable node (`a40`, `rtx8000`, `l40s`, `h100`, or `h200`).
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

## 30 cm camera-rect and scripted baselines (v60 RTX)

```bash
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
slurm_clean sbatch hpc/slurm/camera_rect.sbatch
slurm_clean sbatch hpc/slurm/baselines.sbatch
```

Pass files: `docs/evidence/camera_rect_<jobid>.json` and
`docs/evidence/baselines_<jobid>.json` with `"ok": true`. Do not tick those
ROADMAP rows from a script that never ran on a GPU.

## URDF import

Job `21077217` (2026-08-28, `cn-r-2` A40, 34 s) converted the rewritten BDS
flatten. Root layer `artifacts/usd/ur5e_pruner_abs/ur5e_pruner_abs.usda` with
payloads (~3.9 MB). Contains `ur5e__shoulder_*` through `wrist_3` plus
`mock_pruner__{base,camera0,tof0,tof1,tool0}`. **No linear slider.** Importer
warned that PhysX joint drives had no stiffness/damping; `ArticulationCfg`
supplies those from `ur5e_pruner.yaml`.

## Cache policy

Jobs prefer `/scratch/$USER` for Omniverse and CUDA caches (`setup_node_cache`).
If that directory is missing, caches fall back to Lustre. Keep Kit caches and
generated USD off the stak home filesystem.
