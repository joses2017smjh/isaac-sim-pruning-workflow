# How Isaac Sim runs on this cluster

This is Jose's measured path from
[`bhl-robustness-ladder`](https://github.com/joses2017smjh/bhl-robustness-ladder/),
not a NVIDIA tutorial and not Isaac Sim 5.x. Pruning jobs use the same venv,
SIF, and `bhl_exec` launcher. Do not reinvent it.

## 1. The original blocker

Isaac Sim **5.1**'s RTX renderer segfaults here inside
`omni.usd.create_hydra_engine`. Any camera that needs rendering kills the
process. That is why BHL had no colour on the published stack.

## 2. First workaround — sidestep the renderer

`RayCasterCamera` intersects a pinhole ray bundle with the scene mesh in Warp
and never asks for a Hydra engine. That gave depth without RTX (BHL: 1.6% of
throughput at 4,096 envs; plane check 100% finite, 2.9% mean relative error).
A mesh ray-cast has no colour. Depth yes, RGB never.

For pruning, ray-cast ToF / Warp depth can still train on v51. RGB, rendered
depth, and anything that needs materials must use v60.

## 3. The actual fix — Isaac Sim 6.0

6.0 does not segfault. Second venv `Humanoid_Lite/venv-isaac60` beside the
locked 5.1 stack, selected by `BHL_STACK=v60`, so no published BHL number
moved. Probe (job `21036831`, then pruning job `21077170`): RGB is not a flat
frame; `distance_to_image_plane` is metric.

A bare `UsdGeom.Cube` with no material under a default light looks black.
Depth can still be correct. Always bind a `UsdPreviewSurface` and aim a light.
Pruning's cube/plane smoke does that.

## 4. The venv is not a full RL install

Job `21036909` died on `ModuleNotFoundError: No module named 'rsl_rl'`. v60
was built with Isaac Sim and Isaac Lab and **no RL library**. Installing the
trainer is a separate step. Pruning uses skrl; do not assume it is in
`venv-isaac60` until an import check on that interpreter says so.

## 5. Isaac Lab 3.x is warp-first

The renderer that works is Lab **3.0.0b2**. The inherited pruning
`DirectRLEnv` is Lab **2.x**. Lab 3 breakages BHL already paid for:

| Breakage | Fix |
|---|---|
| `AdditiveUniformNoiseCfg` removed | alias to `UniformNoiseCfg` |
| `SimulationCfg.physx` → `.physics` | lazy property shim |
| `asset.data.*` is warp `ProxyArray`; `isaaclab.utils.math` is torch.jit and rejects it | unwrap `.torch` (zero-copy) around scripted math |
| `ProxyArray.shape` is the warp shape; obs manager sizes vec3 as `()` | `.torch` at the observation boundary |
| `.dtype` is a ctypes type; `torch.tensor` rejects it | same |
| curriculum `env_ids` must have no default | remove the default |

Isaac Lab's own math functions fail on Isaac Lab's own data until unwrapped.
`isaaclab_pruning.sim.lab3_compat.apply()` is the pruning copy of that shim.
Call it before constructing the env on v60.

A 2.x number and a 6.0 / 3.x number are not the same table.

## 6. Cameras in the stage are not observations

BHL's smoke reported 9/9 ok with obs width **194 for every task** because the
cameras were mounted and never added as observation terms. Sighted arms would
have trained as copies of the blind arm. The obs column is the check: BHL
ended at 194 / 322 / 578.

Pruning variants A/B/C last-dims **must** disagree. C and D share width on the
8×8 ToF grid (content comparison); native-resolution C is a second table.
A shared `observation_space = 128` with `getattr(self, "flow", None)` falling
through is the BHL trap. `observation_width()` is the contract;
`PruningEnvCfg.__post_init__` sets `observation_space` from it. The batched
env smoke **asserts** `cfg.observation_space == obs.shape[-1]` per variant and
`not allclose(C, D)` when ToF ≠ metric. Do not substitute a log grep.

RGB at 8×8 is 192 numbers per camera against depth's 64. If RGB ever wins a
comparison, colour vs width is a separate question. Wrist RGB stays off until
a renderer job validates the geometrically selected `camera_offset`
(`close_lateral`, `[0.0, -0.06, 0.10]` m, `docs/evidence/camera_offset_raycast.json`).

## Batched env smoke (one RTX slot)

`hpc/slurm/env_smoke.sbatch` is the next gate. One job, in order: trainer
import on `$PY` inside `bhl_exec`, construct A/B/C/D cfgs, one DirectRLEnv,
assert last-dims, `env.step(0)`, read PhysX contacts, write
`docs/evidence/smoke_<jobid>.json`. v60 has `rsl_rl` and may not have `skrl`;
do not fight a dependency install on a beta Lab.
