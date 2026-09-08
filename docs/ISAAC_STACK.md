# How Isaac Sim runs on this cluster

This is Jose's measured path from
[`bhl-robustness-ladder`](https://github.com/joses2017smjh/bhl-robustness-ladder/),
not a NVIDIA tutorial and not Isaac Sim 5.x. Pruning jobs use the same venv,
SIF, and `bhl_exec` launcher. Do not reinvent it.

Pruning now has its own complete robot render: job `21208215` recorded 140
frames and passed the short environment smoke on 2026-09-07. Watch the
[Isaac inspection demo](ISAAC_RENDER.md); inspect the
[smoke](evidence/smoke_21208215.json) and [render](evidence/render_21208215.json)
reports. The robot is the original UR5e with the reviewed mock pruner.

## 1. The original blocker

Isaac Sim **5.1**'s RTX renderer segfaults here inside
`omni.usd.create_hydra_engine`. Any camera that needs rendering kills the
process. That is why BHL had no colour on the published stack.

## 2. First workaround — sidestep the renderer

`RayCasterCamera` intersects a pinhole ray bundle with the scene mesh in Warp
and never asks for a Hydra engine. That gave depth without RTX (BHL: 1.6% of
throughput at 4,096 envs; plane check 100% finite, 2.9% mean relative error).
A mesh ray-cast has no colour. Depth yes, RGB never.

The pruning environment now instantiates two 8x8
`MultiMeshRayCasterCamera` sensors and consumes their 15 Hz
`distance_to_camera` output. That implementation targets the pinned v60 API;
it is not evidence of v51 compatibility. The v60 environment smoke passed in
job `21208215`, including a changed range table after controlled motion. RGB,
rendered depth, and anything that needs materials must use v60.

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
trainer is a separate step. The pruning training entry point expects skrl;
do not assume it is in `venv-isaac60`. Job `21208215` imported `rsl_rl`, but
`skrl` was absent. A successful robot render is not evidence of a working PPO
training run.

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

Width checks remain insufficient on their own. Flow is still zero and metric
depth is still fixed at 1.20 m, but ToF no longer comes from the old 0.40/0.42 m
constants: its buffers start invalid and are refreshed from two registered
`MultiMeshRayCasterCamera` instances. The smoke therefore must record both
sensor frames/poses and prove their raw 8x8 tables change after controlled EEF
motion against the opt-in, non-colliding cuboid target. Job `21208215` passed:
64/64 shared finite pixels per sensor changed by median **3.961 / 3.800 mm**
after a commanded 5 mm motion. Observation widths were **150 / 278 / 86 / 86**
for A/B/C/D. Those widths validate the interface, not live flow or learned depth.

RGB at 8×8 is 192 numbers per camera against depth's 64. If RGB ever wins a
comparison, colour vs width is a separate question. BDS has a hard-coded
camera0 translation and a RealSense-named CAD mount, but the exact model and
calibrated optical transform are unresolved. `close_lateral`
(`[0.0, -0.06, 0.10]` m from the control EEF) is a separate simulation
candidate. The inspection renderer now uses an explicitly simulation-defined
exterior wrist mount at `[0.0, -0.09, -0.025]` m in the control-tool frame, with
a fixed toe-in initialized toward the known fixture. It records RTX RGB and
ground-truth metric depth without claiming hardware calibration. These recorded
images do not replace the policy observation placeholders. The dashboard
computes Farneback flow and a brown-pixel candidate overlay offline. See
[`ROBOT_SENSOR_SOURCES.md`](ROBOT_SENSOR_SOURCES.md) and
[`ISAAC_RENDER.md`](ISAAC_RENDER.md).

## Batched env smoke (one RTX slot)

The source/import prerequisites are now closed: job `21136450` passed and its
content-addressed USD is selected by the robot config, while the dual-ToF
wiring exists in the environment. `hpc/slurm/env_smoke.sbatch` remains the GPU
gate. It imports the available trainer, constructs the A/B/C/D configs and one
DirectRLEnv, asserts last-dims, holds the absolute tool pose, reads PhysX
contacts, then commands a 5 mm optical-axis motion and requires both ToF tables
to respond before writing a green `docs/evidence/smoke_<jobid>.json`.

Attempt `21146271` reached `phase: construct` but failed with `ok: false` and no
traceback because the old cleanup ordering swallowed the caught diagnostic.
Retry `21153271` wrote the exact error first: the manually constructed
`DirectRLEnv` passed `{ENV_REGEX_NS}/Robot` directly into a v60 spawn function,
which requires a globally rooted path. All directly constructed robot/sensor/
target expressions now use `/World/envs/env_.*/...`. Job `21153411` registered
all four PhysX backends, then proved `SceneEntityCfg.resolve()` still ran before
physics created the articulation `_root_view`. Resolution is now post-super and
the USD spawner enables the contact-report API required by v60. Job `21153625`
failed on the raw Warp Jacobian. The runtime now reads the link-origin Jacobian
and converts Lab `xyzw` poses at the core `wxyz` boundary. Subsequent jobs
`21185961` and `21186027` step successfully but fail the hold-drift gate:
the latter measured 20.12 mm against a 5 mm limit. Both live ToF grids are
finite; those historical attempts never reached the controlled-motion test.

Job `21201622` exposed the missing contact instrumentation: the unmerged URDF
nests rigid links, so generic contact activation stopped at an ancestor.
Exact-path sensors now cover all **20** rigid bodies. They measured about
**180 N** on `mock_pruner__base` in the floor-level fixture while the independent
tool-pose paths agreed. The successful `21208215` smoke mounted the base 0.70 m
above the floor, with its wall raised by the same amount. It retained the
**5 mm** hold limit and measured zero drift at recorded precision over six
60 Hz steps; the subsequent 5 mm motion ended with **0.360 mm** error.

The fixed-base Jacobian remains indexed by `body_index - 1`, with explicit
handling of Lab's all-joint `slice(None)`. The controller now uses SVD damped
least squares (`0.05` damping), uniformly bounds joint corrections to `0.05`
rad per update, and applies dynamics gravity compensation `+g(q)`. Gravity and
the 800/40 arm drive gains remain enabled. The original floor-level failure
has not been relabelled as a pass.

In the same detached allocation, the renderer captured **14.0 physics seconds**
of approach, inspection, and retreat. All capture checks passed; render-only
updates advanced no physics steps. The tool moved **247.37 mm** and returned
within **1.653 mm** of the first captured position. Both ToF grids updated;
**40.01% / 42.31%** of their individual rays were valid across the sequence.
All measured contact forces stayed zero, so this is not a collision-response
test. It does not actuate a cut, sever wood, run a learned policy, or supply
offline CV results to the controller.

No further GPU job is needed for this video. Baselines, learned observations,
and PPO still need separate implementation/runtime evidence. The portable
[CPU demo](DEMO.md) remains a different analytic workflow. Use the pinned lock,
capture wrapper, and media commands in [ISAAC_RENDER.md](ISAAC_RENDER.md) to
reproduce the actual Isaac output; the [SLURM ledger](../SLURM_JOBS.md) records
both successful and failed attempts.
