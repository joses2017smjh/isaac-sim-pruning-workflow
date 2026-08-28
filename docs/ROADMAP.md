# Implementation gates

This turns the research plan into falsifiable gates. A checked item means code
and evidence exist in this repository; it does not mean the entire phase is
complete.

## Phase 0 — contracts and compute

- [x] Preserve the upstream Isaac Lab harness at `5701a77`.
- [x] Record immutable revisions for external robot/orchard sources.
- [x] Implement OpenCV `T_wc` plus legacy Blender-Euler annotation output.
- [x] Add an analytic three-view 1 m cube reconstruction test.
- [x] Distinguish planar z-depth from Euclidean camera range in code and tests.
- [x] Run the cube/plane test in Isaac Sim on an RTX node (job `21077170`, A40 `cn-r-4`).
- [x] Pin the working stack: `bhl.sif` sha256 `d427d9c3…9bcac29` + `venv-isaac60` (Isaac Sim 6.0.0.1 / Lab 3.0.0b2).

Hard gate for Blender comparison: cube/plane in Isaac (job `21077170`) is
green. The remaining millimetre check is Phase 1 (one Blender pose, trunk
median < 2 mm).

## Phase 1 — trees and orchard

- [x] Add validated L-Py `cylinder_data` and full-world-sidecar loaders.
- [x] Add direct `UsdGeom.Cylinder` authoring with collision LOD.
- [x] Add trunk/branch collision LOD and active-cut neighbourhood support.
- [x] Author V-trellis posts, wires, ground, and lighting as ASCII USDA.
- [x] Record debug trees `00000`–`00009` and held-out `00042` / `00065`.
- [x] Add batch metadata→USD conversion that skips held-out trees by default.
- [x] Convert and validate Envy `00000`–`00009` through Isaac (`pxr`, 1798 cylinders on `00000`, metres, Z-up).
- [ ] Match one Blender pose at median trunk error below 2 mm.
- [ ] Convert all 100 Envy + 100 UFO assets.
- [ ] Bind `bark_brown_02` as UsdPreviewSurface (needs Isaac materials).

Why cylinders, not capsules: the Blender generator uses finite cylinders.
Capsules change each end by a radius and cannot pass a millimetre depth check.

## Phase 2 — robot and sensors

- [x] Encode the two real mock-pruner ToF offsets without inventing a camera pose.
- [x] Add batched range noise, status, random dropout, and thin-target dropout.
- [x] Encode UR5e joint names, limits, and actuators (`ur5e__`). Slider is
      documented on the real machine and is **not** spawned in v1
      (`joint_names_expr` is arm-only; imported USD has no slider).
- [x] Place mouth/failure AABBs in the EEF frame.
- [x] Fit OBBs from binary STL (run `tools/fit_cutter_boxes.py` after fetch).
- [x] Inverse-variance depth fusion for variant D.
- [x] Wrist-camera extrinsic *candidates* with geometric ray-cast scoring
      (cylinder colliders + jaw AABB; no renderer).
- [x] Import the BDS UR5e + mock-pruner URDF to USD (job `21077217`, no slider).
- [x] Resolve package mesh paths and author `ArticulationCfg` against that USD.
- [x] Replace nominal cutter boxes with fitted STL AABBs (`docs/evidence/cutter_boxes_fitted.json`).
- [ ] Select and document wrist-camera extrinsics.

Hard gate: `camera_offset` stays unjustified until the ray-cast (or a later
visibility/occlusion) experiment names a pose. Scoring does not need RTX.

## Phase 3 — task and baselines

- [x] Add batched mouth/failure OBB intersection and perpendicularity gates.
- [x] Add a ground-truth cut-point oracle ordered by radius then neighbourhood.
- [x] Add all-nearby-wood failure-zone broad phase.
- [x] Add dense reward with an alignment-weight ablation hook.
- [x] Add radius/neighbourhood curriculum (thick branch → thin spur).
- [x] Add scripted ToF pan/pitch/roll/approach (original reimplementation).
- [x] Add CuRobo UR5e placeholder spheres and a not-yet-configured status.
- [ ] Add contact/collision state from PhysX (ContactSensor wired; needs env-smoke JSON).
- [ ] Configure the CuRobo UR5e oracle on the imported USD.
- [ ] Run both baselines in Isaac.

Hard gate: do not report a learned policy without scripted and oracle baselines.
`tools/train.py` refuses to start if those flags are unset.

## Phase 4 — policies and the robustness ladder

- [x] Observation variants A flow / B ToF / C metric-student / D fused.
- [x] Five-seed protocol (`0..4`) and 20 run IDs.
- [x] skrl PPO config sized for the pruning task.
- [x] Continuous ladder `d ∈ [0, 1]` with the plan's randomization axes.
- [x] Injected cut-point error for the perception-sensitivity sweep.
- [x] Variant A/B/C observation widths must differ; C/D match at 8×8
      (`observation_width()`, `PruningEnvCfg.__post_init__`).
- [ ] One v60 job: trainer import + env construct A–D + obs asserts + step +
      PhysX contact (`hpc/slurm/env_smoke.sbatch` → `docs/evidence/smoke_<jobid>.json`).
- [ ] Port DirectRLEnv through Lab 3.x if that smoke needs more than the three
      existing surface shims — subclass Lab 3 rather than add a fourth.
- [ ] Train variants A–D × 5 seeds on ray-cast ToF.
- [ ] Per-axis ladder sensitivity.

DA2-ft runs once to propose a target, not at every PPO step. Protocol field
`da2_in_ppo_loop: false` is asserted.

## Phase 5 — evaluation

- [x] Episode metrics: success, cut error, perpendicularity, collisions, steps.
- [x] Success vs injected cut-point error bins.
- [x] 30 cm camera-rect helper (`CAMERA_RECT_DEPTH = 0.30`).
- [x] Isaac vs PyBullet ranking-inversion test.
- [ ] Held-out Envy `00042` / `00065` and untouched UFO rollouts in Isaac.
- [ ] PyBullet sim2sim numbers.
- [ ] 30 cm box rendered in Isaac and compared to Blender.
- [ ] ROS 2 hardware-in-the-loop demo (stretch).
