# Implementation gates

This turns the research plan into falsifiable gates. A checked item means code
and evidence exist in this repository; it does not mean the entire phase is
complete.

## Phase 0 — contracts and compute

- [x] Preserve the now-unavailable upstream Isaac Lab harness at `5701a77` in
      Jose's fork history.
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
- [x] Match one Blender pose at median trunk error below 2 mm
      (`docs/evidence/blender_trunk_mm_lpy_envy_00000.json`: Envy `00000` shot 1,
      orchard tilt −17.143°, median trunk **0.00055 mm**).
- [x] Convert all 100 Envy + 100 UFO assets
      (`docs/evidence/trees_converted_manifest.json`: ASCII USDA + `bark_brown_02`,
      1798 cylinders on Envy `00000`, 2960 on UFO `00000`, held-out Envy included as assets).
- [x] Bind `bark_brown_02` as UsdPreviewSurface (tree USDA `Looks/bark_brown_02` and
      orchard Looks; hydra still needs a light, same as Gate 0).

Why cylinders, not capsules: the Blender generator uses finite cylinders.
Capsules change each end by a radius and cannot pass a millimetre depth check.

## Phase 2 — robot and sensors

- [x] Record the two reviewed mock-pruner ToF offsets and their source parent
      `mock_pruner__base` separately from control EEF `mock_pruner__tool0`.
- [x] Add batched range noise, status, random dropout, and thin-target dropout.
- [x] Encode UR5e joint names, limits, and actuators (`ur5e__`). Slider is
      documented on the real machine and is **not** spawned in v1
      (`joint_names_expr` is arm-only; imported USD has no slider).
- [x] Place versioned legacy mouth/failure proxy AABBs in the EEF frame.
- [x] Fit OBBs from binary STL (run `tools/fit_cutter_boxes.py` after fetch).
- [x] Inverse-variance depth fusion for variant D.
- [x] Wrist-camera extrinsic *candidates* with geometric ray-cast scoring
      (cylinder colliders + jaw AABB; no renderer).
- [x] Prove the tracked BDS generated snapshot imports to USD (job `21077217`,
      no slider). This snapshot has stale ToF/tool fixed transforms.
- [x] Mark the stale snapshot non-runtime and reject default articulation spawn;
      retain an explicit diagnostic-only override.
- [x] Resolve package mesh paths and author `ArticulationCfg` against that USD.
- [x] Replace nominal cutter boxes with fitted STL AABBs (`docs/evidence/cutter_boxes_fitted.json`).
- [x] Select and document wrist-camera extrinsics
      (`docs/evidence/camera_offset_raycast.json`: `close_lateral` `[0, -0.06, 0.10]` m,
      1259/1478 cuts visible). This is a simulation candidate, not the physical
      BDS camera0 frame. Wrist RGB stays `enabled: false`.
- [x] Generate URDF from pinned BDS Xacro/config and selected UR5e calibration;
      record source/calibration/generated-file/mesh hashes and fixed transforms
      (`docs/evidence/urdf_generation_ur5e_mock_pruner_bdsdfede4c0_ur18e6f603_calib_3941312424972580002_urdf6b02ce9330be.json`).
- [x] Import that fresh URDF and promote the exact content-addressed root
      (`docs/evidence/urdf_import_21136450.json`: `status: complete`, `ok: true`,
      six UR joints, no slider, provenance-verified fixed transforms).
- [ ] GPU-validate the implemented pair of live 8x8
      `MultiMeshRayCasterCamera` sensors. They track `mock_pruner__base` with
      the reviewed `mock_pruner__tof0/tof1` offsets; require both range tables
      to respond to controlled EEF motion before accepting the runtime gate.
- [ ] Identify the physical camera model and calibrated optical transform; then
      renderer-check the selected simulation view.

Hard gate: the BDS Xacro has a camera0 translation and its CAD archive has a
RealSense-named mount, but model/optical calibration remain unknown.
`close_lateral` is only the ray-cast simulation winner. Wrist RGB stays off.

## Phase 3 — task and baselines

- [x] Add batched mouth/failure OBB intersection and perpendicularity gates.
- [x] Add a ground-truth cut-point oracle ordered by radius then neighbourhood.
- [x] Add all-nearby-wood failure-zone broad phase.
- [x] Add dense reward with an alignment-weight ablation hook.
- [x] Add radius/neighbourhood curriculum (thick branch → thin spur).
- [x] Add scripted ToF pan/pitch/roll/approach (original reimplementation).
- [x] Transform reviewed base-frame ToF points to `mock_pruner__tool0` in the
      scripted baseline and regression-test the 8 cm standoff. Runtime link-pose
      validation remains open until the live-sensor smoke passes.
- [x] Add CuRobo UR5e placeholder spheres and a not-yet-configured status.
- [ ] Validate contact/collision state from the wired PhysX `ContactSensor`.
      Job `21146271` failed opaquely at `phase: construct` because its evidence
      flush preceded cleanup/exception capture. Job `21153271` then preserved
      the exact unresolved `{ENV_REGEX_NS}/Robot` error; directly constructed
      assets now use globally rooted v60 paths. Job `21153411` next proved
      entity resolution preceded PhysX articulation-view creation. Resolution
      is now post-super and contact reporting is explicitly activated. Job
      `21153625` exposed the raw Warp Jacobian; that and quaternion ordering
      are corrected. Jobs `21185961`/`21186027` now step, but the diagnostic
      measured 20.12 mm hold drift. The contact tensor covers one body, so
      full-arm coverage still needs verification (see `SLURM_JOBS.md`).
- [x] Configure the CuRobo UR5e oracle on the imported USD
      (`docs/evidence/curobo_spheres.json`: link bounding spheres from
      pybullet-tree-sim collision STLs). Runtime still needs an Isaac job.
- [ ] Run both baselines on live sensor observations in Isaac
      (`hpc/slurm/baselines.sbatch`), after a green environment smoke.

Hard gate: do not report a learned policy without scripted and oracle baselines.
`tools/train.py` refuses to start if those flags are unset.

## Phase 4 — policies and the robustness ladder

- [x] Observation builders/contracts A flow / B ToF / C metric-student / D fused.
- [x] Five-seed protocol (`0..4`) and 20 run IDs.
- [x] skrl PPO config sized for the pruning task.
- [x] Continuous ladder `d ∈ [0, 1]` with the plan's randomization axes.
- [x] Injected cut-point error for the perception-sensitivity sweep.
- [x] Variant A/B/C observation widths must differ; C/D match at 8×8
      (`observation_width()`, `PruningEnvCfg.__post_init__`).
- [ ] Close all live observation feeds: dual ToF is implemented but awaits a
      passing GPU smoke; flow and metric-student remain placeholders.
- [ ] One v60 job: trainer import + env construct A–D + obs asserts + step +
      PhysX contact + sensor prim/transform evidence + geometry-response delta
      (`hpc/slurm/env_smoke.sbatch` → `docs/evidence/smoke_<jobid>.json`).
      Attempt `21146271` is an opaque construct failure and `21153271` is the
      diagnosed global-path failure. Job `21153411` diagnosed the pre-physics
      entity-resolution failure. Job `21186027` measured two complete live
      ToF grids but failed the hold-drift threshold before controlled motion.
      No runtime pass is claimed and no workflow job remains queued.
- [ ] Port DirectRLEnv through Lab 3.x if that smoke needs more than the three
      existing surface shims — subclass Lab 3 rather than add a fourth.
- [ ] Train variants A–D × 5 seeds on ray-cast ToF, only after the live env
      smoke and both baseline gates pass.
- [ ] Per-axis ladder sensitivity.

The protocol places DA2-ft once at episode start, outside PPO. That inference
integration and the trainer remain unfinished; `da2_in_ppo_loop: false` is a
tested configuration contract, not evidence of a running model.

## Portable demo — completed 2026-09-05

- [x] Procedural scene, exact finite-cylinder ToF ray casts, configured noise,
      scripted servo, bounded insertion, and geometric cut/stop decisions.
- [x] Measured clear-approach, blackout, and nearby-wood scenarios with strict
      JSON replay, 18-second GIF, poster, and offline sensor-frame scrubber.
- [x] Correct lateral feedback direction, roll axis, invalid-depth fusion,
      and test these through the complete CPU loop.
- [x] Verify installation and all 133 CPU tests in an isolated environment;
      add demo generation to GitHub Actions.

This uses ideal tool motion and an explicitly synthetic metric estimate.
It closes the portable demonstration path, not the remaining Isaac/PPO gates.
See [capture instructions](DEMO.md) and [reviewer gaps](REVIEWER_NOTES.md).

## Phase 5 — evaluation

- [x] Episode metrics: success, cut error, perpendicularity, collisions, steps.
- [x] Success vs injected cut-point error bins.
- [x] 30 cm camera-rect helper (`CAMERA_RECT_DEPTH = 0.30`).
- [x] Synthetic ranking-inversion unit test (not Isaac-vs-PyBullet results).
- [ ] Held-out Envy `00042` / `00065` and untouched UFO rollouts in Isaac.
- [ ] PyBullet sim2sim numbers.
- [ ] 30 cm box rendered in Isaac and compared to Blender.
- [ ] ROS 2 hardware-in-the-loop demo (stretch).
