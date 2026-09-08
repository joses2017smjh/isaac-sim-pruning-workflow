# Implementation gates

This turns the research plan into falsifiable gates. A checked item means code
and evidence exist in this repository; it does not mean the entire phase is
complete.

Latest demonstrated milestone: job `21208215` completed a **14-second Isaac
inspection** and a separate short motion/ToF smoke. The original UR5e works;
the clip does not cut wood or run a learned vision policy. See the
[recording and evidence](ISAAC_RENDER.md).

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
- [x] Render a procedural finite-cylinder tree fixture and backdrop beside the
      original UR5e in Isaac (`21208215`). This demonstration scene is authored
      in USD; it is not a Blender PLY import or a held-out orchard evaluation.

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
      BDS camera0 frame. The inspection renderer uses a separate fixed exterior
      mount and toe-in; this ray-cast candidate is not its camera calibration.
- [x] Generate URDF from pinned BDS Xacro/config and selected UR5e calibration;
      record source/calibration/generated-file/mesh hashes and fixed transforms
      (`docs/evidence/urdf_generation_ur5e_mock_pruner_bdsdfede4c0_ur18e6f603_calib_3941312424972580002_urdf6b02ce9330be.json`).
- [x] Import that fresh URDF and promote the exact content-addressed root
      (`docs/evidence/urdf_import_21136450.json`: `status: complete`, `ok: true`,
      six UR joints, no slider, provenance-verified fixed transforms).
- [x] GPU-validate the implemented pair of live 8x8
      `MultiMeshRayCasterCamera` sensors. They track `mock_pruner__base` with
      the reviewed `mock_pruner__tof0/tof1` offsets. The short
      [smoke `21208215`](evidence/smoke_21208215.json) records median range changes
      of 3.961 / 3.800 mm over 64 shared finite pixels per sensor after a
      controlled 5 mm tool command. Noise was disabled.
- [x] Render overview, close-up and wrist RGB plus simulator depth on the pinned
      stack. [Job `21208215`](evidence/render_21208215.json) records 140 frames
      and a fixed simulation-defined wrist-camera transform.
- [ ] Identify the physical camera model and calibrated optical transform.
- [ ] Validate the camera with the separate 30 cm rectangle depth gate.

Hardware gate: the BDS Xacro has a camera0 translation and its CAD archive has a
RealSense-named mount, but model/optical calibration remain unknown. This does
not block a simulation-defined camera. The inspection RGB now works; camera
feeds into the training policy remain unfinished.

## Phase 3 — task and baselines

- [x] Add batched mouth/failure OBB intersection and perpendicularity gates.
- [x] Add a ground-truth cut-point oracle ordered by radius then neighbourhood.
- [x] Add all-nearby-wood failure-zone broad phase.
- [x] Add dense reward with an alignment-weight ablation hook.
- [x] Add radius/neighbourhood curriculum (thick branch → thin spur).
- [x] Add scripted ToF pan/pitch/roll/approach (original reimplementation).
- [x] Transform reviewed base-frame ToF points to `mock_pruner__tool0` in the
      scripted baseline and regression-test the 8 cm standoff. The short smoke
      now validates the control-tool pose and live sensor motion response;
      it does not measure the scripted baseline's task success rate.
- [x] Add CuRobo UR5e placeholder spheres and a not-yet-configured status.
- [x] Instrument all 20 rigid bodies with PhysX contact reporting and verify
      exact-name coverage in [smoke `21208215`](evidence/smoke_21208215.json).
      The previous one-body tensor missed approximately 180 N of floor force
      at `mock_pruner__base`, exposed by [job `21201622`](evidence/smoke_21201622.json).
      The passing fixture elevates the base by 0.70 m; it does not declare the
      original floor-level fixture fixed. Historical failures remain in the
      [job ledger](../SLURM_JOBS.md).
- [x] Pass the unchanged 5 mm hold gate in the raised fixture. Six hold steps
      recorded zero drift at output precision; a 5 mm command ended 0.360 mm
      from its target. Control uses bounded SVD damped least squares and
      measured gravity compensation.
- [ ] Validate long-duration holding, contact response and collision avoidance
      across poses and environments. Instrumentation and a no-contact
      inspection episode do not establish those properties.
- [ ] Integrate reset-time tree-oracle/curriculum selection and nearby-wood
      tensors into the training environment; its target remains a smoke fixture.
- [ ] Implement blade actuation and validated cutting/wood-severing mechanics.
- [x] Configure the CuRobo UR5e oracle on the imported USD
      (`docs/evidence/curobo_spheres.json`: link bounding spheres from
      pybullet-tree-sim collision STLs). Runtime still needs an Isaac job.
- [ ] Run both baselines on live sensor observations in Isaac
      (`hpc/slurm/baselines.sbatch`). The short environment gate is now green,
      but the inspection trajectory is not this baseline evaluation and the
      current CuRobo path reports readiness without executing a plan.

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
- [x] Pass a one-environment v60 smoke in job `21208215`: import `rsl_rl`, build
      A-D configs, construct the B environment, assert A-D observation widths
      150/278/86/86, step, verify 20 contact bodies and sensor transforms, and
      measure a controlled geometry-response delta. This is not four trained
      policies or batched throughput evidence; `skrl` was unavailable in that run.
- [ ] Close all live policy observation feeds. Dual ToF has passing GPU
      evidence; flow is still zero and metric-student depth is constant in the
      training environment. The video's Farneback flow and color segmentation
      run offline on recorded RGB; RTX depth is simulator ground truth.
- [ ] Validate the actual PPO trainer/dependency integration. The successful
      environment smoke does not make `tools/train.py` a training runner.
- [ ] Evaluate a native Lab 3 port before adding further compatibility shims.
      The current pinned-stack smoke passes with the existing surface shims;
      it is not evidence of compatibility with arbitrary future Lab releases.
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
- [x] Verify installation and demo generation in an isolated environment and
      GitHub Actions. The September 5 checkpoint passed 133 local CPU tests;
      this is a historical count, not the current suite size.

This uses ideal tool motion and an explicitly synthetic metric estimate.
It closes the portable demonstration path, not the remaining Isaac/PPO gates.
See [capture instructions](DEMO.md) and [reviewer gaps](REVIEWER_NOTES.md).

## Isaac inspection demo — completed 2026-09-07

- [x] Render the original UR5e/mock-pruner under physical joint drives in a
      procedural USD tree scene, without a trained policy or robot substitution.
- [x] Capture 140 frames at 10 fps with RTX RGB/depth, two changing live ToF
      grids, measured joint/tool state, and contact coverage.
- [x] Observe approach and retreat: maximum tool displacement 247.37 mm,
      closest target distance 96.07 mm, return error 1.65 mm.
- [x] Publish the recording with offline CV overlays and a labelled display-only
      median filter; retain raw inputs, measurements, source/stack fingerprints,
      and the earlier failed rendering.
- [x] Validate capture completeness and demonstrate that extra RTX render
      updates do not advance physics.

Outcome: `approach_inspect_retreat_no_cut`. The known-geometry trajectory uses
a live ToF stop gate; it is not driven by the CV overlays. ToF noise is disabled,
and the stop gate is a diagnostic guard, not a validated hardware safety system.
This milestone closes the requested robot/environment/sensor visualization,
not the full autonomous pruning workflow. [Video and reproduction](ISAAC_RENDER.md).

## Phase 5 — evaluation

- [x] Episode metrics: success, cut error, perpendicularity, collisions, steps.
- [x] Success vs injected cut-point error bins.
- [x] 30 cm camera-rect helper (`CAMERA_RECT_DEPTH = 0.30`).
- [x] Synthetic ranking-inversion unit test (not Isaac-vs-PyBullet results).
- [ ] Held-out Envy `00042` / `00065` and untouched UFO rollouts in Isaac.
- [ ] PyBullet sim2sim numbers.
- [ ] 30 cm box rendered in Isaac and compared to Blender.
- [ ] ROS 2 hardware-in-the-loop demo (stretch).
