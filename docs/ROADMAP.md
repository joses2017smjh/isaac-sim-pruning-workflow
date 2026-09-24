# Implementation gates

September 20 update: the six-task lighting pilot and two earlier full daylight
runs are complete and independently pass. The next priority is the
[Envy/UFO + lighting + learned-depth research audit](RESEARCH_AUDIT_2026-09-20.md),
starting with checkpoint exposure and storage. Its experiments are not queued.
The counts and milestone descriptions below retain their original dates.

This turns the research plan into falsifiable gates. A checked item means code
and evidence exist in this repository; it does not mean the entire phase is
complete.

September 14: the two-tree sequence in `21328323` passes all 17 independent
checks: live approach, gated surrogate release, 809.49 mm measured drop and
<0.001 mm final home error. Both original source trees participate in collisions
and ToF queries; only one known spur is selected. Local tests: **489 passed**,
one simulator-only test deselected, including ten daylight tests.

September 19 audit: morning/evening probes `21329420` / `21329421` each
contain 30 frames and pass capture validation. Their three-second duration
cannot establish release or return. The next bounded batch is the
[six-sequence lighting/tracker pilot](RESEARCH_EXPERIMENTS_2026-09-19.md);
see the [job ledger](../SLURM_JOBS.md) for submission status.
[Recording](ISAAC_RENDER.md) · [aggregate results](evidence/two_tree_summary_2026-09-14.json).
A local uncommitted browser-studio prototype now exists; it was preserved and
not evaluated as part of this experiment checkpoint.

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
- [x] Export the original `orchard_template.blend` meshes, UVs, six bark/soil
      maps, posts, wires, and Sun through USD; render them beside the original
      robot in job `21247873`. This is not a PLY import. Source/export hashes
      and presentation overrides are recorded; procedural sky, displacement,
      and ground-collision parity with Blender are not established.

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
- [ ] Execute a CuRobo plan to the pre-cut standoff. **Blocked on the pinned
      stack (time-boxed probe, September 23):** CuRobo is not installed in
      `venv-isaac60`, and installing it means compiling CUDA extensions against
      torch 2.11 + CUDA 13.0 into, or alongside, a venv that also serves the
      separate Humanoid_Lite project, then a GPU allocation to execute.
      Upstream is compatible in principle (Python >= 3.10, torch >= 2.9 for
      CUDA 13). Nothing was installed or modified. The PyPI name
      `nvidia-curobo` is an unrelated placeholder and must not be used.
      [Probe and path forward](evidence/curobo_feasibility_2026-09-23.json).

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
      training environment. The separate Blender demo now uses live classical
      RGB-D tracking for control, but does not close these training feeds.
      Farneback flow remains an offline diagnostic; RTX depth is simulator
      ground truth, not a learned metric-student prediction.
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

## Blender/live-vision demo — one two-tree sequence validated

- [x] Load the original textured orchard and select an existing spur component
      without editing the source `.blend` or substituting the robot.
- [x] Capture overview, close-up, wrist RGB, optical-Z depth, dual live ToF,
      measured PhysX robot state, and contact data with path tracing and OptiX
      denoising. No compositor median filter is applied to this recording.
- [x] Seed a branch pixel once and issue bounded Cartesian commands from fresh
      pyramidal-LK RGB-D measurements. Job `21247873` applied 60 vision commands;
      final tracking retained four features at confidence 0.9894.
- [x] Separate proposed from physically applied commands, hold on invalid
      tracking, and test fresh ToF/contact guards before detachment. The
      earlier contact and occlusion failures remain preserved, not relabelled.
- [x] Implement a visual jaw surrogate and discrete rigid-piece release with
      fresh-vision, geometry, stability, and hazard gates; cover the logic with
      CPU tests. This is not actuated blade CAD or simulated material fracture.
- [x] Record all 60 requested frames with ten capture checks passing. The
      six-second recording shows a 230.08 mm maximum tool displacement and
      final 35.01 mm mouth-to-tracked-target distance. Both ToF streams changed;
      valid-ray fractions were 33.54% / 33.28%, with noise disabled.
- [x] Independently validate approach, surrogate release, measured fall and
      home return in `21328323`: all 17 sequence checks pass for one target.
- [x] Keep both distinct original Blender trees in collision and ToF queries.
- [x] Add ten tests for morning/noon/evening world-space sunlight presets.
- [x] Revalidate the saved morning/evening 30-frame lighting probes; capture
      completeness passes, while full-sequence release/return remains unestablished.
- [x] Regrade completed array `21360571`: six raw/CLAHE source/morning/evening
      runs pass 17/17 checks on the same tree0 spur; historical failures remain.
      No CLAHE advantage or population generalization is established.
- [x] Publish the lighting pilot and decide the CLAHE candidate. All six planned
      tasks are accounted for with zero incomplete, regenerated by
      `tools/summarize_lighting_pilot.py` into
      [pilot results](evidence/lighting_pilot_results_2026-09-23.json). CLAHE is
      **dropped** as a rejected diagnostic: it changed no task outcome and no
      tracking continuity, failing the pre-registered comparison. Morning raw
      reached the tracker's own `min_features` floor of 4 while still completing.
- [x] Learned-depth offline and live-shadow gates: **both FAIL**. Stage A
      (`21370005`, Isaac RTX): 0 of 234 target frames within 20 mm, +0.4–0.6 m
      systematic overestimate. Live shadow (`21370047`): target read ~2.4× too
      far, RTX depth in control throughout. Learned control stays gated.
      [Stage A](evidence/stage_a_depth_2026-09-23.json) ·
      [execution record](RESEARCH_EXECUTION_2026-09-20.md#results-written-up--september-23-2026).
- [x] Paired Envy/UFO lighting pilot, one tree per family (`21370039`): evening
      raises DA2 tree-mask error 6.1× (Envy) and 7.9× (UFO); DINO adds nothing.
      Not a family or population claim at N = 1.
      [Evidence](evidence/family_pilot_depth_2026-09-23.json).
- [x] Eight-tree Envy/UFO × four-light matrix (`21402687` / `21402688`, all 8
      trees scored, ~8 GPU-min of 120 reserved). Lighting effect **replicated in
      8 of 8 trees** (evening 6.1–8.4× source). UFO worse than Envy under every
      light with no overlap between families, as a larger constant offset rather
      than a worse shape; part of the Envy advantage may be validation-split
      exposure. All gates fail, as registered. The spur target was invisible for
      3 of 8 trees, so target-level numbers rest on 5.
      [Protocol and result](EVAL_PROTOCOL_FAMILY_LIGHTING_2026-09-23.md#result--september-23-2026) ·
      [evidence](evidence/family_matrix_depth_2026-09-23.json).
- [x] Zero-GPU anchoring analysis, registered before running: one metric range
      at the target fixes UFO daylight (0.19 → 0.06 m) but not Envy (spur error
      is not the frame error; a many-point affine fit would, ceiling 0.04–0.06 m);
      nothing fixes evening (ceiling 0.10–0.13 m) or Isaac (ceiling 0.37–0.46 m).
      P1 refuted, P2 and P4 supported, P3 half refuted, all on the record.
      [Protocol and result](EVAL_PROTOCOL_DEPTH_ANCHORING_2026-09-23.md) ·
      [evidence](evidence/depth_anchoring_2026-09-23.json).
- [x] Many-zone anchoring (exploratory, ground-truth zone ranges, no sensor
      model): an 8×8 fit reaches the affine ceiling in every Blender cell and
      turns the close-range target error from 0.4–0.55 m into 0.006–0.02 m;
      on Isaac the ceiling itself is the limit (target 0.57 → 0.16 m).
      [Follow-up](EVAL_PROTOCOL_DEPTH_ANCHORING_2026-09-23.md#follow-up--many-zone-fit-september-23-2026-exploratory).
- [ ] The same fit on the two recorded 8×8 ToF grids (zone-to-pixel
      registration through the recorded poses; the simulation chain is fully
      recorded, the hardware extrinsic is not).
- [x] Single-axis generalization controls, registered with seven predictions
      before submission and judged on September 23 (P1 half, P2 and P3 and P7
      supported, P4 family-dependent, P5 supported with two misses, P6 refuted
      at frame level; the close-range shape is right in Cycles and wrong in
      Isaac, so the renderer control is now necessary): evening brightness against shadow structure
      (`evening_x2.6`, `overcast`, `overcast_div2.6`, four fixed test-time
      normalizations), the Isaac wrist camera model at matrix distance, close
      and 39.7°-pitched rigs at 0.16–0.39 m, an eight-distance sweep, and the
      public relative DA2 head through an all-GT disparity-affine ceiling on the
      controls, the matrix and Isaac Stage A. Frozen launcher
      `tools/queue_generalization_controls.py`; 180 GPU-minutes reserved.
      [Protocol and result](EVAL_PROTOCOL_GENERALIZATION_CONTROLS_2026-09-23.md#result--september-23-2026) ·
      [evidence](evidence/generalization_controls_2026-09-23.json).
- [x] Renderer control for Stage A: the original orchard tree in Cycles at the
      recorded wrist poses (mapping verified to 1e-6 m), two barks. Cycles
      reproduces the Isaac over-estimate (+0.41 m signed, target 0.45 m) from
      the same tree and poses; bark has no effect; the shape ceiling is 0.18 m
      in Cycles against 0.46 m on Isaac, so RTX appearance owns the shape part
      and none of the magnitude.
      [Protocol and result](EVAL_PROTOCOL_TREE0_REPLAY_2026-09-23.md#result--september-23-2026) ·
      [evidence](evidence/tree0_replay_2026-09-23.json).
- [x] Re-fine-tune with photometric jitter against a control arm (repository-
      local trainer, warm start, 6 epochs on the 6,270 surviving frames, about
      3 h per arm on an A40): the jitter cuts Envy evening error to a third
      and fixes the darkness cell in both families, but helps UFO evening by
      only 14–18% and leaves the evening shape ceiling unchanged; the control
      arm changes nothing. Source cost 0.005–0.024 m.
      [Protocol and result](EVAL_PROTOCOL_FINETUNE_2026-09-23.md#result--september-24-2026) ·
      [evidence](evidence/finetune_family_matrix_2026-09-24.json).
- [ ] Re-fine-tune with rendered lighting variation (sun angle, colour, sky):
      the lever the jitter result leaves for the low-sun shading. Needs a
      training-render launcher over L-Py trees outside the matrix register.
- [ ] Vision-guided controller on an Envy or UFO tree in Isaac. No path exists:
      `RenderEnv` presents the Blender orchard export, and the L-Py cylinder USDs
      have no spur-selection route into it. Needs target selection over cylinder
      metadata, a spawn path, and its own registration.
- [ ] Test an explicitly selected target on original orchard tree1 separately.
- [x] Ship a replay studio: a static page that plays recorded runs frame by frame
      with camera video, tracker confidence and feature count, both 8x8
      time-of-flight grids with validity, gate states and proposed against
      applied commands. Three runs are published: the `21328323` success, the
      `21317409` closure failure and a success-rate sweep failure that stopped on
      hazard contact. Every displayed value is read from a capture file; there is
      no browser physics and no policy. Published payload is 1.3 MB against a
      12 MB budget, and CI fails if a restricted asset reaches it.
- [ ] Decide the fate of the earlier uncommitted browser scene-viewer prototype
      (`studio/src`). It animates a synthetic joint loop rather than a recording,
      and its asset exporter copies mock-pruner CAD, so it is not published.
- [x] Measure a task-success rate over pre-registered targets. Result:
      **0 / 40** (Wilson 95% 0–0.088) over 20 seeded spurs × source and
      morning light, with a failure taxonomy from recorded stop reasons.
      [Protocol and result](EVAL_PROTOCOL_2026-09-23.md#result--september-23-2026) ·
      [evidence](evidence/eval_2026-09-23.json). Only tree0 was actually
      measured; see the next gate. No collision-avoidance claim is made.
- [ ] Evaluate tree1 targets. The registered tree1 spurs were outside the
      renderer's accepted candidates and never ran. Needs a new registration
      drawn from the listed candidates, or the geometry audit the renderer
      requires for unlisted components.

The latest recorded outcome is `vision_guided_simulated_detachment_and_retreat`. Known mesh metadata
provides branch identity, axis, and radius; classical image tracking supplies
position updates. This demonstration does not implement learned recognition,
learned depth, PPO training, CuRobo execution, calibrated physical cameras,
blade mechanics, or wood fracture. Those remain separate gates above.

## Phase 5 — evaluation

- [x] Episode metrics: success, cut error, perpendicularity, collisions, steps.
- [x] Success vs injected cut-point error bins.
- [x] 30 cm camera-rect helper (`CAMERA_RECT_DEPTH = 0.30`).
- [x] Synthetic ranking-inversion unit test (not Isaac-vs-PyBullet results).
- [ ] Held-out Envy `00042` / `00065` and untouched UFO rollouts in Isaac.
- [ ] PyBullet sim2sim numbers.
- [ ] 30 cm box rendered in Isaac and compared to Blender.
- [x] ROS 2 **software**-in-the-loop: the perception-to-control loop runs as a
      ROS 2 Humble graph on recorded sensor streams, wrapping the existing
      tracker, bounded command and gates rather than reimplementing them.
      Replaying `21328323` reproduces **199/199 recorded decision states**, with
      command deltas agreeing to a 2 mm stated tolerance (median 0.07 mm).
      RGB-blackout and depth-dropout controls hold on 39/39 frames and never
      authorize approach or release. [Notes](ROS2_SIL.md) ·
      [evidence](evidence/ros2_sil_parity_2026-09-23.json).
      A C++ port of the time-of-flight deprojection is checked against the
      Python original on recorded grids (`ros2/pruning_sil_cpp`, gtest).
- [ ] ROS 2 **hardware**-in-the-loop demo: pending hardware. Requires a physical
      VL53L8CX pair, a calibrated wrist camera and a UR5e driver. The interface
      mapping and the list of what would change are in
      [ROS2_SIL.md](ROS2_SIL.md#what-would-change-for-hardware); none of it has
      been exercised on a rig.

## Authorized research execution (2026-09-20)

Implementation and jobs are now in progress; the preceding audit-only snapshot is historical. Frozen DA2 offline evaluation job **21370005** submitted, no dependencies. The one-frame CPU accuracy gate failed; learned control remains conditional. Checkpoint-specific leakage corrections, current job/result states and storage are tracked in the [execution record](RESEARCH_EXECUTION_2026-09-20.md) and `docs/evidence/research_execution_2026-09-20.json`.
