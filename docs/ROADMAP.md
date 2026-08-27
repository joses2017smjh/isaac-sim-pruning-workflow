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
- [ ] Run the cube/plane test in Isaac Sim on an RTX node.
- [ ] Pin an Isaac Sim + Isaac Lab container by digest and pass headless smoke.

Hard gate: do not compare Isaac and Blender depth until both unchecked items
above pass.

## Phase 1 — trees and orchard

- [x] Add validated L-Py `cylinder_data` and full-world-sidecar loaders.
- [x] Add direct `UsdGeom.Cylinder` authoring with class/instance semantics.
- [x] Add trunk/branch collision LOD and active-cut neighbourhood support.
- [ ] Convert and validate Envy `00000`–`00009`.
- [ ] Match one Blender pose at median trunk error below 2 mm.
- [ ] Convert all 100 Envy + 100 UFO assets.
- [ ] Add `bark_brown_02`, posts, wires, ground, and randomized lighting.

Why cylinders, not capsules: the Blender generator uses finite cylinders.
Capsules change each end by a radius and cannot pass a millimetre depth check.

## Phase 2 — robot and sensors

- [x] Encode the two real mock-pruner ToF offsets without inventing a camera pose.
- [x] Add batched range noise, status, random dropout, and thin-target dropout.
- [ ] Import the pre-flattened UR5e + slider + pruner URDF.
- [ ] Resolve package mesh paths and author `ArticulationCfg`.
- [ ] Fit mouth/failure oriented boxes to the BSD-licensed cutter meshes.
- [ ] Select and document wrist-camera extrinsics.

Hard gate: `camera_offset` remains unset until visibility and jaw-occlusion
experiments justify a pose.

## Phase 3 — task and baselines

- [x] Add batched mouth/failure OBB intersection and perpendicularity gates.
- [x] Add a ground-truth cut-point oracle ordered by radius then neighbourhood.
- [ ] Add all-nearby-wood failure-zone broad phase.
- [ ] Add contact/collision state from PhysX.
- [ ] Port scripted ToF servoing.
- [ ] Configure the CuRobo UR5e oracle.
- [ ] Implement dense reward and curriculum.

Hard gate: do not report a learned policy without scripted and oracle baselines.

## Phase 4/5 — policies and evaluation

- [ ] Flow, ToF, metric-student, and fused observation variants.
- [ ] Five-seed PPO protocol.
- [ ] Continuous robustness ladder and per-axis sensitivity.
- [ ] Held-out Envy `00042` / `00065` and untouched UFO evaluation.
- [ ] PyBullet sim2sim ranking comparison.
- [ ] 30 cm box cross-renderer check.
- [ ] ROS 2 hardware-in-the-loop demo (stretch).

DA2-ft runs once to propose a target, not at every PPO step. A distilled student
is required before metric depth is admitted to the high-rate policy loop.
