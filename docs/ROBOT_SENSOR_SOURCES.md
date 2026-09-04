# Robot and sensor source audit

Reviewed 2026-08-31. Repositories are recorded at immutable revisions in
[`third_party/sources.yaml`](../third_party/sources.yaml).

## Decision

The UR5e and mock-pruner files do exist. The best current mechanical source is
[`lukestroh/branch_detection_system`](https://github.com/lukestroh/branch_detection_system/tree/dfede4c0f251358ebed7a1f90ff887847c2fbeb0),
not `ag-robot`. It contains the selected mock-pruner mesh, its Xacro, two ToF
frames, a camera frame, UR5e calibration files, and a SolidWorks archive with a
part named `RealSenseMount,prt.SLDPRT`.

That does **not** yet make this repository a reproducible hardware-matched
simulation. Four gaps remain:

1. The URDF imported by job `21077217` is a stale generated snapshot whose ToF
   and tool transforms differ from the pinned Xacro/config.
2. The upstream camera frame has a position, but no identified camera model,
   camera-body mesh, calibrated optical transform, intrinsics, or hand-eye
   evidence.
3. The Isaac environment currently fills the A-D observation buffers with
   constants. It does not instantiate live ToF, flow, or metric-depth sensors.
4. The PyBullet cutter mouth is a useful legacy geometry reference, but it is
   not proof of the present mock-pruner's cut/failure volumes.

There is no reviewed GitHub repository that supplies a single, validated,
current UR5e + mock-pruner + dual-VL53L8CX + calibrated wrist-camera contract.
The reproducible route is to compose the pinned sources below and close the
missing calibration in this project without inventing it.

## Source authority

| Source and pin | Use here | What it does not prove | License scope |
|---|---|---|---|
| [`branch_detection_system@dfede4c`](https://github.com/lukestroh/branch_detection_system/tree/dfede4c0f251358ebed7a1f90ff887847c2fbeb0) | Primary mock-pruner Xacro, selected mesh, source frames, UR5e calibration, ROS controller, VL53L8CX bring-up, CAD archive | Camera model/calibration; that a generated `tmp/robot.urdf` is current | Root `NOASSERTION`; selected description, controller, VL53L8CX bring-up/messages packages carry package-local BSD-3-Clause files. No CAD-specific notice was found. |
| [`Universal_Robots_ROS2_Description@89bbe79`](https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/tree/89bbe795f38a7ab00fb66fe8831dfff79dc99edf) | Upstream UR5e description used by the composition | Jose's individual arm calibration or end-effector geometry | BSD-3-Clause |
| [`pybullet-tree-sim@4d9f838`](https://github.com/OSUrobotics/pybullet-tree-sim/tree/4d9f8384da9ddd3329175cc8ce1f2c7df9720387) | Legacy pruning environment and selected cutter-mouth collision geometry | Current BDS pruner geometry; a runtime failure-zone link | BSD-3-Clause |
| [`final-approach-controller@a3a74b7`](https://github.com/lukestroh/final-approach-controller/tree/a3a74b73d3a3a4c18e178234639c773ca23506d7) | Legacy controller and sensor-generation reference | Current sensors: it declares a D435i and two VL6180 devices | BSD-3-Clause |
| [`follow-the-leader@ca92d31`](https://github.com/OSUrobotics/follow-the-leader/tree/ca92d314a8dd756e50e362ecfca8187f25fc7868) | UR5e + RealSense D435 scanning/launch reference | Mock-pruner camera calibration | Root license not declared |
| [`nucleo-f446re-microros-vl53l8cx-static@0a9ebdf`](https://github.com/lukestroh/nucleo-f446re-microros-vl53l8cx-static/tree/0a9ebdf54f3bd2eb892b3fb8b9ae6ca4b30f1d83) | VL53L8CX firmware evidence: continuous 8x8, 15 Hz, raw millimetres | Dual-sensor mounting or simulator noise calibration | Root license not declared |
| [`ag-robot@60b3bee`](https://github.com/lukestroh/ag-robot/tree/60b3bee2323ff04d404516c6630db3626cc51fe0) | Composition intent: Amiga, slider, UR5e, mock-pruner | Its tracked temporary URDF omits mock-pruner/camera/ToF links and contains host paths | Root `NOASSERTION`; selected packages are package-local BSD-3-Clause |
| [`apple-harvest@7ee995a`](https://github.com/OSUrobotics/apple-harvest/tree/7ee995ab4fc7f6bac0dc44b9a2403fea3c2a3bc3) | Newer D435i Xacro and MoveIt sensor-integration pattern | Pruner geometry or pruner camera transform | Root license not declared |
| [`orchard-slam@9fdbdbe`](https://github.com/lukestroh/orchard-slam/tree/9fdbdbeb00429b5acd5d89ec390c7a901f57dfdd) | Amiga IMU/GPS/lidar and orchard-layout reference | Arm-only v1; mock-pruner sensing | Root `NOASSERTION`; relevant packages have local BSD-3-Clause files |

The inherited Isaac harness commit `5701a774af7b8579269f924689aaf79b9574a53c`
is dated 2026-06-30, not 2026-08-27. Its original
`lukestroh/isaaclab-sensor-learning` URL returned repository-not-found during
this audit. The commit survives in Jose's fork history. The existing
[`lukestroh/sensor-learning`](https://github.com/lukestroh/sensor-learning) is
a different project and is not a replacement. An explicit fetch request for
the fork-history entry now stops with the archive URL instead of retrying the
dead upstream.

The [`OSUrobotics/branch_detection_system`](https://github.com/OSUrobotics/branch_detection_system)
fork resolved to the same reviewed `dfede4c` commit during this audit. The
manifest names Luke Strohbehn's source URL and pins the object ID, so the
selected content does not depend on a moving branch in either fork.

## Mock-pruner files that matter

At `branch_detection_system@dfede4c`:

- `branch_detection_system_description/config/mock_pruner.yaml` declares the
  two ToF translations.
- `branch_detection_system_description/urdf/end_effectors/mock_pruner/macro/mock_pruner_macro.urdf.xacro`
  defines `mock_pruner__base`, `camera0`, `tof0`, `tof1`, and `tool0`.
- `branch_detection_system_description/meshes/end_effectors/mock_pruner/MockPruner.STL`
  is the mesh selected for both visual and collision geometry. Versioned and
  historical meshes in the same directory are not selected by that Xacro.
- `branch_detection_system_description/solidworks/shortened_pruner_v2.zip`
  contains the assembly/CAD reference, including `RealSenseMount,prt.SLDPRT`,
  `TOFbracket.SLDPRT`, dovetail parts, and the UR5 attachment plate. The CAD
  archive predates the selected mesh's latest source update, so it is a design
  reference rather than proof that every runtime dimension matches.
- `branch_detection_system_description/config/{robot_calibration,cindy_ur5e_calibration}.yaml`
  contains two different UR calibration hashes. Select the physical arm
  deliberately; do not average or silently choose one.

All reviewed mock-pruner sensor/tool translations are children of
`mock_pruner__base`. `mock_pruner__tool0` is the control EEF, not the parent
frame in which the YAML ToF offsets are expressed. The local rig file now
records both roles explicitly.

## Imported transform mismatch

The pinned current source and the URDF used for job `21077217` disagree:

| Fixed child, parent `mock_pruner__base` | Pinned Xacro/config | Imported/generated snapshot |
|---|---:|---:|
| `mock_pruner__tof0` | `[0.04685226669, 0, 0.14444246761]` | `[0.04891, 0.005, 0.14237]` |
| `mock_pruner__tof1` | `[-0.04685226669, 0, 0.14444246761]` | `[-0.04891, 0.005, 0.14237]` |
| `mock_pruner__camera0` | `[-0.0017977, -0.0715747, 0.0711646]` | same translation |
| `mock_pruner__tool0` | `[0, 0, 0.1601525]` | `[0, 0.0050825, 0.1601525]` |

Both `branch_detection_system_description/urdf/tmp/robot.urdf` and
[`artifacts/urdf/ur5e_pruner_abs.urdf`](../artifacts/urdf/ur5e_pruner_abs.urdf)
carry the snapshot values. Job `21077217` proves that this rewritten snapshot
can be imported and articulated. Its JSON does not record the BDS revision,
fixed-joint table, generated-URDF hash, or mesh hashes, so it cannot prove that
the pinned hardware frames were reproduced.

Do not hand-edit those four transforms. Generate a fresh URDF from the pinned
Xacro/config and the selected arm calibration, rewrite only package mesh paths,
then import that generated artifact.

The articulation builder now refuses this default asset while its config says
`reimport_required: true`. `PRUNING_ALLOW_STALE_USD=1` exists only so importer
diagnostics can inspect the historical asset. A fresh explicit `PRUNING_USD`
can be tested before its evidence-backed path is promoted into the config.

## Camera: known frame, unknown calibrated sensor

The earlier conclusion “`camera_offset` is empty, therefore no camera has been
mounted” was too strong.

Known facts:

- `mock_pruner.yaml` has `camera_offset: ""`.
- The Xacro accepts that argument but does not use it. Instead it hard-codes
  `mock_pruner__camera0` at `[-0.0017977, -0.0715747, 0.0711646]` from
  `mock_pruner__base`, with zero RPY.
- That camera link existed before the latest mock-pruner mesh update.
- The SolidWorks archive contains a RealSense-named mount.

Still unknown:

- exact RealSense model and physical revision;
- camera-body/mount geometry selected by the runtime robot description;
- ROS optical-frame rotation and the correct image forward axis;
- intrinsics/distortion and depth/RGB alignment mode;
- a measured base-to-camera or hand-eye calibration;
- whether the CAD assembly and current physical mock-pruner share a revision.

The local `close_lateral` offset `[0, -0.06, 0.10]` is measured from
`mock_pruner__tool0` and won a tree visibility ray-cast. It is a useful
simulation-camera candidate, not the physical `camera0` transform. It must not
replace the upstream frame or be described as CAD measured. Wrist RGB remains
disabled until the physical camera contract is identified and the chosen
simulation view passes the renderer/30 cm checks.

`follow-the-leader` uses a D435 on a UR5e and is valuable for launch and optical
frame conventions. Its real launch composes a tool-to-mount translation and a
separate mount-to-camera transform. Those values belong to a different tool and
must not be copied onto the mock-pruner. `apple-harvest` is likewise an
integration pattern for a harvesting gripper, not a calibration source.

## ToF generations and current evidence

Do not merge sensor facts from different hardware generations:

| Source generation | Declared sensors | Use |
|---|---|---|
| Current reviewed BDS description | two `mock_pruner__tof*` frames; VL53L8CX bring-up/messages present | Primary v1 geometry and message reference |
| Standalone final-approach controller | RealSense D435i + two VL6180 | Legacy control reference only |
| Older Arduino pruning controller | two VL53L0X + LSM9DS0 IMU | Historical only |

The BDS VL53L8CX config declares 8x8 zones, 65 degree diagonal FoV, and
0.03-3.4 m clipping. The reviewed Nucleo firmware sets continuous 8x8 operation
at 15 Hz and publishes raw millimetre arrays. A controller may run faster while
reusing a sample, but the sensor stream itself must not be documented as
20-50 Hz without new firmware evidence.

The BDS filter/covariance files can seed a noise study, not finish one. The bag
names indicate a 10 cm campaign and the covariance JSON distinguishes black and
white targets; neither establishes a range-wide material model. The filter also
hard-codes a single `vl53l8cx_0` frame and contains a validity condition that
should be checked against real bags before porting. Calibrate two-sensor bias,
dropout, thin-wood response, and cross-sensor correlation from recorded data.

## What Isaac currently simulates

The A-D builders and width contracts exist, but the active environment does
not yet feed them from scene sensors. In
`isaaclab_pruning/sim/pruning_env.py`, `_fill_observation_buffers()` creates:

- zero optical flow;
- ToF tables fixed at 0.40 m and 0.42 m;
- metric depth fixed at 1.20 m;
- fixed validity and variance tables.

`apply_tof_observation()` can replace ToF buffers if something calls it, but no
live `CameraCfg`, ray caster, or VL53L8CX sensor is instantiated in
`PruningEnvCfg` or `_setup_scene()`. No live flow or metric-student producer is
connected either. Consequently, shape checks and `not allclose(C, D)` can pass
while the robot moves through an unchanged synthetic observation.

The scripted baseline now shifts its reviewed base-frame points by the pinned
base-to-tool translation, with a regression test at the 8 cm standoff. That
closes the constant-source frame arithmetic, not the live integration: bind the
transform to the freshly generated asset and mount rays on the actual
`mock_pruner__tof0`/`tof1` links before treating a run as hardware matched.

## Cutter geometry boundary

The selected PyBullet URDF references `cutter-mouth-collision.stl`. The
repository also contains decimated cutter and failure-zone assets, but the
selected URDF does not attach a failure-zone mesh as a runtime link. The local
fitted failure AABB is therefore a documented task heuristic derived from a
legacy repository asset. It is not a measured present-hardware exclusion volume.

Use the BDS selected `MockPruner.STL` for collision/occlusion of the current
tool. Keep the mouth/failure proxy versioned as an experiment contract until it
is registered against current CAD or measured hardware.

## Repositories reviewed but not adopted

- [`OSUrobotics/pruning_interface`](https://github.com/OSUrobotics/pruning_interface)
  contributes tree/interface assets but no complete UR5e + pruner sensor model.
- [`OSUrobotics/robot_arm_calibration`](https://github.com/OSUrobotics/robot_arm_calibration)
  is a general/older arm-calibration reference; it does not identify the
  current mock-pruner camera contract.
- `applevision_moveit_config`, `PickApp`, and older apple-picking repositories
  target harvesting/picking hardware rather than this pruner.
- `lukestroh/sensor-learning` is a small tree-rendering package, not the deleted
  Isaac Lab harness and not a drop-in source recovery.

These may answer adjacent integration questions, but adding their transforms
would mix hardware generations rather than complete the reviewed v1 robot.

## Required implementation order

1. Preserve the harness commit in Jose's fork and keep every external source at
   the immutable revision in the manifest.
2. Select the correct UR5e calibration file. Generate the complete UR5e +
   mock-pruner URDF from pinned source; do not promote a tracked `tmp/*.urdf`.
3. Reimport to a new USD path. Evidence must contain source revision, calibration
   hash, generated-URDF SHA-256, selected mesh SHA-256 values, and the four fixed
   transforms above.
4. Instantiate two live 8x8 ray sensors on `mock_pruner__tof0` and
   `mock_pruner__tof1`, at 15 Hz sensor cadence. Assert their readings change
   when a branch or EEF pose moves and match an analytic plane before adding
   noise.
5. Update the scripted baseline to consume live sensor-frame points and source
   its tested base-to-tool transform from the regenerated asset.
6. Identify the physical camera model/revision from the actual assembly or its
   maintainer. Add body/optical frames and calibration evidence. Only then
   compare it with `close_lateral` and renderer-validate the chosen simulation
   view.
7. Connect live flow and metric-student observations. Prove A-D differ because
   of scene response, not constants.
8. Run env smoke and the scripted/CuRobo baselines. Only after those JSONs are
   green should PPO start.

Minimum live-sensor evidence should report sensor prim paths, parent links,
world transforms, update cadence, units, valid fractions, min/median/max range,
and a before/after geometry-response delta. An observation-width assertion is
necessary, but is not sufficient.

## Licensing rule

Repository-level `NOASSERTION` and package-local BSD-3-Clause are different
scopes. A local package license covers that package; it does not automatically
license sibling packages or a CAD archive. Keep unlicensed reference sources
as `reference_only` or `fetch_only`, preserve their notices when permitted
files are redistributed, and obtain explicit CAD permission before vendoring
or publishing derived mesh files.
