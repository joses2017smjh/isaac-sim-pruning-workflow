# Isaac Sim Pruning Workflow

Closed perception-to-motion research workflow for dormant apple-tree pruning in
Isaac Lab.

Jose Sanchez — Oregon State University — <sanchej7@oregonstate.edu>

Anchors:

- [`spur-depth-service`](https://github.com/joses2017smjh/spur-depth-service):
  metric depth and reconstruction.
- [`bhl-robustness-ladder`](https://github.com/joses2017smjh/bhl-robustness-ladder):
  continuous randomization and sim2sim evaluation.
- [`lukestroh/isaaclab-sensor-learning`](https://github.com/lukestroh/isaaclab-sensor-learning):
  inherited Isaac Lab harness, pinned at `5701a77`.

## Current status

This repository contains a tested simulator-independent foundation. It does
**not** yet claim an imported UR5e/pruner, a passing headless Isaac job, a
trained policy, or pruning success numbers.

Implemented:

- OpenCV camera-to-world/world-to-camera conversion that also emits the legacy
  Blender Euler fields expected by current `spur-depth` consumers.
- Planar-z depth conversion and unprojection, including an analytic three-view
  1 m cube contract test.
- Validated L-Py `cylinder_data` and full world-sidecar loaders.
- Direct semantic `UsdGeom.Cylinder` authoring with collision LOD.
- Fixed two-sensor VL53L8CX rig at the real mock-pruner offsets.
- Batched ToF range noise, variance, status, and thin-target dropout.
- Batched mouth/failure oriented-box intersection and perpendicularity gates.
- Ground-truth cut-point oracle ordered by radius, then neighbourhood clutter.
- Immutable source manifest and fetch tool for robot/orchard dependencies.

See [`docs/ROADMAP.md`](docs/ROADMAP.md) for passed and open gates.

## Control architecture

```text
episode start:
  wrist RGB -> spur-depth -> cut point + branch axis

20–50 Hz final approach:
  ToF / flow / metric-student / fused observation
      -> PPO policy -> differential IK -> UR5e joint targets

terminal success:
  target intersects cutter mouth
  AND wood clears failure zone
  AND cutter is perpendicular within the evaluated tolerance
  AND arm/pruner collision is absent
```

DA2-ft is not run at every PPO step. The measured 185 ms fp16 latency on an
RTX 8000 is incompatible with massively parallel PPO. It proposes the target
once per episode; cheap sensors close the loop.

## Install the core package

The core tests do not launch Isaac Sim:

```bash
python -m pip install -e "source/isaaclab_pruning[dev]"
python -m pytest -q
ruff check source/isaaclab_pruning tests tools
ruff format --check source/isaaclab_pruning tests tools
```

Isaac-dependent commands must run through the Python interpreter shipped with
the pinned Isaac Sim/Isaac Lab environment.

## Fetch pinned robot and orchard sources

No moving branches or submodules:

```bash
python tools/fetch_sources.py
python tools/fetch_sources.py --check
```

Checkouts land under ignored `third_party/src/`. Repositories without declared
licenses stay fetch-only; see [`NOTICE.md`](NOTICE.md).

## Convert L-Py cylinders to USD

The Blender generator builds finite cylinders, so this converter writes
`UsdGeom.Cylinder`, not capsules. That distinction matters for the target
2 mm cross-renderer depth check.

```bash
python tools/cyl_to_usd.py \
  /path/to/trees/metadata/lpy_envy_00000_metadata.json \
  generated/trees/lpy_envy_00000.usda
```

Defaults:

- X tilt: `-17.143°`, matching the Blender orchard generator.
- collision: trunk + branch globally.
- spur/nontrunk: visual and semantic, with optional collision within
  `--active-radius-m` of `--active-cut-point X Y Z`.

A full `cylinders_world/{bark}/{tree}.json` can be supplied with
`--world-sidecar`. Do not pass an `ann/*.json` centroid list; it lacks radius,
length, orientation, and part labels.

## Sensor contract

`mock_pruner_vl53l8cx.yaml` keeps the source hardware offsets:

```text
tof0 = ( +0.04685226669, 0.0, 0.14444246761 ) m
tof1 = ( -0.04685226669, 0.0, 0.14444246761 ) m
```

The wrist camera remains disabled with `offset: null`. The source robot
description leaves `camera_offset` empty, so this project will not fabricate an
extrinsic before visibility/occlusion experiments.

## Depth and pose contract

- Use `distance_to_image_plane` / planar optical-axis z-depth.
- Do not substitute `distance_to_camera` (Euclidean range).
- Isaac input pose: camera origin in world + `quat_w_opencv` `(w, x, y, z)`.
- Stored pose: explicit OpenCV `T_wc`.
- Compatibility fields: Blender `camera.location` + `rotation_euler`.

Current `spur-depth-service` reads the compatibility fields and does not yet
prefer `_T_wc`; both are emitted and tested to agree.

## Compute gate

Slurm inventory includes RTX 8000, L40S, A40, H100, and H200 nodes, while the
usual DGX2 nodes are V100. The V100 has no RT cores: reserve it for CUDA
ray-cast training, not RTX rendering.

[`docs/HPC.md`](docs/HPC.md) records the inventory and smoke command. Visibility
in `sinfo` is not proof of partition access. Gate 0 remains open until the
headless job exits zero with a pinned SIF.

## Research comparison

One environment and reward, five seeds per learned variant:

- A — ground-truth flow first, then RAFT flow.
- B — two noisy 8×8 VL53L8CX sensors.
- C — distilled metric-depth student.
- D — variance-weighted ToF + metric fusion.

Required baselines before policy claims: scripted ToF servoing and a
collision-aware CuRobo oracle. Evaluation keeps Envy `00042` and `00065`
aligned with `spur-depth`, then tests untouched UFO trees and PyBullet sim2sim.

## Provenance note

The upstream harness commit is dated 2026-06-30, not 2026-08-27, and its
repository has no declared root license. This fork preserves its history and
does not apply a blanket license to inherited content. See
[`NOTICE.md`](NOTICE.md) and [`third_party/sources.yaml`](third_party/sources.yaml).
