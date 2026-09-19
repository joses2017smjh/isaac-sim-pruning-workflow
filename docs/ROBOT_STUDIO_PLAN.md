# Pruning Replay Studio — original proposal

September 19 status note: a local uncommitted `studio/` prototype and exporters
now exist. This experiment checkpoint preserves that work; it does not audit or
publish it. Statements below about absent implementation describe the original
proposal, not the current working-directory inventory.

The supplied Robot Studio prompt assumes a trained floating-base robot and a
MuJoCo evaluation harness. This repository does not have those prerequisites.
The useful first product is a static replay studio for the actual Isaac runs.
No browser app, WASM physics, or policy inference is implemented by this proposal.

## What the code and assets contain

- The promoted robot is a fixed-base, six-revolute-joint UR5e with a mock pruner.
  Joint order: `ur5e__shoulder_pan_joint`, `ur5e__shoulder_lift_joint`,
  `ur5e__elbow_joint`, `ur5e__wrist_1_joint`, `ur5e__wrist_2_joint`,
  `ur5e__wrist_3_joint`. The real-machine slider is not spawned.
- The generated URDF and composed USD exist locally. USD selects its `physx`
  variant. An importer-authored `mujoco` USD variant is not a validated MJCF
  model or a MuJoCo parity result. Mesh redistribution rights remain a gate.
- [Robot config](../source/isaaclab_pruning/isaaclab_pruning/config/robot/ur5e_pruner.yaml)
  defines seven action values: absolute root-frame tool position plus `wxyz`
  quaternion. They are not six normalized joint targets or torques. The renderer
  bounds Cartesian steps to 4 mm, then uses differential IK and joint drives
  with stiffness 800, damping 40 and gravity compensation.
- The environment uses 120 Hz physics and decimation 2. The recorded demonstration
  updates its camera/control decision every 0.1 s and saves 10 Hz frames.
  This is not a 50 Hz trained-policy rollout or a per-physics-step dataset.
- [Observation assembly](../source/isaaclab_pruning/isaaclab_pruning/policies/observations.py)
  concatenates goal XYZ, six joint positions, six velocities and seven EEF-pose
  values, then the variant's sensor data. Widths are A/B/C/D = 150/278/86/86.
  B appends ToF0, ToF1, validity0, validity1 as four flattened 8×8 arrays.
  Training flow remains zero and its metric-depth buffer is a placeholder.
  The demo's live seeded RGB-D controller is a separate execution path.
- [Training CLI](../tools/train.py) refuses to train: PPO is not implemented.
  No project actor checkpoint or ONNX export was found. Do not invent an actor,
  normalizer, reward trace, torque trace or policy comparison from these recordings.
- Captures contain measured joint positions/velocities, tool commands, poses,
  wrist calibration, ToF, contact samples, RGB/depth, release state and timestamps.
  The source-frame index establishes which prior image supplied a command.

## Proposed layout

```text
studio/
  src/config/robot.config.ts       joint order, frames, camera and asset manifest
  src/config/scene.ts              SceneConfig and URL validation
  src/replay/                     binary decoder, synchronized timeline
  src/viewer/                     robot/scene, cameras, sensor panels
  public/assets/manifest.json     fetch-on-demand assets; no embedded CAD
  NOTES.md                        browser, frame and licensing gotchas
tools/export_studio_episode.py    read-only exporter from Isaac captures
tests/test_studio_export.py       schema, timing and quaternion checks
```

Use TypeScript/Vite/React/Three.js if approved. First verify browser-loadable
assets and rights; the original USD package is not directly a browser GLB.
Do not represent a simplified mesh or a replay as Isaac physics in the browser.

## Proposed contracts

`SceneConfig`: schema version, robot/asset version, episode IDs, camera preset,
playback speed, loop, comparison mode, lighting preset, seed and current frame.
Validate all values. Serialize in a canonical key order for URL-hash round trips.
Browser relighting changes only the viewer; recorded wrist images and original
sensor readings retain their capture illumination and calibration.

`manifest.json`: schema version, robot ID, controller ID (not a policy ID),
joint order, units, `world_up: Z`, `quaternion_order: wxyz`, variable-timestamp
support, per-episode source hashes, scene/light preset, outcome, channels and
missing-channel declarations. Preserve failed and cancelled episodes as such.

`episode_NNN.bin`: little-endian Float32 rows with explicit layout:
`[t, base_pos(3), base_quat_wxyz(4), qpos(6), qvel(6), command_pose_root_wxyz(7)]`.
Do not substitute six actions for the seven-value Cartesian command. A sidecar
stores source-frame indices, cut events, sensor samples, camera calibration and
image/video references. Missing data stays missing. Binary export precision must
be tested against the original JSON; never fabricate torque or reward channels.

## Milestones and acceptance

1. **Asset and scene viewer.** Load the licensed robot and two trees; free orbit,
   front/side/top/follow views, mounted-camera viewport, measured FPS HUD.
   A labelled sine-wave joint preview tests rigging only. Acceptance requires a
   real browser run and joint/frame checks; 60 fps is a target, not a claim.
2. **Recorded replay.** Export the successful sequence and closure failure;
   shared timeline, play/pause, frame step, speed, synchronized ToF/contact/tool
   panels, and A/B comparison. Test frame count, timestamps, quaternion order,
   event alignment and source hashes. Compare controllers/runs, not two policies.
3. **Configuration and deterministic prompts.** Support the actual vocabulary:
   `ur5e`, `two trees`, `success`, `closure failure`, `morning`, `noon`, `evening`,
   camera names and `seed N`. Display ignored tokens. Test canonical config and
   URL round trips. Changing the seed must not imply a new simulated episode.
4. **Dataset export, separately gated.** Start with this documented replay schema.
   Claim LeRobot compatibility only after version-pinned loader round-trip tests.
   Existing 10 Hz captures cannot recover actions/observations for unsaved physics
   steps. A future recorder must log pre-step action ordering explicitly.

Defer original M3/M4: first create and validate a desktop MuJoCo model, establish
parity against its own browser port, and obtain an executed baseline/trained
actor. Desktop-MuJoCo versus browser-MuJoCo parity would still not establish
PhysX/MuJoCo parity. WASD locomotion and push recovery do not fit this fixed-base
arm demo. No backend, API keys, telemetry, or LLM service is needed for replay.

Following the supplied prompt, implementation stops at this proposal until Jose
approves the replay-first scope and schemas. After approval, build and verify
one milestone at a time, then show its working artifact before continuing.
