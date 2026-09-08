# Real Isaac robot recording

This path renders the imported UR5e and mock pruner with PhysX motion, a
procedural tree fixture, RTX cameras, two live ToF sensors, and contact data.
It is separate from the analytic CPU demo and does not require a trained policy.

## Where the scene and tree come from

**This recording does not load Blender `.ply` files.**
[`RenderEnv._spawn_task_geometry`](../hpc/inner/render_pruning_workflow.py)
authors the trunk, branches, target spur, backdrop tree, pedestal, and ground
directly as USD geometry. The robot is the separately imported UR5e/mock-pruner
USD. RTX renders that assembled stage; PhysX steps the robot in it.

The repository has a separate dataset path:
Blender/L-Py `cylinder_data` JSON (or a full world-coordinate sidecar) →
[cylinder loader](../source/isaaclab_pruning/isaaclab_pruning/geometry/cylinders.py) →
[USD authoring](../source/isaaclab_pruning/isaaclab_pruning/usd/cylinders.py).
The [conversion manifest](evidence/trees_converted_manifest.json) records
100 Envy and 100 UFO trees, but those assets were **not loaded by job `21208215`**.
That conversion reconstructs finite cylinders from metadata; it is not a PLY
mesh import and does not establish identical Blender appearance or materials.

Using a particular Blender-exported tree in the next robot recording remains
a separate integration step: select the tree asset, validate its units and
placement, wire its geometry into sensing/collision, and select a reachable
target on that tree. The current recording proves the robot/sensor/render path,
not that dataset integration.

## Completed inspection recording

[![UR5e inspection with actual cameras and live sensors](demo/isaac_workflow.gif)](https://github.com/joses2017smjh/isaac-sim-pruning-workflow/releases/download/isaac-inspection-2026-09-07/isaac_workflow.mp4)

Job `21208215` completed on RTX 8000 node `cn-gpu6` in 2m44s. It recorded
140 frames at 10 fps: **14 seconds** of physically stepped robot motion.
The original UR5e and mock pruner work; no substitute robot was needed.

Tool-to-target distance went from 342.98 mm to 96.07 mm and back to 343.84 mm.
Maximum displacement was 247.37 mm; return error was 1.65 mm. The report records
`approach_inspect_retreat_no_cut`, with all ten capture checks passing.
Both ToF streams changed; individual rays were valid 40.01% / 42.31% of the time.
Missing rays are displayed as missing, not filled with invented measurements.

[Download the MP4](https://github.com/joses2017smjh/isaac-sim-pruning-workflow/releases/download/isaac-inspection-2026-09-07/isaac_workflow.mp4)
or [open the 1440×960 poster](demo/isaac_workflow.png).
For a presentation, play one **14-second** loop, then pause at **7 seconds**:
overview on the left, wrist RGB/depth on the right, ToF/state/flow below.
The GIF uniformly samples the entire sequence to stay below 1.9 MB; use the
MP4 for all frames. RGB display has a labelled 3×3 median filter to reduce RTX
speckle. Raw camera files, CV inputs, depth, and sensor metrics are unchanged.

Tracked evidence: [render](evidence/render_21208215.json),
[stack/source fingerprints](evidence/render_preflight_21208215.json),
[control smoke](evidence/smoke_21208215.json), and
[media provenance and frame metrics](demo/isaac_workflow.json).
The complete local capture is `artifacts/isaac_render/job_21208215/`;
the local MP4 is `artifacts/isaac_render/job_21208215/media/isaac_workflow.mp4`.

The smoke also passed: six-step hold drift was zero at recorded precision,
and a 5 mm command ended 0.360 mm from its target. Each ToF grid had 64 shared
valid pixels for that controlled-motion comparison. All 20 rigid bodies are
instrumented; the inspection itself recorded no contact. This is not a
long-duration stability test or proof of collision avoidance.

## Preserved earlier outcome: control failure

![Actual Isaac scene and measurements from the failed approach](demo/isaac_control_failure.gif)

Job `21201622` recorded 140 frames at 10 fps: 14 seconds of robot motion.
Rendering and both ToF streams passed their artifact checks, but the robot
did not approach the target. The wrist camera was mostly obstructed and the
raw rendering was noisy. The visible `STOPPED FAILURE` label is intentional.
The GIF samples the full timeline; the full-resolution MP4 keeps all frames.

[Render evidence](evidence/render_21201622.json) and
[runtime/source fingerprints](evidence/render_preflight_21201622.json) are tracked.
Full local capture: `artifacts/isaac_render/job_21201622/`.
Full video: `artifacts/isaac_render/job_21201622/media/isaac_workflow_stopped_failure.mp4`.
Large MP4s and raw sensor arrays stay outside Git.

The recorder later corrected stopped-frame tracking-error telemetry to compare
against the command actually held, not the continuing scheduled trajectory.
Historical reports are unchanged. Do not use that old stopped-frame field as
a controller accuracy metric; the completed inspection did not enter a stop.

The same allocation exposed the original hold failure as contact-constrained:
the mock-pruner body pressed against the floor with roughly 180 N upward force.
Both tool-pose calculations agreed. The successful smoke elevates the robot
mount and its dedicated sensor target by 0.70 m, retaining the 5 mm hold limit.
The controller also uses bounded SVD-based damped least squares and measured
gravity compensation. The original floor-level fixture is not declared fixed
by relabelling its failed evidence.

## Reproduce on the pinned cluster stack

The imported robot USD and BHL v60 installation must already exist. The
[environment lock](ENVIRONMENT.lock.json) pins Isaac Sim, Isaac Lab, Python,
and the container digest; preflight rejects a mismatch before launching Kit.

From the repository root:

```bash
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
slurm_clean sbatch hpc/slurm/render_pruning_workflow.sbatch
```

Each allocation creates a fresh `artifacts/isaac_render/job_<jobid>/` folder.
The wrapper checks the raw images, depth arrays, frame count, and report after
Kit exits. `ok: true` on the rendering report means a valid recording, not a
successful pruning operation; always read `task_outcome` and `metrics` too.

Compose the video after the capture finishes:

```bash
/nfs/hpc/share/$USER/Humanoid_Lite/venv-isaac60/bin/python tools/compose_isaac_workflow.py \
  --input-dir artifacts/isaac_render/job_<jobid> \
  --output-dir artifacts/isaac_render/job_<jobid>/media --display-denoise
```

For CPU-only composition elsewhere, install the package's `[render]` extra.
It includes OpenCV, Pillow, and a bundled ffmpeg encoder. Input must be an
existing Isaac capture; the compositor does not invent camera frames.

## Sensor and task scope

- RGB and metric depth come from RTX; depth is simulator ground truth, not
  inferred depth from a neural network.
- Both 8×8 ToF tables come from live ray casting at reviewed mounting offsets.
  Noise is disabled in these diagnostic captures and this is not a fitted
  hardware noise model.
- Farneback optical flow and brown-pixel segmentation run on recorded wrist
  RGB. They are classical offline computer vision, not the controller's input.
- A simulation-defined wrist camera is not a claim of physical-camera calibration.
- Joint/tool state and contact forces are measured from the simulated robot.
- Scripted inspection does not actuate a blade, sever wood, or establish a
  learned-policy success rate. Those remain separate implementation gates.
- The range stop gate runs once per captured frame (10 Hz), against the
  configured tree meshes. It is a diagnostic guard, not a validated hardware
  safety controller. The scene and target are procedural and known in advance.
