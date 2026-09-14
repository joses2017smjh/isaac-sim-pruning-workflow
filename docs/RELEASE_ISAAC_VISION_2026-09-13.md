# Isaac Sim: live vision approach and preserved tracking failure

Real RTX cameras, PhysX-driven UR5e motion, simulator depth, and two live 8×8
ToF grids. These clips use the original textured Blender orchard's first tree.
They do not demonstrate a completed pruning task.

- `isaac_blender_live_approach.mp4`: 6 seconds, 60 applied vision commands,
  230.08 mm tool displacement. All ten recording checks pass. The run ends
  35.01 mm from the tracked target at the mouth, before closure or retreat.
- `isaac_blender_live_approach_close.mp4`: the same run's actual close camera.
- `isaac_blender_tracking_stop.mp4`: 7.1-second partial/cancelled run.
  At 5.5 seconds, only three features meet the required four-inlier gate.
  Motion stops; no closure or detachment occurs. The original report remains
  incomplete and is not relabelled as a terminal result.

Path tracing and OptiX denoising run inside RTX. No compositor median filter
was applied. Dashboard Farneback flow is an offline diagnostic; the recorded
seeded LK tracker supplied live position feedback to the controller.

JSON assets preserve capture provenance and separate recording validity from
sequence completion. Initial target identity, axis and radius come from scene
metadata. Depth is simulator ground truth. The jaw is a visual proxy and the
cut model is discrete rigid-piece release, not actuated CAD or wood fracture.

Additional two-tree evidence:

- `isaac_two_trees_tracking_failure.mp4`: 20 seconds / 200 frames from job
  `21316823`. Both original trees are visible and participate in collisions
  and ToF ray casting. 63 applied vision commands moved the tool 243.72 mm,
  then tracking stopped at 6.3 seconds. No closure, detachment, or retreat.
  The recording passes its eleven checks; the independent task grade fails.
- `frames_21316823.json.gz`: original captured telemetry, compressed without
  modification. With `render_21316823.json` renamed to `report.json` and the
  decompressed telemetry named `frames.json`, run
  `python tools/validate_vision_sequence.py --input-dir PATH` to reproduce
  the rejected sequence grade.

Job `21317169` tests a fixed virtual camera aligned along the jaw opening.
The current outcome is tracked in `SLURM_JOBS.md`; no completed cut is claimed.
