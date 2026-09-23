# Isaac Sim: two-tree release sequence and recorded failures

Historical release notes. September 20 status: the subsequent daylight pilot
has completed locally; see the [current ledger](../SLURM_JOBS.md) and
[audit](RESEARCH_AUDIT_2026-09-20.md). This documentation update does not add or
publish assets to the September 13 release.

Start with `isaac_two_trees_vision_sequence.mp4`: **20 seconds / 200 frames**,
both original Blender trees, UR5e, live RGB-D tracking, dual ToF and PhysX motion.

Job `21328323` passes all eleven capture checks and all **17 independent task
checks**: 68 applied vision commands, one gated release at **7.8 seconds**,
**809.49 mm** measured piece fall, and **<0.001 mm** final home error.
`isaac_two_trees_vision_sequence_wrist.mp4` shows the actual mounted-camera view.
Pause at 7.8 seconds, then watch the branch fall and the arm return.

This is one known spur in a two-tree scene. The jaw is a visual proxy, release
is a discrete rigid-body event, and depth is simulator ground truth. No physical
blade actuation, wood fracture, trained recognition or multi-tree success rate
is claimed. Tracking loss after completed release is retained in the dashboard;
home-directed retreat no longer needs to follow the falling target.

Failure comparisons:

- `isaac_two_trees_closure_failure.mp4`: job `21317409`, 200 frames.
  Confidence falls to 0.14165 below the 0.15 limit at 7.7 s. Closure stops at
  2/3; no release or return. Fixed timing alone did not complete the task.
- `isaac_two_trees_tracking_failure.mp4`: job `21316823`, 200 frames.
  Tracking stops at 6.3 s after 63 applied commands. No closure or release.
- `isaac_blender_live_approach.mp4` and `_close.mp4`: earlier one-tree,
  six-second approach, 60 commands, no closure or return.
- `isaac_blender_tracking_stop.mp4`: earlier one-tree cancelled partial run,
  7.1 seconds preserved; tracking stops at 5.5 seconds.

Path tracing and OptiX denoising run in RTX. No compositor median filter is
applied. Farneback flow is an offline diagnostic, not a controller input.
The source Blender file and original exported assets remain unchanged.

The public aggregate result is `docs/evidence/two_tree_summary_2026-09-14.json`
in the repository. New full raw telemetry and path-bearing provenance remain
local pending permission to publish. Previously released JSON assets, including
`frames_21316823.json.gz`, are unchanged; they document older runs, not the
successful run's raw input. Do not treat any recording's scheduler completion
as a task pass without its independent sequence grade.

Morning/evening sunlight comparison captures are queued, not included yet.
Browser trajectory replay is proposed, not implemented. See the repository
job ledger and `docs/ROBOT_STUDIO_PLAN.md` for current scope and dependencies.

## Authorized research execution (2026-09-20)

Implementation and jobs are now in progress; the preceding audit-only snapshot is historical. Frozen DA2 offline evaluation job **21370005** submitted, no dependencies. The one-frame CPU accuracy gate failed; learned control remains conditional. Checkpoint-specific leakage corrections, current job/result states and storage are tracked in the [execution record](RESEARCH_EXECUTION_2026-09-20.md) and `docs/evidence/research_execution_2026-09-20.json`.
