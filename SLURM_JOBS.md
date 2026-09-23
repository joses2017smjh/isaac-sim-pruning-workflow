# SLURM job ledger

Last reconciled: **2026-09-20 17:00 PDT** (`America/Los_Angeles`).
The September 14 queue table is retained as historical evidence.

This ledger covers jobs produced by this repository's `prune-*` submission
scripts and the two upstream v60 probes explicitly cited by the repository
documentation. Scheduler completion and application-gate success are separate:
a `COMPLETED (0:0)` allocation is not a pass unless its expected evidence says
so. Accounting values below come from `sacct`; live state comes from `squeue`;
gate results come from job-specific JSON. Paths under `logs/` are local and
gitignored, so the tracked evidence is the durable GitHub record.

The existing HPC prose dates job `21036831` to 2026-08-26; Slurm accounting
records its submission/start on 2026-08-24, which is the date used here.

## Success-rate sweep submitted — September 23, 2026

Two frozen arrays for the [pre-registered protocol](docs/EVAL_PROTOCOL_2026-09-23.md).
**Both completed September 23. Result: 0 of 40 trials passed** (Wilson 95%
0–0.088). Slurm reports all 40 as FAILED; that is the runner's exit code for a
non-passing grade, not the evidence. The evidence is
[`eval_2026-09-23.json`](docs/evidence/eval_2026-09-23.json).

| Outcome, per array | Source `21400715` | Morning `21400716` |
|---|---|---|
| Passed 17/17 | 0 | 0 |
| Stopped on hazard contact | 3 | 3 |
| Stopped on invalid vision | 3 | 3 |
| Stopped on ToF minimum clearance | 1 | 1 |
| Layout refused, startup contact > 5 N | 3 | 3 |
| Infrastructure: unlisted tree1 target, never presented | 10 | 10 |

The ten infrastructure trials per array are an error in the target register: the
renderer refuses tree1 spurs outside the export's listed candidates, and the
register sampled outside them. They aborted in about a minute each and stay in
the denominator. Details in the
[protocol result](docs/EVAL_PROTOCOL_2026-09-23.md#an-error-in-the-target-register).

Submission record, kept as written at the time:

| Array | Batch | Light | Tracker | Tasks | Declared GPU-minutes | State at submission |
|---|---|---|---|---|---|---|
| `21400715` | `targets-source-20260923` | source | raw | `0-19%1` | 340 | PENDING `(QOSMaxGRESPerUser)` |
| `21400716` | `targets-morning-20260923` | morning | raw | `0-19%1` | 340 | PENDING `(Dependency)` |

Both sweep the same 20 pre-registered spurs, 10 from each original tree, drawn
with seed `20260923` from the 444 components that pass the existing jaw-fit
screen. Frozen code revision `00818fb034dd3a6fec721825cd2e353f1b545a0d`,
404 source files hashed, target register hashed into each plan.

`21400716` carries `--dependency=afterany:21400715` so only one of the two runs
a GPU at a time, keeping each plan's declared `max_concurrent_gpus: 1` true.
`afterany` means a failure in the first array still releases the second, so its
failures stay visible rather than leaving the batch pending forever. **No
existing job was modified, cancelled, held or requeued**; nine unrelated
`lh-cl2-*` jobs were running or queued at submission and were left alone.

Every one of the 20 targets counts in the denominator, including any rejected
for visibility inside its trial or left incomplete. N does not shrink.

## September 20 completed-job reconciliation

At the timestamp above, no `prune-*` jobs remain running or pending. No job was
submitted, cancelled or modified by this audit. All jobs below completed with
`0:0` and now pass fresh capture validation and all 17 sequence checks.

All six pre-registered tasks are accounted for; none is incomplete. Grades below
are recomputed from `report.json` and `frames.json` by the independent grader,
and each fresh grade agrees with the one stored beside the capture.

| Job | Raw allocation ID | Light | Tracker | Grade | Release (s) | Commands | First tracking-loss reason | Tracking loss (s) | Drop (mm) | Final home error (mm) | Min features | Min confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `21360571_0` | `21362298` | source | raw | 17/17 | 7.8 | 68 | optical_flow_failed | 7.9 | 809.49 | 0.001140 | 18 | 0.581 |
| `21360571_1` | `21363248` | source | clahe | 17/17 | 7.8 | 68 | optical_flow_failed | 7.9 | 809.49 | 0.000248 | 21 | 0.621 |
| `21360571_2` | `21363331` | morning | raw | 17/17 | 7.8 | 68 | optical_flow_failed | 7.9 | 809.49 | 0.000426 | **4** | **0.174** |
| `21360571_3` | `21363374` | morning | clahe | 17/17 | 7.8 | 68 | optical_flow_failed | 7.9 | 809.49 | 0.001094 | 7 | 0.495 |
| `21360571_4` | `21363432` | evening | raw | 17/17 | 7.8 | 68 | optical_flow_failed | 7.9 | 809.49 | 0.000484 | 13 | 0.499 |
| `21360571_5` | `21360571` | evening | clahe | 17/17 | 7.7 | 67 | optical_flow_failed | 7.8 | 809.49 | 0.001101 | 15 | 0.403 |
| `21358986` | `21358986` | morning | earlier full run | 17/17 | 7.8 | 68 | optical_flow_failed | 7.9 | 809.49 | 0.001195 | — | — |
| `21358987` | `21358987` | evening | earlier full run | 17/17 | 7.8 | 68 | optical_flow_failed | 7.9 | 809.49 | 0.001048 | — | — |

Elapsed times were 00:16:14 to 00:16:41, mean 16.5 minutes, one A40 per task at
`%1`. The last two rows are **outside the pre-registered array**: they are earlier
full daylight runs that were missing from the ledger, and they are not folded into
the paired comparison.

In every run the first tracking loss occurs on the frame *after* release, as the
detached piece leaves the region of interest during home-directed retreat. That is
the expected post-release behaviour, not a control failure; no run recorded a stop.

Every run uses original `tree0_SPUR_component_8235`, RTX optical-Z depth and
200 frames. Measured piece fall is 809.49 mm in each. There is no demonstrated
CLAHE task-completion advantage and no Envy/UFO or learned-depth result.
The array's raw allocation IDs explain the differing `job_id` values inside
its reports; they are not extra experiments. No pilot dependency remains.

The two margin columns are tracker-internal diagnostics over pre-release tracking
frames, not the protocol's primary metric. They are reported because **morning raw
sits exactly on the tracker's own `min_features = 4` accept floor**: one fewer
surviving feature would have rejected the frame. The task still completed.

Regenerate the whole table's source with one command:

```bash
python tools/summarize_lighting_pilot.py \
  --batch-dir artifacts/vision_robustness/lighting-20260919 \
  --output docs/evidence/lighting_pilot_results_2026-09-23.json
```

Pilot captures: `artifacts/vision_robustness/lighting-20260919/run_*`.
Earlier full daylight captures: `artifacts/isaac_render/job_21358986` and
`job_21358987`. These two runs were absent from the previous ledger.
[Pilot results](docs/evidence/lighting_pilot_results_2026-09-23.json) is the
regenerated machine evidence behind the table above: per-run scheduler rows,
input hashes, fresh grades, tracking profiles and the paired comparison.
[Earlier audit](docs/evidence/repository_audit_2026-09-20.json) preserves the
September 20 scheduler times, hashes, metrics and grades.
[Research follow-up](docs/RESEARCH_AUDIT_2026-09-20.md) is planned, not submitted.

## Historical September 19 experiment checkpoint

Independent revalidation found both daylight probes complete: `21329420`
(morning, A40 `cn-r-4`, 3m25s) and `21329421` (evening, A40 `cn-s-1`, 3m20s),
both `COMPLETED (0:0)` on September 14. All 11 capture checks pass for each
30-frame probe. Neither reaches closure/release/return; both fail the full-task
grade as expected. Regrading `21328323` still passes all 17 sequence checks.

The [research protocol](docs/RESEARCH_EXPERIMENTS_2026-09-19.md) defines six
full 20-second raw/CLAHE trials under source/morning/evening illumination.
The new launcher uses an A40 array with concurrency one, 8 CPUs, 48 GB and
25 minutes per task (maximum 150 GPU-minutes). It freezes committed source,
checks external asset hashes, strips inherited allocation options, and retains
failed independent sequence grades. Existing running, pending and user-held
jobs are preserved. Array **`21360571_[0-5%1]`** was accepted at 16:12 PDT;
`squeue` at 16:13 PDT reports all six tasks **PENDING**, reason `(None)` at that
snapshot. No existing job was cancelled, held/released or otherwise modified.
Source revision: `b4b4250fae14df0c1f39c70f60cf21960bae90ba` (387 frozen files,
25 pinned external files). [Submission summary](docs/evidence/vision_pilot_submission_2026-09-19.json).

| Array task | Lighting | Tracker |
|---|---|---|
| `21360571_0` | source | raw baseline |
| `21360571_1` | source | CLAHE |
| `21360571_2` | morning | raw baseline |
| `21360571_3` | morning | CLAHE |
| `21360571_4` | evening | raw baseline |
| `21360571_5` | evening | CLAHE |

Local plan, source snapshot, original queue snapshot and sbatch receipt:
`artifacts/vision_robustness/lighting-20260919/`. Logs use
`logs/prune-vision-pilot-21360571_<task>.out` within that batch directory.
Each run writes `run_<index>_<light>_<method>/experiment_result.json` and
`sequence_grade.json`. A missing report is incomplete, never a pass. Work stopped
at queue acceptance at that historical checkpoint; GPU results were pending
then and are reconciled above. CPU verification: 514 passed, 9 USD-dependent skips, 1 simulator test
deselected; lint/format and portable demo reproduction pass.

## Historical queue — September 14, 13:51 PDT

The pruning full-sequence job has completed. Two new daylight probes wait on
the per-user GPU quota. Ten unrelated allocations are running; none was modified.
This is a timestamped scheduler snapshot, not an audit of unrelated applications.

| Job | Partition | Name | State | Node or pending reason | This workflow |
|---|---|---|---|---|---|
| `21329420` | `ampere` | `prune-morning` | `PENDING` | `QOSMaxGRESPerUser` | Morning, 30 frames / 3 s; ten-minute limit |
| `21329421` | `ampere` | `prune-evening` | `PENDING` | `QOSMaxGRESPerUser` | Evening, 30 frames / 3 s; ten-minute limit |
| `21328608_3`, `21328607_4` | `ampere` | `marl-terrain` | `RUNNING` | `cn-s-1`, `cn-r-4` | Unrelated |
| `21329202` | `dgx2` | `interactive` | `RUNNING` | `dgx2-5` | Unrelated |
| `21328608_4`, `21328743_0` | `dgxh` | `marl-terrain`, `tier3-arms` | `RUNNING` | `dgxh-1` | Unrelated |
| `21317025_5`, `21317024_6`, `21329167` | `gpu` | `maze-ppo`, `ood-advanced` | `RUNNING` | `cn-gpu5` | Unrelated |
| `21317023_5`, `21328607_3` | `gpu` | `maze-ppo`, `marl-terrain` | `RUNNING` | `cn-gpu7`, `cn-gpu6` | Unrelated |
| `21329392` | GPU partitions | `cloth-isaac-eval` | `PENDING` | `QOSMaxGRESPerUser` | Unrelated |
| `21328607_[5-11]`, `21328608_[5-11%1]`, `21328743_[1-2%1]`, `21317023_6` | GPU partitions | Terrain/arms/maze arrays | `PENDING` | `JobArrayTaskLimit` | Unrelated |
| `21328744_[0-2%1]` | GPU partitions | `tier3-arms` | `PENDING` | `Dependency` | Unrelated |

The previously listed unrelated `21247857` ended `TIMEOUT (0:0)` after
6h00m13s on September 11. No claim is made about its application result.
Earlier queue snapshots are historical; use fresh `squeue` before acting.

## Blender and render-quality integration — September 9–14

The original **two-tree** export has completed rendering. Both trees have
collisions and ToF coverage; the wider overview shows both trees. The latest
validated task sequence is `21328323`; prior tracking/closure failures remain failures. The original
`.blend` is unchanged; [export provenance](docs/evidence/blender_two_tree_export.json)
records distinct `tree0.usdc` and `tree1.usdc` hashes and their shared origin.

| Job | Accounting | Recording versus task result |
|---|---|---|
| `21316823` | A40 `cn-r-5`, 14m00s; `COMPLETED (0:0)` | 200 frames / 20 seconds; all eleven recording checks pass. 63 vision commands, 243.72 mm movement; tracking lost at 6.3 seconds near the jaws. No closure/release/retreat. [Report](docs/evidence/two_tree_summary_2026-09-14.json), [rejected sequence grade](docs/evidence/two_tree_summary_2026-09-14.json), [GIF](docs/demo/isaac_two_trees_tracking_failure.gif). |
| `21317169` | A40 `cn-r-5`, 8m58s; `CANCELLED by 19646` | Gap-aligned camera reached four stable alignment frames and closure. At 7.8 s, floating-point division left closure at 0.9999999999999994; vision failed on the next frame. No release. Cancelled after the latched stop; 111 telemetry records and 114 wrist images preserved. [Partial report](docs/evidence/two_tree_summary_2026-09-14.json), [timer evidence](docs/evidence/two_tree_summary_2026-09-14.json). |
| `21317409` | A40 `cn-r-2`, 13m50s; `COMPLETED (0:0)` | Deadline corrected; 200 frames captured, but tracking confidence fell to 0.14165 at 7.7 s (limit 0.15), with four of 28 original points remaining. 68 applied commands, 262.93 mm movement, closure only 2/3; no release or retreat. [Report](docs/evidence/two_tree_summary_2026-09-14.json), [rejected grade](docs/evidence/two_tree_summary_2026-09-14.json). |
| `21328323` | A40 `cn-s-1`, 15m53s; `COMPLETED (0:0)` | **Full surrogate-release sequence passes.** 200 frames / 20 s; 11 capture and 17 independent sequence checks. 68 applied commands, 262.94 mm movement; one gated release at 7.8 s, 809.49 mm measured drop, <0.001 mm final home error. One known spur in the two-tree scene, not physical cutting. [Aggregate results](docs/evidence/two_tree_summary_2026-09-14.json), [GIF](docs/demo/isaac_two_trees_vision_sequence.gif). |
| `21329420`, `21329421` | A40, 3m25s / 3m20s; `COMPLETED (0:0)` September 14 | Rechecked September 19: morning/evening 30-frame probes pass all 11 capture checks; neither reaches closure, release or return. |

[CPU replay](docs/evidence/two_tree_summary_2026-09-14.json) reproduces
the previous loss at frame 76 without maintenance, and tracks all 200 saved
frames with maintenance. Post-stop images do not prove counterfactual motion.

| Job | Allocation / accounting | Application evidence |
|---|---|---|
| `21222688` | A40 `cn-r-6`, 2m47s; `COMPLETED (0:0)` | [Quality probe](docs/evidence/render_21222688.json): 30 frames, 1280×720 overview; raw RTX images visibly cleaner with PathTracing/OptiX. Still the procedural fixture and scripted inspection. `/rtx/post/aa/op` readback changed from requested 0 to 1; other capture settings matched. Configured 64 samples is not a measured sample count. |
| `21222710` | A40 `cn-r-6`, 5m35s; `FAILED (1:0)` | [First Blender scene](docs/evidence/render_21222710.json): 60 frames, original tree/posts/wires imported. Trellis intersects arm; wrist seed sees housing at 37.24 mm. Zero vision commands, zero detachments; `vision_stopped_failure`. Rendered images exist, but the both-ToF-live check failed. Failed clip preserved. |
| `21224517` | A40 `cn-r-6`, 58s; `FAILED (1:0)` | [90° layout retry](docs/evidence/render_21224517.json): startup contact gate rejected 34,443.24 N before recording. No render or task pass. |
| `21227646` | A40 `cn-r-6`, 57s; `FAILED (1:0)` | [Pose-reader failure](docs/evidence/render_21227646.json): startup contact gate passed, but NumPy physics tensors are unsupported with the GPU pipeline. No camera frames. Corrected to the Torch frontend with explicit CPU transfer for JSON. |
| `21247873` | A40 `cn-s-2`, 5m40s; `COMPLETED (0:0)` | [Live approach](docs/evidence/render_21247873.json): 60 frames / 6 seconds, all ten capture checks pass. 60 physically applied vision commands, 230.08 mm displacement, final tracked-mouth distance 35.01 mm. No recorded stop or contact, but no closure, detachment, or retreat: `vision_approach_incomplete`. [Actual GIF](docs/demo/isaac_blender_live_approach.gif). |
| `21298152` | RTX 8000 `cn-gpu7`, 6m05s; `CANCELLED by 19646` | Deliberately cancelled after measured throughput projected beyond the 25-minute allocation. 35 camera frames and a 31-record checkpoint remain locally. No final capture/task pass. |
| `21300015` | RTX 8000 `cn-gpu7`, 10m53s; `CANCELLED by 19646` | **Task stopped; recording partial.** At frame index 54 (5.5 simulated seconds), three LK roundtrip inliers remained, below the unchanged four-feature gate. 55 vision commands were applied, then motion held; no closure or detachment. Cancelled after the latched stop to avoid further held-frame rendering. 76 wrist PNGs and a 71-record checkpoint remain local. [Partial report](docs/evidence/render_partial_21300015.json) remains `stage: record`, `ok: false`, `task_outcome: not_started`; frame telemetry establishes the stop, not that unfinished label. |

The CPU geometry screen found that rotating the original orchard 180° clears
the starting arm posture, but a post still intersects the later approach.
Job `21247873` selected original source spur `8235` with orchard yaw 150° and
recorded a live vision-driven approach. The longer test uses that same layout.
Colliders and the 5 N startup contact gate remain enabled. A sampled layout
screen is not continuous-path or cutting clearance. These runs do not establish learned perception, physical
blade actuation, or wood fracture. [Recording details](docs/ISAAC_RENDER.md).

## Repository and referenced stack jobs

| Job | Purpose and allocation | Slurm state | Application/gate status | Log | Evidence |
|---|---|---|---|---|---|
| `21036831` (`build60`) | Upstream v60 RTX probe; `ampere`, `cn-r-1`, 1m04s; started 2026-08-24 23:36 | `COMPLETED (0:0)` | **Documented stack pass, but no repo-local raw evidence.** The docs report finite depth and RGB std 24.4. It establishes that v60 can render; it is not a pruning environment pass. | None in this repository | [HPC account](docs/HPC.md#which-isaac-stack-this-cluster-can-actually-render), [stack account](docs/ISAAC_STACK.md#3-the-actual-fix--isaac-sim-60) |
| `21036909_[0-1]` (`rgb60`) | Upstream two-task array on `gpu`, `cn-gpu5`: task 0 was raw job `21036977` (4m12s), task 1 raw job `21036909` (16s); started 2026-08-24 23:49/23:54 | Both `FAILED (1:0)` | **Failed.** The repository documents `ModuleNotFoundError: No module named 'rsl_rl'`. No local artifact supports a stronger diagnosis for each individual task. | None in this repository | [Stack account](docs/ISAAC_STACK.md#4-the-venv-is-not-a-full-rl-install) |
| `21076907` (`prune-isaac-smoke`) | First pruning cube/plane RTX smoke; `ampere`, `cn-r-1` A40, 1m03s; started 2026-08-28 12:20 | `COMPLETED (0:0)` | **Gate failed.** Kit rendered, but report serialization raised `TypeError: Object of type bool is not JSON serializable`; there is no job evidence. Superseded by `21077170`. | `logs/prune-isaac-smoke-21076907.out` (local) | Expected `docs/evidence/isaac_smoke_21076907.json` is absent |
| `21077170` (`prune-isaac-smoke`) | Gate-0 cube/plane/RGB smoke; `ampere`, `cn-r-4` A40, 44s; started 2026-08-28 12:40 | `COMPLETED (0:0)` | **PASS.** Evidence has `ok: true`, all checks true, cube 1.5000 m, plane 2.0000 m, RGB std 37.994, Isaac Sim 6.0.0.1 and Isaac Lab 3.0.0b2. | `logs/prune-isaac-smoke-21077170.out` (local) | [job evidence](docs/evidence/isaac_smoke_21077170.json); local `logs/isaac_smoke.json` is byte-identical |
| `21077217` (`prune-urdf-import`) | Legacy BDS snapshot import; `ampere`, `cn-r-2` A40, 34s; started 2026-08-28 12:45 | `COMPLETED (0:0)` | **Conversion passed, asset is stale/historical only.** `imported: true` and no slider, but ToF/tool transforms disagree with pinned source and the evidence lacks source, transform, generated-URDF, and mesh hashes. Runtime use is blocked by default. | `logs/prune-urdf-import-21077217.out` (local) | [job evidence](docs/evidence/urdf_import_21077217.json); local `logs/urdf_import.json` is byte-identical |
| `21079145` (`prune-env-smoke`) | First batched env/contact/observation smoke; `ampere`, `cn-r-1` A40, 14s; started 2026-08-28 16:45 | `FAILED (1:0)` | **FAIL / incomplete.** Scene creation began, but the inner run produced no report; `bhl_exec` terminated `squashfuse_ll` after a timeout and the wrapper's missing-evidence check failed. This is not an A-D, contact, or live-sensor pass. | `logs/prune-env-smoke-21079145.out` (local) | Expected `docs/evidence/smoke_21079145.json` is absent |
| `21125352` (`prune-urdf-import`) | Pinned, provenance-checked URDF import; `gpu`, `cn-gpu5` Quadro RTX 8000, 1m11s; started 2026-09-01 20:11 | `COMPLETED (0:0)` | **Gate FAILED despite scheduler completion.** Input asset ID and absolute-URDF SHA were verified, but the converter returned a nested `_abs` root instead of the required content-addressed root. Evidence has `status: failed`, `ok: false`, `imported: false`, and `stage_validation: null`. Do not promote this output. | `logs/prune-urdf-import-21125352.out` (local) | [failed import evidence](docs/evidence/urdf_import_21125352.json), [input-generation provenance](docs/evidence/urdf_generation_ur5e_mock_pruner_bdsdfede4c0_ur18e6f603_calib_3941312424972580002_urdf6b02ce9330be.json) |
| `21136450` (`prune-urdf-import`) | Fresh retry with Isaac Lab 3's nested `_abs` layout modeled and an independent wrapper postflight; `gpu`, `cn-gpu7`, 37s; started 2026-09-03 04:29 | `COMPLETED (0:0)` | **PASS.** Evidence is complete/green, inventories and hashes all 11 outputs, opens the composed stage, finds exactly six active UR revolute joints and no slider, and verifies the reviewed camera0/ToF/tool transforms. The root layer SHA-256 is `6ffa65568f85…da82`; this asset is promoted. | `logs/prune-urdf-import-21136450.out` (local) | [successful import evidence](docs/evidence/urdf_import_21136450.json) |
| `21146271` (`prune-env-smoke`) | First live dual-ToF geometry smoke; `ampere`, `cn-s-1` A40, 28s; started 2026-09-03 15:57 | `FAILED (1:0)` | **FAIL / diagnostic incomplete.** Reached environment construction and recorded correct A-D widths, but Kit cleanup occurred before the caught exception was flushed. No sensor/contact/runtime pass is claimed. The retry writes all results before cleanup. | `logs/prune-env-smoke-21146271.out` (local) | [job evidence](docs/evidence/smoke_21146271.json) (`ok: false`, phase `construct`) |
| `21153271` (`prune-env-smoke`) | Diagnostic retry after making pass/failure evidence durable; `ampere`, `cn-s-1` A40, 29s; started 2026-09-03 23:33 | `FAILED (1:0)` | **FAIL, diagnosed.** v60 rejected `{ENV_REGEX_NS}/Robot` because this manual `DirectRLEnv._setup_scene` path bypasses `InteractiveScene` token expansion. The tracked traceback identifies the exact call chain. Robot, contact, ToF, tree, and smoke-target paths are now globally rooted. | `logs/prune-env-smoke-21153271.out` (local) | [job evidence](docs/evidence/smoke_21153271.json) (`ok: false`, phase `construct`, traceback present) |
| `21153411` (`prune-env-smoke`) | Retry with globally rooted `/World/envs/env_.*/...` paths; `ampere`, `cn-s-1` A40, 10s; started 2026-09-03 23:46 | `FAILED (1:0)` | **FAIL, diagnosed.** Robot, contact, and both ray-caster backends registered. Then `SceneEntityCfg.resolve()` read joint names before `DirectRLEnv` started physics and created the articulation `_root_view`. Resolution is now post-super; the robot spawner also enables v60 contact-report APIs. | `logs/prune-env-smoke-21153411.out` (local) | [job evidence](docs/evidence/smoke_21153411.json) (`ok: false`, phase `construct`, traceback present) |
| `21153625` (`prune-env-smoke`) | Post-physics entity resolution and contact activation; `ampere`, `cn-s-1`, 17s; started 2026-09-03 23:54 | `FAILED (1:0)` | **FAIL, diagnosed.** Reset and observation assembly returned widths 150/278/86/86. First step received a raw Warp array where `.clone()` required Torch. Replaced the raw PhysX accessor with the v60 link-origin Jacobian. | `logs/prune-env-smoke-21153625.out` (local) | [job evidence](docs/evidence/smoke_21153625.json) |
| `21185961` (`prune-env-smoke`) | Link-origin Jacobian and explicit xyzw/wxyz boundaries; `gpu`, `cn-gpu6`, 29s; started 2026-09-05 09:43 | `FAILED (1:0)` | **FAIL at hold gate.** Live stepping works, but translation drift exceeded 5 mm. Evidence did not yet include the drift value; diagnostic retry follows. | `logs/prune-env-smoke-21185961.out` (local) | [job evidence](docs/evidence/smoke_21185961.json) |
| `21186027` (`prune-env-smoke`) | Instrumented hold, joint, ToF, and contact trace; `gpu`, `cn-gpu7`, 17s; started 2026-09-05 09:53 | `FAILED (1:0)` | **FAIL, measured.** Hold translation drift 20.12 mm (limit 5 mm), rotation 0.00309 rad. Both 8×8 ToF grids reached frame 2 with 64/64 finite returns. Contact tensor is finite but only `(1,1,3)`; full-arm coverage is unverified. No reset termination occurred. Controlled motion was not reached. | `logs/prune-env-smoke-21186027.out` (local) | [diagnostic evidence](docs/evidence/smoke_21186027.json) |
| `21201586` (`prune-render`) | First full-workflow render; `gpu`, `cn-gpu7`, 3m59s; started 2026-09-06 19:56 | `CANCELLED by 19646 (0:0)` | **FAIL / cancelled deliberately.** Smoke diagnostics rejected Lab 3's `slice(None)` joint selection. The subsequent renderer hung during Replicator warmup, so this attempt was cancelled before a video existed. The retry supports slice selections and uses native simulation render updates. | `logs/prune-render-21201586.out` (local) | [failed smoke evidence](docs/evidence/smoke_21201586.json) |
| `21201622` (`prune-render`) | Native render-update retry; `gpu`, `cn-gpu7`, 1m57s; started 2026-09-06 20:02 | `COMPLETED (0:0)` | **Recording PASS; task STOPPED FAILURE.** Recorded 140 real RTX frames. The robot did not complete the intended approach. Full 20-body contact reporting exposed approximately 180 N of floor force on `mock_pruner__base` during the hold smoke. The failed clip and report remain available. | `logs/prune-render-21201622.out` (local) | [render evidence](docs/evidence/render_21201622.json), [smoke evidence](docs/evidence/smoke_21201622.json), [failed demo](docs/ISAAC_RENDER.md) |
| `21208115.0` / `.1` | Render launch attempts inside existing `ood-advanced` allocation on `cn-gpu7`; started 2026-09-07 21:27 / 21:28 | `FAILED (2:0)` / `FAILED (1:0)`; both 0s | **Launch failures.** First attempt could not resolve `bash`; the second inherited Slurm's empty export environment and lacked `USER`. Correct launch uses absolute `/bin/bash` and `srun --export=ALL`. Neither produced simulator evidence. | Local interactive output | No application report |
| `21208115.2` | Corrected interactive launch, raised fixture, bounded IK; `cn-gpu7`, 2m23s; started 2026-09-07 21:28 | `CANCELLED by 19646 (0:9)` | **Incomplete recording, 61 frames.** Hold and motion measurements passed numerically, but final smoke serialization still called `len()` on a slice, so its saved `ok` remained false. The render was interrupted; accounting does not establish an OOM diagnosis. Both issues are superseded by the detached job below. | `artifacts/isaac_render/alloc_21208115_step2/` (local) | [failed smoke report with passing measurements](docs/evidence/smoke_21208115_step2.json) |
| `21208215` (`prune-render`) | Detached corrected render; `gpu`, `cn-gpu6`, 2m44s; started 2026-09-07 21:34 | `COMPLETED (0:0)` | **Smoke and recording PASS.** 140 frames / 14 s; outcome `approach_inspect_retreat_no_cut`. Maximum measured tool displacement 247.37 mm, closest target distance 96.07 mm, retreat return error 1.65 mm. All ten recording checks passed, including two changing live ToF grids and no physics advancement from extra render updates. This is an inspection demonstration, not pruning or policy success. | `logs/prune-render-21208215.out` (local) | [render evidence](docs/evidence/render_21208215.json), [passing smoke](docs/evidence/smoke_21208215.json), [pinned preflight](docs/evidence/render_preflight_21208215.json), [capture guide](docs/ISAAC_RENDER.md) |

### Hold/contact diagnosis and passing fixture

The earlier 20.12 mm drift was not cleared by relaxing the 5 mm threshold.
Reporting all 20 nested rigid bodies revealed approximately 180 N of upward
floor contact at the mock pruner. The previously reported single body missed
that contact. Job `21208215` places the robot base 0.70 m above the floor and
retains the 5 mm gate, gravity, and 800/40 arm drive gains. The controller now
uses bounded SVD damped least squares and measured gravity compensation.

The passing smoke records zero translation and rotation drift at its recorded
precision, a 0.3597 mm final error after a 5 mm motion command, all 20 expected
contact bodies, and median range changes of 3.961 / 3.800 mm across 64 shared
finite pixels on each ToF grid. A-D observation widths are 150/278/86/86;
those widths do not establish live learned-depth or flow policy inputs.

## Job 21125352 output disposition

Asset ID:
`ur5e_mock_pruner_bdsdfede4c0_ur18e6f603_calib_3941312424972580002_urdf6b02ce9330be`.

- Required root layer, which is absent:
  `artifacts/usd/<asset-id>/<asset-id>.usda`.
- Actual converter root layer:
  `artifacts/usd/<asset-id>/<asset-id>_abs/<asset-id>_abs.usda`.
- The actual root is 1,757 bytes with SHA-256
  `03d80e6999b0fbc9825e947dc412fde1fb6dc6bf898839e422b54278af23a7cc`,
  matching the failed evidence.
- Eleven partial output files are inventoried and hashed in the evidence.
  However, the importer rejected the path before composed-stage inspection;
  units, Z-up, six UR revolute joints, absence of slider prims, and
  base/camera0/ToF/tool transforms therefore did **not** pass the importer gate.
- The failed directory was moved intact (not deleted) to
  `artifacts/usd/quarantine/<asset-id>_job21125352_failed`. It must not be
  selected through `PRUNING_USD` or described as the current robot asset.
- The evidence's embedded `job` fields are null. The job ID/node association is
  established by its job-specific filename, matching log, and Slurm accounting,
  not by those missing fields.

## Next jobs and dependencies

Orders are released on application evidence, not merely Slurm state. Import,
short control smoke, inspection, and the one-target two-tree surrogate-release
demo now pass their respective gates. Full raw new captures remain local
pending publication permission; the public summary includes source hashes.
The short morning/evening captures were revalidated September 19; full lighting
trials are the next experiment. A local uncommitted browser prototype is
preserved separately from this experiment checkpoint.

Baseline and training remain unsubmitted: this single demonstration is not a
scripted-ToF success-rate evaluation, executed CuRobo plan or PPO rollout.

| Order | Intended job | Status and dependency | Required pass evidence |
|---|---|---|---|
| 1 | Fresh pinned import via [`hpc/slurm/import_urdf.sbatch`](hpc/slurm/import_urdf.sbatch) | **Complete: job `21136450` passed and is promoted.** The importer models the nested `_abs` root, the failed output remains quarantined, and the wrapper independently rejects non-green JSON evidence. | [Passing evidence](docs/evidence/urdf_import_21136450.json): `status: complete`, `ok/imported: true`, output hashes, and successful stage validation |
| 2 | A-D/contact/live-ToF smoke via [`hpc/slurm/env_smoke.sbatch`](hpc/slurm/env_smoke.sbatch) | **Complete in render job `21208215`.** The raised fixture passes unchanged hold/motion gates and verifies all contact bodies. This is a one-environment smoke, not batched throughput evidence. | [Passing smoke](docs/evidence/smoke_21208215.json), including live sensor transforms and geometry-response deltas |
| 3 | Scripted/CuRobo baseline smoke via [`hpc/slurm/baselines.sbatch`](hpc/slurm/baselines.sbatch) | **Not submitted.** The previous environment blocker is cleared for the raised fixture. Baseline execution still needs a matching collision-free fixture and actual planner implementation; the current CuRobo path reports readiness only. | `docs/evidence/baselines_<jobid>.json` with `ok: true`, scripted-ToF success and finite contact; record CuRobo availability honestly |
| 4 | 30 cm camera rectangle via [`hpc/slurm/camera_rect.sbatch`](hpc/slurm/camera_rect.sbatch) | **Not submitted.** A simulation-defined wrist camera now renders. Its fixed exterior mount/toe-in is not a calibrated physical camera; the 30 cm geometric depth check remains separate. | `docs/evidence/camera_rect_<jobid>.json` with `ok: true` and median depth within 5 mm of 0.30 m |
| 5 | Robot/environment/sensor inspection render via [`hpc/slurm/render_pruning_workflow.sbatch`](hpc/slurm/render_pruning_workflow.sbatch) | **Complete: `21208215`.** Pinned stack, 140 frames, independent capture validation, and measured approach/retreat. No replacement robot was needed. | [Render report](docs/evidence/render_21208215.json) and [preflight](docs/evidence/render_preflight_21208215.json) |
| 6 | Original Blender orchard with online RGB-D approach | **Approach demonstrated: `21247873`.** Original textures/meshes, seed visibility gate, causal visual tracking, 60 applied commands, live ToF and measured motion. | [Render report](docs/evidence/render_21247873.json), [preflight](docs/evidence/render_preflight_21247873.json), [GIF](docs/demo/isaac_blender_live_approach.gif) |
| 7 | Full vision → surrogate release → piece fall → home return | **Complete for one target: `21328323`, 17/17 independent checks.** Both trees remain in the scene; no multi-target robustness claim. | Independent [`validate_vision_sequence.py`](tools/validate_vision_sequence.py): one gated release, post-event measured fall, home-directed retreat, final home error ≤3 mm, no recorded stop, captured contact ≤5 N |

PPO A-D × five seeds remains downstream of successful orders 1–3 and is not
queued. There is no training submission script in `hpc/slurm/` to list as a
pending job here.

## Interpretation rules

- Trust job-specific evidence over Slurm's terminal state when they disagree.
- Missing required evidence is a failed/incomplete gate, never an implicit pass.
- Preserve `21077217` as proof that the converter could load the legacy
  snapshot, but never use it as proof of current transforms.
- Preserve `21125352` and its partial hashes as failure evidence; do not rename
  its nested root into a pass after the fact.
- Check `squeue` and `sacct` again immediately before any future submission;
  this file is a timestamped audit, not a live scheduler view.

## Authorized research execution (2026-09-20)

Implementation and jobs are now in progress; the preceding audit-only snapshot is historical. Frozen DA2 offline evaluation job **21370005** submitted, no dependencies. The one-frame CPU accuracy gate failed; learned control remains conditional. Checkpoint-specific leakage corrections, current job/result states and storage are tracked in the [execution record](docs/RESEARCH_EXECUTION_2026-09-20.md) and `docs/evidence/research_execution_2026-09-20.json`.

New preflights: **21370018** Envy 00000 and **21370019** UFO 00000 submitted; two 256x144 source/evening frames each, no dependencies. The 2 m plane test confirmed Cycles optical-Z exactly at center and off axis; lighting determinism passed.

Execution update: 21370005 COMPLETED (600 frames; learned-control gates FAIL); 21370018/19 Blender preflights COMPLETED; 21370021 DINO interface preflight COMPLETED (afterok:21370005). CPU suite 518 passed, 10 skips. New jobs: 21370026 short live DA2 shadow + family USD imports, 21370027/28 paired Envy/UFO pilots (24 frames each), all submitted without dependencies. Source SUN/Filmic baseline and fixed 8mm minimum spur-segment rule documented in frozen code; initial failures retained.

Preflight/result update: 21370039 COMPLETED (48 DA2 +48 DINO predictions); 21370026 FAILED before rendering (unsupported profile), 21370040 FAILED before rendering (mesh-only validator on cylinder assets), both retained. Corrected 21370047 RUNNING; both Envy/UFO USD imports now pass in Isaac. H.264 video smoke passed and was visually inspected. Initial one-tree-per-family evening degradation is diagnostic only, pending four-tree matrix.

September 23 reconciliation: **21370047 COMPLETED** (0:0, 5 min 29 s, cn-gpu5);
its live-shadow gates FAILED. Stage A `21370005` and pilot `21370039` results
are published in [`stage_a_depth_2026-09-23.json`](docs/evidence/stage_a_depth_2026-09-23.json)
and [`family_pilot_depth_2026-09-23.json`](docs/evidence/family_pilot_depth_2026-09-23.json).
The eight-tree matrix was **submitted September 23** after the user's GPU and
storage approvals: render array **`21402687`** (`0-5%1`, one Blender Cycles
tree per task) and evaluation **`21402688`** (`afterany:21402687`). Frozen code
revision `ff4707a45c5dd87561bbfd6edf1c756d252c7719`, 488 files hashed, batch
`artifacts/generalization/family-matrix-20260923/`, storage preflight passed at
1.568 TB projected against the 1.6 TB line. Both PENDING at submission (render
array reason `None`; five unrelated `lh-v3-search` tasks were running and were
left alone).

**Completed September 23.** Render tasks `21402687_0..5` COMPLETED (0:0) in
53–62 s each on cn-gpu6/cn-gpu7; evaluation `21402688` COMPLETED (0:0) in
2 min 10 s on cn-gpu7. All 8 registered trees rendered and scored, 192 frames per
model, both checkpoint hashes matching the plan. About 8 GPU-minutes used of the
120 reserved. Result: lighting effect replicated in 8 of 8 trees; UFO worse than
Envy under every light as a larger constant offset; all gates fail.
[Evidence](docs/evidence/family_matrix_depth_2026-09-23.json) ·
[protocol result](docs/EVAL_PROTOCOL_FAMILY_LIGHTING_2026-09-23.md#result--september-23-2026).
The eight labelled clips planned on September 20 were composed on CPU into
`artifacts/generalization/family-matrix-20260923/clips/` (30 MB, local only;
L-Py renders are not redistributed). The
[clip manifest](docs/evidence/eight_clip_manifest_2026-09-20.json) is now marked
executed with tree IDs, checkpoint hashes, job IDs and pooled metrics.
A zero-GPU anchoring analysis on the same saved predictions (no job) is in
[`depth_anchoring_2026-09-23.json`](docs/evidence/depth_anchoring_2026-09-23.json).
