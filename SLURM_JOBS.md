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

### Generalization controls (registered September 23, not yet submitted)

A second frozen batch, `artifacts/generalization/generalization-controls-20260923/`,
re-renders the eight matrix trees under ten single-axis conditions (62 frames
per tree, 38 shared geometry passes) with `hpc/slurm/controls_render_frozen.sbatch`
(array `0-7%1`, 15 min reserved each) and scores them with
`hpc/slurm/controls_eval_frozen.sbatch` (`afterany` on the array, 60 min
reserved): DA2 metric on 688 frames including four test-time photometric
variants derived in the job, the public relative DA2 head on those plus the
pinned matrix (192) and Stage A (600) frames, then six-view DINO on the far-rig
training-camera cells. 180 GPU-minutes reserved, about 40 expected, one GPU at a
time, 2.5 GB preflight estimate. **Submitted September 23** after the user's approval of the 180 GPU-minute
budget and the push: render array **`21403403`** (`0-7%1`) and evaluation
**`21403404`** (`afterany:21403403`), frozen code revision `fbc446d`, storage
preflight passed at 1.597 TB projected (share measured at 1.575 TB) against
the 1.6 TB line. Both PENDING at submission; the user's unrelated `lh-v3`,
`ood-advanced` and `sf04` jobs were left alone. Receipts are in the batch
directory.

**Completed September 23.** Render tasks `21403403_0..7` COMPLETED (0:0) in
2 min 21 s to 2 min 30 s each on cn-gpu5; evaluation `21403404` COMPLETED (0:0)
in 12 min 28 s. 688 of 688 registered frames scored (DA2), 1,480 relative-head
frames, 432 DINO frames with 40 off-rig groups skipped by design; about
32 GPU-minutes used of 180. Baseline cells reproduced the matrix within
0.001 m. [Evidence](docs/evidence/generalization_controls_2026-09-23.json) ·
[protocol result](docs/EVAL_PROTOCOL_GENERALIZATION_CONTROLS_2026-09-23.md#result--september-23-2026).
[Protocol](docs/EVAL_PROTOCOL_GENERALIZATION_CONTROLS_2026-09-23.md).

### Strategies, tree1, renderer control and fine-tunes (submitted September 23)

All submitted after the user's approval of the budgets and of raising the
storage line to 1.7 TB (share measured 1.578 TB; code revision `1c8ef03`).

| Batch | Jobs | Reserved | Notes |
|---|---|---|---|
| `strategy-tool-axis-20260923` (10 tree0 targets, `tool_axis_standoff`) | array `21404500` `0-9%1`, all tasks ended (10–16 min recorded, under 1 min refused), 0/10 | 170 min | [protocol and result](docs/EVAL_PROTOCOL_STRATEGIES_2026-09-23.md#result--september-24-2026) |
| `strategy-horizontal-20260923` (`horizontal_standoff`) | array `21404502` `0-9%1`, all ended, 0/10 | 170 min | |
| `strategy-fine-step-20260923` (`fine_step`, 400 frames) | array `21404503` `0-9%1`, 50-min tasks, 20–30 min recorded, 0/10 | 340 min | |
| `tree1-listed-baseline-20260923` (7 listed tree1 spurs, baseline) | array `21404504` `0-6%1`, all ended, **2/7 pass** (`_1` spur 14944, `_2` spur 15004, both exit 0:0) | 119 min | first recorded passes on a registered target |
| `tree0-replay-20260923` (Cycles at recorded wrist poses, two barks) | render `21404508` COMPLETED 12 min, eval `21404509` COMPLETED 2 min | 50 min | 156/156 scored; [result](docs/EVAL_PROTOCOL_TREE0_REPLAY_2026-09-23.md#result--september-23-2026) |
| `finetune-jitter-20260923` | train `21404511` COMPLETED 2 h 57 min, eval `21404512` FAILED at its third step | 540 min | matrix (192) and controls (688) scored; the Stage A step failed on a plan-format mistake in the launcher, rescored by `21405526` |
| `rescore-finetune-stage-a-20260923` | `21405526` COMPLETED 4 min, both arms' best.pth on Stage A | 20 min | completes the approved evaluation step |
| `finetune-control-20260923` | train `21404513` COMPLETED 2 h 55 min, eval `21404514` FAILED at its third step (same plan-format mistake), Stage A rescored by `21405526` | 540 min | [result](docs/EVAL_PROTOCOL_FINETUNE_2026-09-23.md#result--september-24-2026) |

**First recorded passes.** `21404504_1` (spur 14944) and `21404504_2` (spur
15004), tree1 listed candidates under the baseline strategy and the source
light, passed all 17 grader checks; their neighbour 14884 reached closure and
failed only the post-release drop check. This is the listed-candidate
population, not the seeded register. Slurm exit 0:0 on those two tasks is the
runner reporting the grader's pass, not evidence by itself.

**Disclosed deviation.** The four Isaac arrays were meant to be chained with
`afterany` so that one A40 ran at a time; the submission loop read the wrong
key from the launcher's result and submitted them independently. Each array
still runs one task at a time (`%1`), but up to four Isaac tasks can run
concurrently. Under rule 2 no submitted job is modified; the reserved minutes
are unchanged. The depth chain is serial as intended.

### Perception fixes and the tree1 population (submitted September 27)

Approved by the user on September 27 (all 111 trials, 2,775 GPU-minutes
reserved). Code revision `7853132`; protocol and predictions committed before
submission: [protocol](docs/EVAL_PROTOCOL_PERCEPTION_2026-09-26.md). Share
measured at 1.610 TB (du exit 1 from unreadable cache paths; total kept, 20 GB
reserve applied); the round projects to 1.683 TB against the 1.7 TB line. Every
array runs one 25-minute A40 task at a time; the batches form two `afterany`
chains, so at most two tasks of this round run at once.

| Chain | Batch | Variant | Array job | Trials |
|---|---|---|---|---|
| A | `perc-similarity-tree0-20260926` | `similarity_tracker`, tree0 ten | `21442138` | 10 |
| A | `perc-similarity-tree1-20260926` | `similarity_tracker`, tree1 listed seven | `21442140` | 7 |
| A | `perc-mount-flip-20260926` | `mount_side`, 8353 and 19264 | `21442141` | 2 |
| A | `perc-both-tree0-20260926` | `similarity_mount`, tree0 ten | `21442142` | 10 |
| A | `perc-both-tree1-20260926` | `similarity_mount`, tree1 listed seven | `21442143` | 7 |
| B | `tree1-listed-repeat-r1/r2/r3-20260926` | baseline, three repeats | `21442144`, `21442145`, `21442146` | 21 |
| B | `tree1-listed-morning-20260926` | baseline, morning light | `21442147` | 7 |
| B | `tree1-listed-evening-20260926` | baseline, evening light | `21442151` | 7 |
| B | `tree1-seeded-ten-20260926` | baseline, the tree1 half of the Sept 23 register | `21442152` | 10 |
| B | `tree1-seeded30-20260926` | baseline, new seeded draw of 30 | `21442153` | 30 |

**Outcome (September 28).** All 111 tasks finished (Slurm: 12 COMPLETED, 99
FAILED; the 12 are exactly the graded passes, and Slurm state is not the
grade). Wall time 942 GPU-minutes of 2,775 reserved. All 111 planned runs are
accounted for: 70 graded, 41 refused at startup, none incomplete. Graded,
aggregated per batch and scored:
[result](docs/EVAL_PROTOCOL_PERCEPTION_2026-09-26.md#result--september-28-2026).
Every one of the 70 recorded runs has a GIF, MP4, poster and frame JSON composed
on CPU beside its capture (`<run>/media/`, 280 MB, local only); the 41 startup
refusals recorded no frames and have none.

**Storage correction (September 27).** The share is Lustre project 30762 with a
block quota of 1.5 TiB soft (1,649,267,441,664 B) and 2 TiB hard, grace 4 weeks
2 days. The 1.6 and 1.7 TB lines approved on September 23 were above the soft
quota; the original check had read the user quota, which is unset. This round
was projected against 1.7 TB and will probably cross the soft quota. The user
chose to let it run and free space later, and asked that future batches check
the real quota: `tools/research_storage.py` now refuses at 1.5 TiB (with the
20 GB reserve) and treats 2 TiB as the hard limit.

### Rendered lighting variation in training (submitted September 27)

Approved by the user on September 27 (6,010 GPU-minutes reserved), together
with the storage choice to refuse at the 2 TiB hard limit rather than the
1.5 TiB soft quota. Code revision `715b811`; protocol and eleven predictions
committed before submission:
[protocol](docs/EVAL_PROTOCOL_LIGHTING_TRAINING_2026-09-27.md). Share usage
1.614 TB by `df`; the full render's preflight records that the projection
crosses the soft quota. One `afterany` chain, one GPU at a time:

| Step | Batch | Job | Reserved |
|---|---|---|---|
| Pilot render (lpy_envy_00001, 90 frames + 2 source checks) | `training-lighting-pilot-20260927` | `21442469` | 90 min |
| Full render, 74 trees of 60 frames | `training-lighting-full-20260927` | array `21442470` `0-73%1`, 60-min tasks | 4,440 min |
| Full render, 5 trees of 90 frames | same batch | array `21442471` `0-4%1`, 80-min tasks | 400 min |
| Fine-tune arm A (`rendered_jitter`), manifest built in-job | `finetune-rendered-jitter-20260927` | train `21442472`, eval `21442473` | 540 min |
| Fine-tune arm B (`rendered_control`) | `finetune-rendered-control-20260927` | train `21442474`, eval `21442475` | 540 min |


**Throttle reset (September 28, user-authorized).** Array `21442470` sat for
about a day with reason `JobArrayTaskLimit` and no task running. At the user's
explicit instruction, the one scheduler change made to a queued job in this
project was `scontrol update JobId=21442470 ArrayTaskThrottle=1`, which set the
concurrency limit to the value it was submitted with. The reason changed to
`Priority`. Nothing else was altered.

### Known-map re-oriented approach (submitted September 28)

Approved by the user on September 28 (300 GPU-minutes reserved). Code revision
`3cd1024`; protocol and three predictions committed before submission:
[protocol](docs/EVAL_PROTOCOL_PLANNED_APPROACH_2026-09-27.md). Every result is
labelled *known-map plan, jaw orientation set at the planned pose* and is
reported apart from the unchanged-gate results. Share usage 1.652 TB by
`lfs quota -p 30762` (over the 1.5 TiB soft quota, grace running); each
preflight records `over_soft_quota: true` against the 2 TiB hard line. Strategy
`planned_pose`, source light, 200 frames, one 25-minute A40 task at a time, one
`afterany` chain:

| Batch | Targets | Array job | Trials |
|---|---|---|---|
| `planned-pose-r1-20260928` | 530, 19444 (planned); 14944, 15004 (identity plan) | `21461306` `0-3%1` | 4 |
| `planned-pose-r2-20260928` | same | `21461307` `0-3%1`, after `21461306` | 4 |
| `planned-pose-r3-20260928` | same | `21461308` `0-3%1`, after `21461307` | 4 |

### Resume of the timed-out lighting tree (submitted September 28)

Approved by the user on September 28 (60 GPU-minutes; 50 reserved). Task 8 of
`21442470` (`lpy_envy_00014`, job `21445630`) hit a transient CUDA init failure
on `cn-gpu5`. Cycles fell back to CPU and the task timed out with 9 of 60
frames, all 9 verified. `cn-gpu5` rendered 38 other tasks of the array on GPU.
The renderer now refuses a CPU fallback and gains `--resume`. The resume writes
the 51 missing frames into the same tree directory of
`training-lighting-full-20260927`, after copying the old manifest aside, so the
two queued rendered-lighting fine-tunes pick them up. It refuses to start once
either fine-tune has built its manifest. Code revision `7252be6`.

| Batch | Job | Reserved |
|---|---|---|
| `training-lighting-resume-20260928` (51 frames of `lpy_envy_00014`) | array `21461896` `0-0%1` | 50 min |

### Scheduler changes at the user's request (September 28, late)

The user asked, with `ampere` drained for maintenance until October 1, to let
the pruning jobs take any suitable free GPU. The user chose each option after
the hardware risks were stated:

- **Lighting render concurrency.** `scontrol update ArrayTaskThrottle=4` on
  `21442470` and `21442471`, which were throttled at 1. The tasks are
  independent: each renders its own tree into its own directory with
  node-local scratch.
- **Rendered-lighting fine-tunes.** `21442472` and `21442474` now run on
  `dgxh,ampere` (H100 or A40), with the `a40` constraint removed; the fp32
  trainer's torch build includes sm_90. RTX 8000 is excluded because the 7 h
  runtime cap could cut training below the registered steps. Both now also
  depend on the `lpy_envy_00014` resume `21461896`, so its frames enter first.
- **Planned-approach chain.** `21461306` → `21461307` → `21461308` (A40 only,
  25-minute tasks) never started. It was cancelled and resubmitted to
  `gpu,ampere` with the constraint `a40|rtx8000` and 45-minute tasks, 540
  GPU-minutes reserved (approved). H100 is excluded because it has no RT cores.
  The protocol amendment was committed before submission. Code revision
  `bb7e3b5`.

| Batch | Array job | Placement |
|---|---|---|
| `planned-pose-gpu-r1-20260929` | `21464467` `0-3%1` | gpu,ampere; a40\|rtx8000; 45 min |
| `planned-pose-gpu-r2-20260929` | `21464468` `0-3%1`, after `21464467` | same |
| `planned-pose-gpu-r3-20260929` | `21464469` `0-3%1`, after `21464468` | same |

**Resume outcome (September 29).** `21461896` completed on `cn-gpu6`: it
rendered the 51 missing frames of `lpy_envy_00014` on GPU at 15.5 s per frame,
and 60 of 60 frames now pass the geometry check. The earlier manifest is kept as
`render_manifest.before_resume_21461896.json`.

### Outcomes (September 30)

- **Rendered-lighting render.** `21442470` finished all 74 trees (task 8
  resumed as `21461896`), and `21442471` all 5 trees of 90 frames. The
  `ampere` nodes returned on September 29; with the concurrency limit raised
  to 4, the last tasks ran two at a time on A40.
- **Fine-tunes.** Arm A trained 2 h 41 min on `cn-r-5` (`21442472`) and was
  evaluated on `cn-gpu7` (`21442473`); arm B trained on `cn-r-2` (`21442474`)
  and was evaluated on `cn-gpu7` (`21442475`). All COMPLETED.
  [Result](docs/EVAL_PROTOCOL_LIGHTING_TRAINING_2026-09-27.md#result--september-30-2026).
- **Planned approach.** `21464467` → `21464468` → `21464469` completed: 5 runs
  on A40 and 7 on RTX 8000. Slurm marked the six planned-target runs FAILED;
  the grader has 530 and 19444 at 9/17 and the controls at 17/17. A GIF, MP4,
  poster and frame JSON were composed for all 12 runs (51 MB, local).
  [Result](docs/EVAL_PROTOCOL_PLANNED_APPROACH_2026-09-27.md#result--september-30-2026).

### Jaw-shadow counterfactual (submitted September 30)

Approved by the user on September 30 (540 GPU-minutes reserved). Code revision
`28489db`, which committed the protocol, the target registers and the
`jaw_no_shadow` scene option before submission:
[protocol](docs/EVAL_PROTOCOL_JAW_SHADOW_2026-09-30.md). One `afterany` chain,
one task at a time, on `gpu,ampere` with the constraint `a40|rtx8000` and
45-minute tasks. Share usage at submission was 1.674 TB, over the soft quota
and under the 2 TiB hard limit.

| Batch | Arm | Light, targets | Array job |
|---|---|---|---|
| `jaw-shadow-eve-a-r1-20260930` | A (baseline) | evening, 14944 + 15004 | `21491132` |
| `jaw-shadow-eve-b-r1-20260930` | B (no jaw shadow) | evening, 14944 + 15004 | `21491134` |
| `jaw-shadow-mor-a-r1-20260930` | A | morning, 15004 | `21491136` |
| `jaw-shadow-mor-b-r1-20260930` | B | morning, 15004 | `21491137` |
| `jaw-shadow-eve-a-r2-20260930` | A | evening, 14944 + 15004 | `21491145` |
| `jaw-shadow-eve-b-r2-20260930` | B | evening, 14944 + 15004 | `21491147` |
| `jaw-shadow-mor-a-r2-20260930` | A | morning, 15004 | `21491148` |
| `jaw-shadow-mor-b-r2-20260930` | B | morning, 15004 | `21491149` |

**Jaw-shadow counterfactual outcome (October 1).** All 12 tasks finished, all on
A40. Slurm marked the 6 arm-A tasks FAILED and the 6 arm-B tasks COMPLETED; the
grader agrees: every arm-A run stops at the original appearance frame and every
arm-B run passes 17/17.
[Result](docs/EVAL_PROTOCOL_JAW_SHADOW_2026-09-30.md#result--october-1-2026).

### Render-gap check (submitted October 1)

Approved by the user on October 1 at 180 GPU-minutes reserved. Code revision
`a6e8266`; the protocol was committed at `dc85782` before submission:
[protocol](docs/EVAL_PROTOCOL_RENDER_GAP_2026-09-30.md). Placement `gpu,ampere`.
Both jobs were held by the October 1 08:00–16:00 maintenance reservation at
submission.

| Batch | Jobs | Reserved |
|---|---|---|
| `render-gap-20260930` | render array `21499598` `0-7%1` (15 min each), evaluation `21499599` afterany (60 min) | 180 min |

**Render-gap run (October 1).** All 8 render tasks COMPLETED in 6m19s–7m05s
(RTX 8000, `cn-gpu5`–`cn-gpu7`), and the evaluation `21499599` COMPLETED in
16m26s: about 70 of the 180 reserved GPU-minutes. Slurm states are not
verdicts; G0–G3 are scored by a committed scorer. The published aggregation of
both arms ran as CPU job `21501978` (`share`, 4m),
[result](docs/EVAL_PROTOCOL_RENDER_GAP_2026-09-30.md#result--october-1-2026): G0 and G2 supported, G1
partly supported, G3 refuted.

### Jaw in the camera's view: P0 gate (CPU, October 1)

These are CPU jobs on `share`, run outside the interactive session, which has
only 6 GB. A checker run there had been killed for lack of memory, and the OOM
killer had also taken VS Code processes.

| Job | What | Outcome |
|---|---|---|
| `21501901` | The three replays (flags off, mask, mask + hold) of the 129 recorded runs, from a shared clone at the registration commit `a6e8266` | COMPLETED in 6m43s on `cn-b05`; all three exit 0 |
| `21501940` | Preview of the uncommitted checker (not evidence) | All four checks passed |
| `21501970` | Committed checker `f3442df` on those replays | **P0 passed**: [evidence](docs/evidence/jaw_in_view_p0_replay_2026-10-01.json) |
| `21501971` | Foundation CI steps on a clean clone at `f3442df` (no recorded artifacts, as on GitHub) | COMPLETED: ruff check and format clean; 836 passed, 19 skipped; demo ran |
| `21502003` | Preview of the hardened checker, after its adversarial review (not evidence) | All four checks passed |
| `21502011` | Hardened checker `f16c8ca` on the same replays | **P0 passed**: [recheck evidence](docs/evidence/jaw_in_view_p0_replay_recheck_2026-10-01.json) |

### Depth-aware appearance check, held-out replay (CPU, October 1)

| Job | What | Outcome |
|---|---|---|
| `21502037` | `--regression` (129 runs) and `--heldout` (12 runs), from a shared clone at `ff363c9` | COMPLETED in 3 min on `cn-b05`; base exact on 141 of 141. [H1–H4 supported](docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_HELDOUT_2026-10-01.md#result--october-1-2026) |

### Depth-aware appearance check in closed loop (submitted October 1)

The user approved 450 GPU-minutes on October 1.
- **Protocol** registered at `2938ede`:
  [protocol](docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_CLOSED_LOOP_2026-10-01.md).
- **C0 gate** (CPU job `21502409`, `cn-b11`, 12 min): passed on 141 of 141
  recordings, with a largest float difference of 0.0.
  [Evidence](docs/evidence/depth_loop_c0_2026-10-01.json).
- **CI** on a clean clone at `2938ede` (`21502410`): green.
- **Submission** from `1bd1ea1`, after the C0 evidence and the C1–C6 scorer
  were committed:
  - `afterany`-chained behind jaw-hold C, one task at a time;
  - `gpu,ampere`, constraint `a40|rtx8000`, 45-minute tasks;
  - share usage at submission 1.694 TB, over the soft quota and under the
    2 TiB hard limit.

| Batch | Light | Targets | Strategy | Array job |
|---|---|---|---|---|
| `depth-loop-eve-r1-20261001` | evening | 14944, 15004 | `baseline_depth_appearance` | `21502510` (afterany `21501986`) |
| `depth-loop-mor-r1-20261001` | morning | 15004 | `baseline_depth_appearance` | `21502520` |
| `depth-loop-eve-r2-20261001` | evening | 14944, 15004 | `baseline_depth_appearance` | `21502533` |
| `depth-loop-mor-r2-20261001` | morning | 15004 | `baseline_depth_appearance` | `21502536` |
| `depth-loop-src-20261001` | source | 14944, 15004 | `baseline_depth_appearance` | `21502537` |
| `depth-loop-ctl-19444-20261001` | source | 19444 | `planned_pose_depth_appearance` | `21502538` |
| `depth-loop-ctl-12142-20261001` | source | 12142 | `tool_axis_standoff_depth_appearance` | `21502539` |

Every result carries the label *depth-aware appearance check D_strict + J;
changes the 0.35 appearance gate's rule; simulator depth*. It is never pooled
with unchanged-gate runs.

**Depth closed-loop runs (October 1).** All 10 tasks ran on `cn-gpu5` (RTX
8000) and used 256 of the 450 reserved GPU-minutes. These states reveal each
run's pass or fail: `run_vision_experiment.py` exits 0 only when the capture
completed, the grader passed all 17 checks and the configuration matched. They
were seen before the scorer was finished, and the scorer discloses this. The
grades themselves come only from the grade files.

| Task | Run | Slurm state | Elapsed |
|---|---|---|---|
| `21502510_0` | evening 14944 r1 | FAILED (exit 1) | 23m57s |
| `21502510_1` | evening 15004 r1 | COMPLETED | 26m36s |
| `21502520_0` | morning 15004 r1 | COMPLETED | 26m46s |
| `21502533_0` | evening 14944 r2 | FAILED (exit 1) | 23m50s |
| `21502533_1` | evening 15004 r2 | COMPLETED | 26m38s |
| `21502536_0` | morning 15004 r2 | COMPLETED | 26m58s |
| `21502537_0` | source 14944 | COMPLETED | 27m02s |
| `21502537_1` | source 15004 | COMPLETED | 26m13s |
| `21502538_0` | control 19444 | FAILED (exit 1) | 25m01s |
| `21502539_0` | control 12142 | FAILED (exit 1) | 22m59s |

Grading: CPU job `21505683` (`share`, October 2), `aggregate_eval.py` at
`6920d21`, seven batches, all exit 0. Scored by CPU job `21507189`: C1–C6
supported, [result](docs/EVAL_PROTOCOL_DEPTH_APPEARANCE_CLOSED_LOOP_2026-10-01.md#result--october-2-2026).

### Follow-ups (CPU, October 2)

| Job | What | Outcome |
|---|---|---|
| `21505806` | At `9b0e062`: foundation CI on a clone without recordings; the tests that need recordings; the revised render-gap scorer; the P0 recheck | COMPLETED in 2m54s: ruff clean, 886 passed and 19 skipped, demo ran; 23 passed; [render-gap verdicts](docs/evidence/render_gap_verdicts_2026-10-02.json) unchanged; [P0 recheck](docs/evidence/jaw_in_view_p0_replay_recheck_2026-10-02.json) passed |
| `21506428` | At `0282145`: the depth closed-loop scorer, whose notes still said "a Slurm state is not a grade" | COMPLETED in 41s; its output was never opened and is superseded by `21507189` |
| `21507188` | At `45763d0`: the jaw-in-view scorer (committed blind at `3ac43cb`) on the grades committed unopened at `45763d0` | COMPLETED in 1m46s (`cn-a10`, 0.5 GB): [verdicts](docs/evidence/jaw_in_view_verdicts_2026-10-02.json) |
| `21507189` | At `45763d0`, afterany `21507188`: the depth closed-loop scorer, notes corrected | COMPLETED in 32s: [verdicts](docs/evidence/depth_loop_verdicts_2026-10-02.json), C1–C6 supported |
| `21507190` | At `45763d0`: foundation CI on a clone without recordings | COMPLETED in 1m56s: ruff clean, 950 passed and 19 skipped, demo ran |
| `21516610` | At `1386e63`: a GIF, MP4 and poster for every run of the jaw-in-view and depth closed-loop batches (`tools/compose_batch_media.py`) | COMPLETED in 16m48s: 16 of 16 composed (about 72 MB, local). Four before-and-after GIFs are in the README: 530 with and without the jaw hold, and evening 15004 with and without the depth check |

### Jaw in the camera's view (submitted October 1)

The user approved 270 GPU-minutes on September 30. On October 1, after the gate
waiver was put to them explicitly, they chose "hold + waiver, with controls".
Submitted after P0 passed, from code revision `3c6a211`, which contains the P0
evidence; the [protocol](docs/EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md) was
committed at `a6e8266`.
- Strategy: `planned_pose_jaw_hold`, with `PRUNING_JAW_SELF_MASK=1` and
  `PRUNING_CLOSURE_HOLD=1`.
- Source light, `afterany`-chained batches, one task at a time, on
  `gpu,ampere` with the constraint `a40|rtx8000` and 45-minute tasks.
- Share usage at submission was 1.692 TB, over the soft quota and under the
  2 TiB hard limit.

| Batch | Register | Targets | Array job |
|---|---|---|---|
| `jaw-hold-a-20261001` | [a](docs/evidence/eval_targets_jaw_hold_a_2026-09-30.json) | 530, 19444 | `21501983` |
| `jaw-hold-b-20261001` | [b](docs/evidence/eval_targets_jaw_hold_b_2026-09-30.json) | 530, 14944 | `21501985` (afterany `21501983`) |
| `jaw-hold-c-20261001` | [c](docs/evidence/eval_targets_jaw_hold_c_2026-09-30.json) | 530, 15004 | `21501986` (afterany `21501985`) |

Every result carries the label *known-map plan; jaw self-mask; closure hold
with freshness and frame-reuse checks waived on held frames*. It is reported
apart from every unchanged-gate result and never pooled.

**Jaw-in-view runs (October 1).** All 6 tasks ran on `cn-gpu5` (RTX 8000) and
used 181.5 of the 270 reserved GPU-minutes. As above, these states reveal pass
or fail (exit 0 only on a complete capture, 17/17 and a matching
configuration). They were seen before the scorer was finished, and the scorer
discloses this.

| Task | Target | Slurm state | Elapsed |
|---|---|---|---|
| `21501983_0` | 530 | COMPLETED | 36m28s |
| `21501983_1` | 19444 | FAILED (exit 1) | 24m45s |
| `21501985_0` | 530 | COMPLETED | 33m18s |
| `21501985_1` | 14944 | COMPLETED | 27m13s |
| `21501986_0` | 530 | COMPLETED | 33m23s |
| `21501986_1` | 15004 | COMPLETED | 26m19s |

Grading: CPU job `21505682` (`share`, October 2), `aggregate_eval.py` at
`6920d21`, three batches, all exit 0. Scored by CPU job `21507188`.
- 530 passed 17/17 in 3 of 3 runs.
- P2–P6 and P8 supported; P7 partly supported.
- P1 refuted, by one pixel on the jaw's outline.

[Result](docs/EVAL_PROTOCOL_JAW_IN_VIEW_2026-10-01.md#result--october-2-2026).

### Agreement arm in closed loop (submitted October 3)

The user approved 270 GPU-minutes on October 3, on the condition that the
runs were submitted only after three things passed:
- the design study;
- the C0 gate;
- a blind-reviewed scorer.

All three were in place first:
- **The design study.** Its evidence is in
  [replay](docs/evidence/agreement_design_replay_2026-10-03.json) and
  [shadow](docs/evidence/agreement_design_closure_shadow_2026-10-03.json),
  from CPU jobs `21531565` and `21531637`.
- **The C0 gate.** It passed checks A–G, together with CI, in CPU job
  `21531480`: [evidence](docs/evidence/agreement_c0_2026-10-03.json).
- **The scorer.** `tools/score_agreement_loop.py` was reviewed blind,
  amended, and committed with the
  [protocol](docs/EVAL_PROTOCOL_AGREEMENT_ARM_CLOSED_LOOP_2026-10-03.md) at
  `e3204cf`.

Submitted from `e3204cf`. Each batch is one run, `afterany`-chained, one
task at a time, on `gpu,ampere` with the constraint `a40|rtx8000` and
45-minute tasks. The share's project usage was 1.673 TB (over the 1.5 TiB
soft quota, under the 2 TiB hard limit).

| Batch | Light | Target | Strategy | Array job |
|---|---|---|---|---|
| `agree-eve-r1-20261003` | evening | 14944 | `baseline_depth_agreement` | `21532163` |
| `agree-eve-r2-20261003` | evening | 14944 | `baseline_depth_agreement` | `21532164` (afterany `21532163`) |
| `agree-eve-r3-20261003` | evening | 14944 | `baseline_depth_agreement` | `21532166` (afterany `21532164`) |
| `agree-eve-r4-20261003` | evening | 14944 | `baseline_depth_agreement` | `21532167` (afterany `21532166`) |
| `agree-ctl-19444-20261003` | source | 19444 | `planned_pose_depth_agreement` | `21532169` (afterany `21532167`) |
| `agree-ctl-12142-20261003` | source | 12142 | `tool_axis_standoff_depth_agreement` | `21532170` (afterany `21532169`) |

Every result carries the label *agreement arm of the depth-aware appearance
check (+ J); changes the confidence gate's input; simulator depth*. It is
reported apart from D_strict + J and every unchanged-gate result, and never
pooled. The scorer was committed before submission, so Slurm states, which
reveal pass or fail, cannot shape it.

**Agreement-arm runs (October 3).** All 6 tasks ran on an A40 (`cn-s-1`) and
used 91 of the 270 reserved GPU-minutes: the evening runs took 15m42s–15m58s,
the controls 13m17s and 14m14s. Slurm states: the four evening 14944 runs
COMPLETED and both controls FAILED (exit 1). These states reveal pass or fail;
the scorer was committed before submission.

Grading: CPU job `21544527` (`share`, October 4), `aggregate_eval.py` at
`7498eee`, six batches, all exit 0. The grades were committed unopened at
`25c4da2`. Scored by CPU job `21544534` at `25c4da2`: A1–A5 supported, and
evening 14944 passed 17/17 in 4 of 4,
[result](docs/EVAL_PROTOCOL_AGREEMENT_ARM_CLOSED_LOOP_2026-10-03.md#result--october-4-2026).
Media for all 6 runs (local): CPU job `21544591`.

### Known-map planner rebuild (CPU, October 3–4)

The rebuilt `tools/known_map_planner.py` was committed at `f1372ea`. Its
acceptance test and candidate search run from a clean clone at that commit:
- 12 acceptance parts plus an IK-sensitivity replay (`21544543`–`21544555`);
- the acceptance assembly (`21544556`);
- 24 search shards (`21544557`–`21544580`);
- the search assembly (`21544581`).

All are on `share` with constraint `el9`.

All 38 jobs COMPLETED. The acceptance test passed every item
([evidence](docs/evidence/acceptance_known_map_planner_2026-10-04.json)),
and the search found 34 qualifying targets
([evidence](docs/evidence/search_known_map_planner_2026-10-04.json)).
