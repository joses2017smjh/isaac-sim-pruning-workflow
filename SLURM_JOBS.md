# SLURM job ledger

Last reconciled: **2026-09-05 10:03 PDT** (`America/Los_Angeles`).

This ledger covers jobs produced by this repository's `prune-*` submission
scripts and the two upstream v60 probes explicitly cited by the repository
documentation. Scheduler completion and application-gate success are separate:
a `COMPLETED (0:0)` allocation is not a pass unless its expected evidence says
so. Accounting values below come from `sacct`; live state comes from `squeue`;
gate results come from job-specific JSON. Paths under `logs/` are local and
gitignored, so the tracked evidence is the durable GitHub record.

The existing HPC prose dates job `21036831` to 2026-08-26; Slurm accounting
records its submission/start on 2026-08-24, which is the date used here.

## Current queue

The full user queue at the timestamp above is below. No workflow job remains
queued. Both listed allocations were left untouched; no job was
cancelled, held, reprioritized, or otherwise modified.

| Job | Partition | Name | State | Node or pending reason | This workflow |
|---|---|---|---|---|---|
| `21185770` | `gpu` | `ood-advanced` | `RUNNING` | `cn-gpu6` | Unrelated; untouched |
| `21185936` | `gpu` | `lh-ft` | `RUNNING` | `cn-gpu6` | Unrelated; untouched |

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

Orders are released on application evidence, not merely Slurm state. The
baseline remains unsubmitted because the environment smokes have not passed and
the measured hold/contact issues require investigation. No blind retry, baseline,
or training allocation is queued.

| Order | Intended job | Status and dependency | Required pass evidence |
|---|---|---|---|
| 1 | Fresh pinned import via [`hpc/slurm/import_urdf.sbatch`](hpc/slurm/import_urdf.sbatch) | **Complete: job `21136450` passed and is promoted.** The importer models the nested `_abs` root, the failed output remains quarantined, and the wrapper independently rejects non-green JSON evidence. | [Passing evidence](docs/evidence/urdf_import_21136450.json): `status: complete`, `ok/imported: true`, output hashes, and successful stage validation |
| 2 | Batched A-D/contact/live-ToF smoke via [`hpc/slurm/env_smoke.sbatch`](hpc/slurm/env_smoke.sbatch) | **Blocked by measured hold drift and incomplete contact coverage.** Both sensor grids now return live data. Job `21186027` provides the joint/tool trace needed to diagnose the 20.12 mm drift. | A new job-specific report with `ok: true`, stable hold, complete contact coverage, sensor transforms, and nonzero controlled geometry-response deltas |
| 3 | Scripted/CuRobo baseline smoke via [`hpc/slurm/baselines.sbatch`](hpc/slurm/baselines.sbatch) | **Blocked on order 2.** Do not report a scripted-ToF or CuRobo success rate before the live environment gate passes. | `docs/evidence/baselines_<jobid>.json` with `ok: true`, scripted-ToF success and finite contact; record CuRobo availability honestly |
| 4 | 30 cm camera rectangle via [`hpc/slurm/camera_rect.sbatch`](hpc/slurm/camera_rect.sbatch) | **Blocked on physical camera model/optical-transform selection and renderer configuration.** The CPU `close_lateral` result is only a simulation candidate. | `docs/evidence/camera_rect_<jobid>.json` with `ok: true` and median depth within 5 mm of 0.30 m |

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
