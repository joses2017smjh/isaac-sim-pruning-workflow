# Pruning research audit — September 20, 2026

This checkpoint inspects the current data, saved captures, implementation and
job states, and updates the project documents. It does not execute the new
Envy/UFO learned-depth study. The September 19 submission record remains
historical evidence. No job was submitted, cancelled or modified by this audit.

## Repository and evidence

Primary: `isaac-sim-pruning-workflow`, branch `develop`, inspected commit
`78c01c466d598c4ddd9dadc3b6e4f759ca180847`. The pilot actually executed frozen
revision `b4b4250fae14df0c1f39c70f60cf21960bae90ba`.
Existing untracked studio/exporter/tests/daylight-media work is preserved.
The local companion data/code directory is `../Computer_Vision`; it contains
`TRANSFER.md`, `Dataloader/generate_tree2.py`, training/inference scripts,
manifests and checkpoints. Its root has no usable Git metadata or README, so
its remote identity, branch and commit are **unverified**, not inferred from a
similarly named repository. No companion files were changed.

[Job/capture evidence](evidence/repository_audit_2026-09-20.json) records scheduler
IDs, raw allocation IDs, report hashes, fresh independent grades, target identity,
ToF validity and first tracking loss. [Tree inventory](evidence/tree_inventory_2026-09-20.json)
records all 200 assets and the precise limits of the manifest inspection.

## Completed jobs and result limits

At **2026-09-20 17:00 PDT**, `squeue` contained no `prune-*` jobs. All eight jobs
below are `COMPLETED (0:0)`. The six-task pilot has no remaining dependency;
its original `%1` limit was array concurrency, not a dependency. No follow-up
jobs or Envy/UFO jobs were queued by this audit.

| Job | Raw allocation ID | Light | Tracker | Elapsed | Release (s) | Commands | Final home error (mm) | Grade |
|---|---|---|---|---|---|---|---|---|
| `21360571_0` | `21362298` | source | raw | 00:16:21 | 7.8 | 68 | 0.001140 | 17/17 |
| `21360571_1` | `21363248` | source | clahe | 00:16:25 | 7.8 | 68 | 0.000248 | 17/17 |
| `21360571_2` | `21363331` | morning | raw | 00:16:37 | 7.8 | 68 | 0.000426 | 17/17 |
| `21360571_3` | `21363374` | morning | clahe | 00:16:41 | 7.8 | 68 | 0.001094 | 17/17 |
| `21360571_4` | `21363432` | evening | raw | 00:16:31 | 7.8 | 68 | 0.000484 | 17/17 |
| `21360571_5` | `21360571` | evening | clahe | 00:16:36 | 7.7 | 67 | 0.001101 | 17/17 |
| `21358986` | `21358986` | morning | earlier full run | 00:16:34 | 7.8 | 68 | 0.001195 | 17/17 |
| `21358987` | `21358987` | evening | earlier full run | 00:16:14 | 7.8 | 68 | 0.001048 | 17/17 |

All runs contain 200 frames / 20 seconds, pass the 11 saved capture checks,
and pass a fresh 17-check independent sequence grade. The capture validator
also opens the first/middle/last overview, wrist and depth frames; it does not
remeasure every image. The six saved pilot grades agree with fresh checks;
two floating-point displacement values differ by only approximately
`5.6e-17 m`. Original evidence was not overwritten.

Every run targets **`tree0_SPUR_component_8235`**, with simulator **RTX optical-Z**
depth. Each records approximately **809.492 mm** of rigid-piece fall. Both
original orchard trees participate in the scene, but none of these runs targets
tree1. All six pilot runs retain tracking through release; subsequent tracking
loss is recorded and allowed during home-directed retreat. No failed historical
capture was replaced or relabelled.

Raw and CLAHE each complete all three tested lighting conditions. The evening
CLAHE run releases one capture interval earlier (7.7 versus 7.8 s), which does
not establish a general improvement. There is one trial per condition on one
target. Noon, Envy/UFO variation, learned depth and calibrated outdoor lighting
were not tested by this batch. It cannot answer metric-depth generalization.

## Tree and split inventory

| Family | PLY | Cylinder metadata | Converted USDA present | RGB tree directories in inspected Data layout | Depth tree directories |
|---|---|---|---|---|---|
| Envy | 100 | 100 | 100 | 9 | 100 |
| UFO | 100 | 100 | 100 | 0 | 0 |

Both families cover IDs `00000`–`00099`. Counts come from actual local files,
metadata JSON and existing conversion outputs, not README totals. Render counts
mean directory presence under `Computer_Vision/Data/*/{rgb,depth}/*/<tree>`;
per-view completeness was not checked and other render locations may exist.
Existing converted assets do not prove that a tree was used in an Isaac run.

The inspected top-level `train_manifest.csv` has 9,120 rows and 80 Envy IDs.
Its IDs match `splits.json` (`bark_brown_02`, seed 42), with 20 validation IDs.
Envy `00042` and `00065` are validation IDs in **this split**. No UFO IDs appear
in these two files. Other manifests and actual checkpoint training history are
not yet reconciled. Therefore **no tree is certified unseen by a candidate
checkpoint**, and no Envy→UFO zero-shot claim is justified yet. The machine
inventory contains the full inspected train/validation ID lists.

The local `pretrained/da2_metric_finetuned_vitl.pth` is present (1,341,424,667
bytes), but was not loaded, hashed or evaluated in this documentation audit.
`run_da2_finetune_bark02.sh` references this checkpoint through the
`hpc-share/Computer_Vision` alias. That inference script alone does not establish
how the weights were trained. DINO selection, input reproduction, checkpoint
provenance, training lighting and checkpoint-specific leakage remain open.

## Lighting and learned depth

Isaac already implements deterministic `source`, `morning`, `noon`, `evening`
presets. Source/morning/evening have the single-target results above; noon has
implementation tests but no completed sequence in this batch.

The companion `generate_tree2.py` currently has `LOW_LIGHT_MODE = False` and
`ENABLE_DARK_LIGHT_CAM = False`. Its normal world is overcast RGB
`(0.7, 0.75, 0.82)`, strength `0.5`. The existing low-light path uses world
`(0.10, 0.13, 0.20)`, strength `0.144`, sun Euler `(78°, 0°, 45°)`, energy
`2.903`, color `(1.0, 0.68, 0.35)`, angular size `8°`. The optional dark-camera
world uses strength `0.2903`. The sun docstring still says `1.2`, inconsistent
with executable code. Preserve these distinct existing paths when introducing
presets; they do not establish the lighting used to train any checkpoint.

Offline DA2 versus RTX evaluation, DINO comparison, learned-depth shadow mode
and learned-depth control are **not executed**. The renderer still controls
from RTX depth; the training environment's metric-student observation remains
constant (`1.20`) in `PruningEnv._initialize_observation_buffers`. The PPO CLI
still exits with “PPO runner is not implemented”; `run_baselines.py` checks
CuRobo readiness but does not execute a plan.

## Prioritized next work

| Priority / category | Scientific question and evidence gap | Required work and assets | Expected cost | Success measure |
|---|---|---|---|---|
| P0 research | What did each frozen checkpoint see? Top-level split alone is insufficient. | Map DA2/DINO weights to architecture, hashes, training logs/manifests, bark, cameras and light. | CPU audit; checkpoint I/O; one GPU inference smoke each | Checkpoint-specific exposure table with unresolved fields explicit |
| P0 engineering | Can the pilot fit safely on storage? | Storage refusal/preflight helper; account usage and shared-cache paths; estimate modalities and views | Small CPU task | Projected use below user limits before any render submission |
| P1 research | Does illumination alone change learned depth? | Same held-out geometry/views/textures, explicit Blender presets, frozen DA2, RGB and referenced GT/masks | Two-tree render smoke, then a bounded two-family × four-light pilot | RMSE/MAE/percentiles, coverage, target-local error, outliers and inference latency |
| P1 research | Does one model transfer across tree families? | Checkpoint-linked held-out split; at least four preselected target trees per family; DINO only with reproducible inputs | Inference first; retrain only if provenance demands it | Family/light results with leakage controls and preserved failures |
| P1 research | Can perception transfer into Isaac? | Stage A saved RGB versus RTX optical-Z, then B shadow mode; Stage C only after documented sanity gates | Offline inference, then one short GPU shadow sequence | Target-depth error, temporal stability, latency and unchanged control gates |
| P1 research | Does the controller work on the other original tree? | Explicit tree1 target selection and independently graded run; keep separate from L-Py population | One bounded Isaac smoke/full run after geometry checks | Valid target, recorded release/failure, independent grade |
| P2 research | Do surviving features stay on the intended surface? | Surface-membership/depth-drift analysis, more targets/views/occlusion; causal learned tracker comparator | Saved replay first, bounded GPU comparison later | Coverage plus local metric error and time of tracking loss |
| P2 engineering | Is the local studio export/test contract correct? | Investigate the untracked tree1 transform assertion versus actual mesh bounds | CPU inspection | Correct geometry invariant with original transforms preserved |
| P2 presentation | Can each claim be traced to evidence? | Maintain status, ledger, notices, hashes and eight-clip manifest at every job change | Low | No queued work labelled validated; MP4 linked to measured outcome |
| Later research | Do physical sensing and motion assumptions transfer? | Calibrated camera/ToF data, sensor noise/dropout, wind, collision planning, actuated blade mechanics | Separate GPU/hardware experiments | Measured calibration, collision, stability and physical cut metrics |
| Low priority | Browser polish, PPO sweeps and unrelated projects | Defer until perception/provenance gates answer the main question | Avoid premature compute | No displacement of the family/light experiment |

After the family/light/depth study, prioritize **multiple targets and approach
views with measured feature surface membership**, then a causal learned tracker
comparison. This directly tests the current single-target limitation. PPO,
CuRobo execution and blade mechanics remain independent projects.

## Eight-clip status

[eight_clip_manifest_2026-09-20.json](evidence/eight_clip_manifest_2026-09-20.json)
reserves Envy/UFO × source/morning/noon/evening. Every entry is **planned**, with
unknown tree/checkpoint/job/grade/video fields null. No matching clips were
produced in this audit. The eight completed original-tree0 recordings above
are not a substitute for those eight conditions. Their capture directories
contain no MP4 at audit time; raw images/telemetry remain intact.

## Storage and projection

Measured September 20; `du` figures below are allocated bytes, not file sizes.

| Scope | Measured use |
|---|---|
| Home, quota-sized `df` mount | 23,203,938,304 B (21.61 GiB) of 25 GiB; 3.39 GiB available |
| Home directory walk | 29,092,970,496 B (27.10 GiB); differs from `df`, cause unresolved |
| User HPC share | At least 1,451,756,738,048 B (1.452 TB / 1.320 TiB) |
| Primary repository | 10,887,434,240 B (10.14 GiB), including 8.65 GiB of artifacts |
| Companion `Data` | 143,341,436,928 B (133.50 GiB) |
| Companion `checkpoints` | 743,517,425,664 B (692.45 GiB) |
| Companion `da2_weights` | 1,341,427,712 B (1.25 GiB) |

The share walk encountered three unreadable Kit screenshot directories, so its
value is a lower bound. Using the user's decimal 1.5 TB warning and 2 TB hard
limits leaves at most **48.24 GB to the warning threshold**. Shared-filesystem
free space is not the user's allocation headroom. Home already exceeds the
20 GB caution threshold by either counter; inspected consumers include `.local`
(3.90 GB), `.cache` (1.24 GB) and `.codex` (1.05 GB). No user data was deleted or
moved, and no large downloads/caches were created in home.

This documentation update is budgeted below 1 MiB, projecting known share usage
below 1,451,757,786,624 B. A new render sweep has **no valid storage estimate yet**;
its tree/view/modalities must be fixed and existing RGB/GT availability checked
first. Large generation remains unstarted. All audit temporaries are in `/tmp`.

## Validation and documentation discipline

Tracked pruning scope: **514 passed, 9 USD-dependent skips, 1 simulator test
deselected**. Full current working tree: **545 passed, 1 failed, 9 skipped,
1 deselected**. The existing untracked
`tests/test_studio_orchard.py::test_actual_local_glb_matches_manifest_units_original_tree_separation_and_hashes`
fails its tree1 translation assertion (`0.0 > 0.1`). This audit does not fix or
validate the studio; it preserves that work and reports the failure. Ruff lint
and format pass across 122 maintained source/test/tool/renderer files.

For every later implementation or submission, update `status.txt`,
`SLURM_JOBS.md`, the relevant research document and machine evidence together.
Record exact job IDs/dependencies and timestamped states. Update README claims
only after measurements; update `NOTICE.md` when asset/model provenance or
redistribution scope changes. Keep source notices and old failed evidence.

## Inspect later

Run from the primary repository with the existing Python environment:

```bash
cd /nfs/hpc/share/$USER/isaac-sim-pruning-workflow
squeue -u "$USER" -o "%.18i %.30j %.12T %.45R %.30E"
sacct -j 21358986,21358987,21360571 --allocations \
  -o JobID,JobIDRaw,State,ExitCode,Elapsed,Start,End -P
PY=/nfs/hpc/share/$USER/miniforge3/envs/depth-env/bin/python
for run in artifacts/vision_robustness/lighting-20260919/run_*; do
  "$PY" tools/validate_isaac_render.py "$run"
  "$PY" tools/validate_vision_sequence.py --input-dir "$run"
done
"$PY" -m json.tool docs/evidence/repository_audit_2026-09-20.json
"$PY" -m json.tool docs/evidence/tree_inventory_2026-09-20.json
```

Storage figures and projection are recorded in the audit evidence and current
`status.txt`. No large generation is authorized by a successful scheduler exit;
the research preflights in the supplied study remain required.

## Authorized research execution (2026-09-20)

Implementation and jobs are now in progress; the preceding audit-only snapshot is historical. Frozen DA2 offline evaluation job **21370005** submitted, no dependencies. The one-frame CPU accuracy gate failed; learned control remains conditional. Checkpoint-specific leakage corrections, current job/result states and storage are tracked in the [execution record](RESEARCH_EXECUTION_2026-09-20.md) and `docs/evidence/research_execution_2026-09-20.json`.
