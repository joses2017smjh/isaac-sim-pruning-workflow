# Research execution — September 20, 2026

The user authorized implementation and job submission after the documentation audit. This execution record supersedes the audit-only status where later measurements exist; the audit remains historical evidence.

## Submitted

- **21370005**, `prune-depth-a`: frozen DA2, 600 saved Isaac wrist RGB/RTX optical-Z pairs from original-tree0 source/morning/evening raw runs. No dependencies. Exclusive output `artifacts/generalization/phase-20260920/stage_a_job_21370005`; logs under the batch `logs/`. Frozen code/manifests and SHA-256 pins live in that batch.
- No learned-control job is authorized by a successful inference alone. Target local error <=20 mm, mean relative error <=10%, valid coverage >=99%, and inference p95 <=100 ms were registered before inference; both offline and live shadow stages must pass.

## Checkpoint exposure correction

See [checkpoint evidence](evidence/checkpoint_provenance_2026-09-20.json). The selected DA2 checkpoint trained on 80 Envy IDs and selected on 20 Envy validation IDs. **No Envy tree among the 100 is untouched by training/model selection.** Envy 00050 was incorrectly called held out in an older local evaluation; it is in this checkpoint's training manifest. DINO validation trees 00042 and 00065 are also in DA2 training. Thus a combined-pipeline unseen-tree claim is unsupported.

The fixed Envy pilot is 00000, 00003, 00008, 00012 (DA2 validation trees, absent from selected DINO manifests); UFO is 00000–00003. UFO is absent from the inspected task fine-tuning manifests. Backbone pretraining exposure is unknown. These choices are geometric/provenance choices, made before observing model outcomes, and will not be replaced to hide failures.

## Preflight

Frozen DA2 loaded strictly by checkpoint hash and completed one CPU frame after disabling the CUDA-only xFormers attention path on CPU. Both the initial backend failure and successful retry are retained under the batch `preflight/`. Successful inference is not successful accuracy: target patch MAE **461.96 mm**, p95 **571.61 mm** on this one source frame; this is insufficient for learned control and not an aggregate result.

Storage preflight: share lower bound 1,451,922,767,872 B; estimated additional output 15 GB; reserved 20 GB for the three unreadable pre-existing Kit screenshot directories/uncertainty. Projected total including reserve 1,486,922,767,872 B, below 1.5 TB warning. Home 23,206,559,744 B used of 26,843,545,600 B; all heavy outputs/cache remain off home. No existing data deleted or moved.

## Remaining execution

Paired Blender presets/render harness, DINO inference, family asset/render/import preflights, live RTX-controlled learned-depth shadow, original tree1 targeting, fixed multi-tree matrix, labeled eight-condition clips, and aggregate analysis. Closed-loop learned-depth runs remain conditional on both gates; failed gates will be reported rather than bypassed.

## Inspect

```bash
squeue -j 21370005
sacct -j 21370005 --format=JobID,JobName,State,ExitCode,Elapsed,NodeList
cat artifacts/generalization/phase-20260920/stage_a_job_21370005/evaluation.json
```

New preflights: **21370018** Envy 00000 and **21370019** UFO 00000 submitted; two 256x144 source/evening frames each, no dependencies. The 2 m plane test confirmed Cycles optical-Z exactly at center and off axis; lighting determinism passed.

Execution update: 21370005 COMPLETED (600 frames; learned-control gates FAIL); 21370018/19 Blender preflights COMPLETED; 21370021 DINO interface preflight COMPLETED (afterok:21370005). CPU suite 518 passed, 10 skips. New jobs: 21370026 short live DA2 shadow + family USD imports, 21370027/28 paired Envy/UFO pilots (24 frames each), all submitted without dependencies. Source SUN/Filmic baseline and fixed 8mm minimum spur-segment rule documented in frozen code; initial failures retained.

Preflight/result update: 21370039 COMPLETED (48 DA2 +48 DINO predictions); 21370026 FAILED before rendering (unsupported profile), 21370040 FAILED before rendering (mesh-only validator on cylinder assets), both retained. Corrected 21370047 RUNNING; both Envy/UFO USD imports now pass in Isaac. H.264 video smoke passed and was visually inspected. Initial one-tree-per-family evening degradation is diagnostic only, pending four-tree matrix.

## Results written up — September 23, 2026

The runs above completed on September 20 but were never written up. Their
numbers are now regenerated from disk by `tools/aggregate_family_eval.py`
through `sql/family/*.sql`, and every figure below traces to an evidence file.

- **21370047 (live shadow) COMPLETED**, 5 min 29 s: 30 approach-only frames on
  `tree0_SPUR_component_8235`, RTX depth in control throughout. The shadow
  DA2 read the target about 2.4× too far (0.75–0.92 m predicted against
  0.29–0.39 m RTX). **The live-shadow gates FAILED**, the same three as
  Stage A: target p95, relative error and latency. The earlier "RUNNING" was
  never updated. The Envy/UFO USD "import passes" means `Usd.Stage.Open` with
  finite bounds inside the Isaac process; no Envy or UFO tree was rendered,
  ray-cast or targeted.
- **Stage A (21370005)**: 0 of 234 target frames within 20 mm, systematic
  overestimate of +0.41 to +0.62 m, latency p95 0.248 s.
  [Evidence](evidence/stage_a_depth_2026-09-23.json).
- **Pilot (21370039)**: evening tree-mask MAE is 6.1× source on Envy 00000 and
  7.9× on UFO 00000 for DA2; DINO adds nothing over DA2. N = 1 tree per family.
  The pilot's target-level metrics are not cited: the unmasked 3×3 window scored
  background, and the visibility annotations carried a half-pixel error.
  [Evidence](evidence/family_pilot_depth_2026-09-23.json).
- The eight-tree matrix is registered in
  [EVAL_PROTOCOL_FAMILY_LIGHTING_2026-09-23.md](EVAL_PROTOCOL_FAMILY_LIGHTING_2026-09-23.md)
  and submitted only on explicit approval.
