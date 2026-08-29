# Dormant-spur pruning in Isaac Sim 6.0

**Measured on this cluster, not a tutorial screenshot.**

Jose Sanchez — Oregon State University

[![Gate 0](docs/demo/gate0_rtx.png)](docs/evidence/isaac_smoke_21077170.json)

Isaac Sim **5.1 RTX segfaults** here. **6.0 does not.** Cube planar-z is **1.5000 m**. Plane is **2.0000 m**. RGB std **38** — a bare cube with no material looked black; that was lighting, not a dead renderer.

```
150  A  flow 8×8
278  B  two VL53L8CX
 86  C  metric student   ⎤ same width — content differs
 86  D  fuse(ToF0+ToF1+C) ⎦
```

![Observation widths](docs/demo/obs_widths.png)

Cameras on the stage are not observations. BHL reported 9/9 ok at width **194 for every task**. This repo **asserts** A≠B≠C and `cfg.observation_space == obs.shape[-1]`.

## What you can see today

| Component | Demo | Evidence |
|---|---|---|
| RTX cube / plane | ![gate0](docs/demo/gate0_rtx.png) | [job 21077170](docs/evidence/isaac_smoke_21077170.json) |
| Finite cylinders, not capsules | ![tree](docs/demo/tree_cylinders.png) | Envy `00000`–`00009` USDA |
| 100 Envy + 100 UFO | — | [200 USDA](docs/evidence/trees_converted_manifest.json) |
| Blender pose, trunk median | — | [0.00055 mm](docs/evidence/blender_trunk_mm_lpy_envy_00000.json) |
| `bark_brown_02` UsdPreviewSurface | — | tree + orchard `Looks/` materials |
| Wrist camera offset (RGB still off) | — | [`close_lateral`](docs/evidence/camera_offset_raycast.json) |
| Cutter mouth / failure from STL | ![cutter](docs/demo/cutter_boxes.png) | [fitted AABB](docs/evidence/cutter_boxes_fitted.json) |
| D fuses **both** ToF + resampled metric | ![fusion](docs/demo/fusion_d.gif) | 8×8 causal table; native C is a second table |
| UR5e + mock-pruner USD | — | [job 21077217](docs/evidence/urdf_import_21077217.json) · **no slider** |

v1 does not spawn the hardware slider. IK is **absolute 7-D pose** (`use_relative_mode: false`). Relative mode is 6-D; keeping 7 with relative on would fail `set_command`.

## Stack that actually runs

`bhl.sif` + `venv-isaac60` · Isaac Sim **6.0.0.1** · Lab **3.0.0b2** · `BHL_STACK=v60`

Not NVIDIA 5.x `create_empty.py`. Not V100 (no RT cores). How we got here: [`docs/ISAAC_STACK.md`](docs/ISAAC_STACK.md).

```bash
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
slurm_clean sbatch hpc/slurm/isaac_headless_smoke.sbatch   # Gate 0 (passed)
slurm_clean sbatch hpc/slurm/env_smoke.sbatch              # next: one slot, A–D asserts
```

PPO is **refused** until both baselines have Isaac job logs.

```bash
python tools/train.py --variant B_tof --seed 0
# exits: missing baselines
```

## Reproduce the demos (no GPU)

```bash
python -m pip install -e "source/isaaclab_pruning[dev]"
python -m pytest -q -m "not isaacsim_ci"
python tools/render_demos.py
```

Control loop: DA2-ft once per episode (185 ms fp16 — not inside PPO). ToF / flow / student close at 20–50 Hz.

Full gates: [`docs/ROADMAP.md`](docs/ROADMAP.md) · HPC: [`docs/HPC.md`](docs/HPC.md)

Anchors: [spur-depth-service](https://github.com/joses2017smjh/spur-depth-service) · [bhl-robustness-ladder](https://github.com/joses2017smjh/bhl-robustness-ladder) · harness `5701a77`
