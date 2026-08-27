# HPC bring-up

Inventory queried from Slurm on 2026-08-27:

- `cn-gpu5`–`cn-gpu7`: 8× RTX 8000, partitions `gpu` / `gpu-dmv`.
- `cn-gpu10`–`cn-gpu12`: 8× L40S, partition `tiamat`.
- `cn-r-*`, `cn-s-*`, `cn-t-1`: A40, partitions `ampere` / `athena`.
- `cn-w-1`: H100; `cn-w-2`: L40S.
- `dgx2-*`: V100. Use these for CUDA ray-cast training only, not RTX rendering.

The account association returned `eecs` with normal QOS and no fixed partition.
Seeing a node in `sinfo` does **not** prove Jose can submit to that partition.
The GPU gate remains open until the smoke job succeeds.

## Gate 0A: container

Use an NVIDIA Isaac Sim container with a matching Isaac Lab installation and
convert it to a local SIF. The exact image digest must be recorded here before
experiments begin; do not rely on a moving tag.

```bash
export ISAAC_CONTAINER=/path/to/pinned-isaac-sim.sif
mkdir -p logs
sbatch hpc/slurm/isaac_headless_smoke.sbatch
```

Pass condition:

1. The job lands on an RTX-capable node.
2. Vulkan/EGL initialization succeeds.
3. Isaac Lab's `create_empty.py --headless` exits zero.
4. The image digest, Isaac Sim version, driver, and node type are recorded.

This repository does not claim the gate has passed yet: no SIF path was present
when the foundation was created.

## Cache policy

The smoke script puts Omniverse cache, config, data, and logs in
`$SLURM_TMPDIR`. Keep large Kit caches and generated USD artifacts off the
stak home filesystem.
