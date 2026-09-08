#!/bin/bash
# Run inside an existing, explicitly selected Slurm GPU allocation via srun.
set -euo pipefail
: "${SLURM_JOB_ID:?requires an allocated Slurm job}"
: "${SLURM_STEP_ID:?requires a separate srun step}"
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export PRUNING_ROOT=/nfs/hpc/share/$USER/isaac-sim-pruning-workflow
export PYTHONPATH="$PRUNING_ROOT/source/isaaclab_pruning${PYTHONPATH:+:$PYTHONPATH}"
export ENABLE_CAMERAS=1
export OMP_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PRUNING_RENDER_DIR="$PRUNING_ROOT/artifacts/isaac_render/alloc_${SLURM_JOB_ID}_step${SLURM_STEP_ID}"
export BENCH_OUT="$PRUNING_ROOT/docs/evidence/smoke_${SLURM_JOB_ID}_step${SLURM_STEP_ID}.json"
export PRUNING_RENDER_FRAMES="${PRUNING_RENDER_FRAMES:-140}"
export PRUNING_RUN_ENV_SMOKE="${PRUNING_RUN_ENV_SMOKE:-1}"
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PRUNING_ROOT PRUNING_RENDER_DIR PRUNING_RENDER_FRAMES PRUNING_RUN_ENV_SMOKE PRUNING_SMOKE_ONLY SLURM_JOB_ID SLURM_STEP_ID SLURMD_NODENAME OPENBLAS_NUM_THREADS MKL_NUM_THREADS"
mkdir -p "$PRUNING_ROOT/artifacts/isaac_render" "$PRUNING_ROOT/docs/evidence" "$PRUNING_ROOT/logs"
mkdir "$PRUNING_RENDER_DIR"
date -Is
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
bhl_exec "$PRUNING_ROOT/hpc/inner/render_pruning_workflow.sh"
