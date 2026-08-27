#!/bin/bash
#SBATCH --job-name=isaac-smoke
#SBATCH --partition=gpu
#SBATCH --constraint=rtx8000
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail

: "${ISAAC_CONTAINER:?Set ISAAC_CONTAINER to an Isaac Sim + Isaac Lab .sif}"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/IsaacLab}"
SCRATCH_ROOT="${SLURM_TMPDIR:-/tmp/${USER}/isaac-${SLURM_JOB_ID}}"

mkdir -p logs "$SCRATCH_ROOT"/{cache,config,data,logs}
export OMNI_USER_CACHE_DIR="$SCRATCH_ROOT/cache"
export OMNI_USER_CONFIG_DIR="$SCRATCH_ROOT/config"
export OMNI_USER_DATA_DIR="$SCRATCH_ROOT/data"
export OMNI_USER_LOG_DIR="$SCRATCH_ROOT/logs"

nvidia-smi
apptainer exec --nv \
    --bind "$PWD:/workspace/isaac-sim-pruning-workflow" \
    "$ISAAC_CONTAINER" \
    "$ISAACLAB_ROOT/isaaclab.sh" -p \
    "$ISAACLAB_ROOT/scripts/tutorials/00_sim/create_empty.py" \
    --headless
