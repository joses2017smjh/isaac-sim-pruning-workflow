#!/bin/bash
#SBATCH --job-name=prune-render
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint="a40|rtx8000|l40s"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --output=/nfs/hpc/share/sanchej7/isaac-sim-pruning-workflow/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/isaac-sim-pruning-workflow/logs/%x-%j.out

set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export PRUNING_ROOT=/nfs/hpc/share/$USER/isaac-sim-pruning-workflow
export PYTHONPATH="$PRUNING_ROOT/source/isaaclab_pruning${PYTHONPATH:+:$PYTHONPATH}"
export ENABLE_CAMERAS=1
export PRUNING_RENDER_DIR="$PRUNING_ROOT/artifacts/isaac_render/job_${SLURM_JOB_ID}"
export BENCH_OUT="$PRUNING_ROOT/docs/evidence/smoke_${SLURM_JOB_ID}.json"
export PRUNING_RENDER_FRAMES="${PRUNING_RENDER_FRAMES:-140}"
export PRUNING_RUN_ENV_SMOKE="${PRUNING_RUN_ENV_SMOKE:-1}"
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PRUNING_ROOT PRUNING_RENDER_DIR PRUNING_RENDER_FRAMES PRUNING_RUN_ENV_SMOKE SLURM_JOB_ID SLURMD_NODENAME"
mkdir -p "$PRUNING_ROOT/logs" "$PRUNING_ROOT/artifacts/isaac_render" "$PRUNING_ROOT/docs/evidence"
# Refuse to reuse output from an earlier run or a requeued allocation.
mkdir "$PRUNING_RENDER_DIR"
date -Is
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
bhl_exec "$PRUNING_ROOT/hpc/inner/render_pruning_workflow.sh"
