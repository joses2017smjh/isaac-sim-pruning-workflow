#!/bin/bash
#SBATCH --job-name=prune-urdf-import
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/hpc/share/sanchej7/isaac-sim-pruning-workflow/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/isaac-sim-pruning-workflow/logs/%x-%j.out

# Phase 2 GPU step. Submit only after logs/isaac_smoke.json has "ok": true.

set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export PRUNING_ROOT=/nfs/hpc/share/$USER/isaac-sim-pruning-workflow
export PRUNING_URDF=${PRUNING_URDF:-$PRUNING_ROOT/artifacts/urdf/ur5e_pruner_abs.urdf}
export PRUNING_USD_DIR=$PRUNING_ROOT/artifacts/usd
export BENCH_OUT=$PRUNING_ROOT/logs/urdf_import.json
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PRUNING_ROOT PRUNING_URDF PRUNING_USD_DIR BENCH_OUT"
mkdir -p "$PRUNING_ROOT/logs" "$PRUNING_USD_DIR"

if [[ ! -f "$PRUNING_URDF" ]]; then
    echo "missing rewritten URDF: $PRUNING_URDF"
    echo "run: python tools/rewrite_urdf_paths.py ... first"
    exit 1
fi

echo "=== node: $(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) | $(date) ==="
bhl_exec "$PRUNING_ROOT/hpc/inner/import_urdf.sh"
echo "=== done: $(date) ==="
