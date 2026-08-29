#!/bin/bash
#SBATCH --job-name=prune-camera-rect
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/hpc/share/sanchej7/isaac-sim-pruning-workflow/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/isaac-sim-pruning-workflow/logs/%x-%j.out

# 30 cm camera-rect in Isaac (Phase 5). Same v60 stack as Gate 0.

set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export PRUNING_ROOT=/nfs/hpc/share/$USER/isaac-sim-pruning-workflow
export BENCH_OUT=$PRUNING_ROOT/docs/evidence/camera_rect_${SLURM_JOB_ID}.json
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PRUNING_ROOT BENCH_OUT"
mkdir -p "$PRUNING_ROOT/logs" "$PRUNING_ROOT/docs/evidence"

echo "=== node: $(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) | $(date) ==="
bhl_exec "$PRUNING_ROOT/hpc/inner/smoke_camera_rect.sh"
echo "=== done: $(date) ==="
python3 - <<PY
import json, sys
from pathlib import Path
report = json.loads(Path("$BENCH_OUT").read_text())
print("camera_rect ok=", report.get("ok"), "depth=", report.get("depth_m"))
sys.exit(0 if report.get("ok") is True else 1)
PY
