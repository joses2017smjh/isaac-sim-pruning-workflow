#!/bin/bash
#SBATCH --job-name=prune-baselines
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --output=/nfs/hpc/share/sanchej7/isaac-sim-pruning-workflow/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/isaac-sim-pruning-workflow/logs/%x-%j.out

# Scripted ToF in Isaac + CuRobo configure status. Not PPO.

set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export PRUNING_ROOT=/nfs/hpc/share/$USER/isaac-sim-pruning-workflow
export PYTHONPATH=$PRUNING_ROOT/source/isaaclab_pruning${PYTHONPATH:+:$PYTHONPATH}
export BENCH_OUT=$PRUNING_ROOT/docs/evidence/baselines_${SLURM_JOB_ID}.json
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PRUNING_ROOT BENCH_OUT PYTHONPATH SLURM_JOB_ID SLURMD_NODENAME"
mkdir -p "$PRUNING_ROOT/logs" "$PRUNING_ROOT/docs/evidence"

echo "=== node: $(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) | $(date) ==="
bhl_exec "$PRUNING_ROOT/hpc/inner/run_baselines.sh"
echo "=== done: $(date) ==="
python3 - <<PY
import json, sys
from pathlib import Path
report = json.loads(Path("$BENCH_OUT").read_text())
print("baselines ok=", report.get("ok"), "scripted=", report.get("scripted_tof"), "curobo=", report.get("curobo_oracle"))
sys.exit(0 if report.get("ok") is True else 1)
PY
