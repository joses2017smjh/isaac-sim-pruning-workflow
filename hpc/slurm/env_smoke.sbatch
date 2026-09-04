#!/bin/bash
#SBATCH --job-name=prune-env-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:50:00
#SBATCH --output=/nfs/hpc/share/sanchej7/isaac-sim-pruning-workflow/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/isaac-sim-pruning-workflow/logs/%x-%j.out

# One v60 slot: skrl/rsl_rl import, A–D obs-width asserts, env.step(0), PhysX contact.
# Do not split this into three jobs.

set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export PRUNING_ROOT=/nfs/hpc/share/$USER/isaac-sim-pruning-workflow
: "${PRUNING_ASSET_ID:?set PRUNING_ASSET_ID to the completed import asset_id}"
: "${PRUNING_USD:?set PRUNING_USD to the completed content-addressed root layer}"
: "${PRUNING_USD_EVIDENCE:?set PRUNING_USD_EVIDENCE to the completed import report}"
export PRUNING_ASSET_ID PRUNING_USD PRUNING_USD_EVIDENCE
export PYTHONPATH=$PRUNING_ROOT/source/isaaclab_pruning${PYTHONPATH:+:$PYTHONPATH}
export BENCH_OUT=$PRUNING_ROOT/docs/evidence/smoke_${SLURM_JOB_ID}.json
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PRUNING_ROOT PRUNING_ASSET_ID PRUNING_USD PRUNING_USD_EVIDENCE BENCH_OUT PYTHONPATH SLURM_JOB_ID SLURMD_NODENAME"
mkdir -p "$PRUNING_ROOT/logs" "$PRUNING_ROOT/docs/evidence"

[[ -f "$PRUNING_USD" ]] || { echo "missing $PRUNING_USD"; exit 1; }
[[ -f "$PRUNING_USD_EVIDENCE" ]] || { echo "missing $PRUNING_USD_EVIDENCE"; exit 1; }

echo "=== node: $(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) | $(date) ==="
bhl_exec "$PRUNING_ROOT/hpc/inner/smoke_env.sh"
echo "=== done: $(date) ==="
if [[ ! -f "$BENCH_OUT" ]]; then
    echo "missing $BENCH_OUT"
    exit 1
fi
python3 - <<PY
import json, sys
from pathlib import Path
report = json.loads(Path("$BENCH_OUT").read_text())
print("smoke json ok=", report.get("ok"), "skrl=", report.get("skrl"), "rsl_rl=", report.get("rsl_rl"))
print("spaces", report.get("observation_space"))
print("last_dim", report.get("obs_last_dim"))
sys.exit(0 if report.get("ok") is True else 1)
PY
