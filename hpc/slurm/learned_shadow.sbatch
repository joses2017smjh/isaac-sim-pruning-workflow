#!/bin/bash
#SBATCH --job-name=prune-shadow
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint="a40|rtx8000|l40s"
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:25:00
set -euo pipefail
BATCH_DIR="${1:?batch}"
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export PRUNING_ROOT="$BATCH_DIR/shadow_code_v3"
export PYTHONPATH="$PRUNING_ROOT/source/isaaclab_pruning${PYTHONPATH:+:$PYTHONPATH}"
export ENABLE_CAMERAS=1
export PRUNING_RENDER_DIR="$BATCH_DIR/shadow_job_$SLURM_JOB_ID"
export BENCH_OUT="$PRUNING_RENDER_DIR/env_smoke.json"
export PRUNING_RENDER_MODE=blender_vision PRUNING_RENDER_QUALITY=pathtraced
export PRUNING_OVERVIEW_WIDTH=1280 PRUNING_RENDER_SAMPLES=64
export PRUNING_RENDER_FRAMES="${4:-30}" PRUNING_RUN_ENV_SMOKE=0 PRUNING_DAYLIGHT=source
export PRUNING_FAMILY_IMPORT_CHECK=1
export PRUNING_TARGET_TREE="${2:-0}" PRUNING_COMPONENT_VERTEX="${3:-8235}"
export PRUNING_SHADOW_DIR="$PRUNING_RENDER_DIR/shadow"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_HOME=/nfs/hpc/share/$USER/.cache/torch
mkdir "$PRUNING_RENDER_DIR"
/nfs/hpc/share/$USER/miniforge3/envs/depth-env/bin/python "$PRUNING_ROOT/tools/shadow_depth.py" \
 --folder "$PRUNING_SHADOW_DIR" --companion /nfs/hpc/share/$USER/Computer_Vision \
 --checkpoint /nfs/hpc/share/$USER/Computer_Vision/checkpoints/full_spur_2tex_all_3view_seed1/best.pth \
 --checkpoint-sha256 5ecc5182a2717e14f65dbb9d2d400aac6927435f6e92e87b7e891fb8f7a82834 \
 > "$PRUNING_RENDER_DIR/shadow_worker.log" 2>&1 &
SHADOW_PID=$!
trap 'kill "$SHADOW_PID" 2>/dev/null || true' EXIT
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PRUNING_ROOT PRUNING_RENDER_DIR PRUNING_RENDER_MODE PRUNING_RENDER_QUALITY PRUNING_OVERVIEW_WIDTH PRUNING_RENDER_SAMPLES PRUNING_RENDER_FRAMES PRUNING_RUN_ENV_SMOKE PRUNING_DAYLIGHT PRUNING_FAMILY_IMPORT_CHECK PRUNING_TARGET_TREE PRUNING_COMPONENT_VERTEX PRUNING_SHADOW_DIR BENCH_OUT SLURM_JOB_ID SLURMD_NODENAME"
set +e
bhl_exec "$PRUNING_ROOT/hpc/inner/render_pruning_workflow.sh"
RENDER_STATUS=$?
set -e
touch "$PRUNING_SHADOW_DIR/stop"
wait "$SHADOW_PID"
exit "$RENDER_STATUS"
