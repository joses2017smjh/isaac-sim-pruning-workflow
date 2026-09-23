#!/bin/bash
#SBATCH --job-name=prune-family
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
BATCH_DIR="${1:?batch}"
TREE_ID="${2:?tree id}"
MODE="${3:-smoke}"
export PYTHONDONTWRITEBYTECODE=1
export COMPUTER_VISION_ROOT=/nfs/hpc/share/$USER/Computer_Vision
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/tmp"
export BLENDER_USER_CONFIG="$TMPDIR/blender-config" BLENDER_USER_SCRIPTS="$TMPDIR/blender-scripts"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
EXTRA=()
if [[ "$MODE" == smoke ]]; then EXTRA=(--one-view --lights source evening --width 256 --samples 4); fi
exec /usr/local/apps/blender/4.2.19/blender --background "$COMPUTER_VISION_ROOT/orchard_template.blend" \
 --threads 4 --python-exit-code 1 --python "$BATCH_DIR/code/tools/render_family_lighting.py" -- \
 --companion "$COMPUTER_VISION_ROOT" --tree-id "$TREE_ID" \
 --output "$BATCH_DIR/${MODE}_${TREE_ID}_job_$SLURM_JOB_ID" "${EXTRA[@]}"
