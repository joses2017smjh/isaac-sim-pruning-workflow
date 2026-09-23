#!/bin/bash
#SBATCH --job-name=prune-family-matrix
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
#SBATCH --array=0-5%2
set -euo pipefail
BATCH_DIR="${1:?batch}"
export PYTHONDONTWRITEBYTECODE=1 COMPUTER_VISION_ROOT=/nfs/hpc/share/$USER/Computer_Vision
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/tmp"
export BLENDER_USER_CONFIG="$TMPDIR/blender-config" BLENDER_USER_SCRIPTS="$TMPDIR/blender-scripts"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
TREES=(lpy_envy_00003 lpy_envy_00008 lpy_envy_00012 lpy_ufo_00001 lpy_ufo_00002 lpy_ufo_00003)
python3 - "$BATCH_DIR" <<'PY'
import json,sys
from pathlib import Path
p=Path(sys.argv[1]);assert json.loads((p/'storage_preflight_v2.json').read_text())['ok']
PY
exec /usr/local/apps/blender/4.2.19/blender --background "$COMPUTER_VISION_ROOT/orchard_template.blend" \
 --threads 4 --python-exit-code 1 --python "$BATCH_DIR/family_code_v3/code/tools/render_family_lighting.py" -- \
 --companion "$COMPUTER_VISION_ROOT" --tree-id "${TREES[$SLURM_ARRAY_TASK_ID]}" \
 --output "$BATCH_DIR/matrix_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}" --width 512 --samples 16
