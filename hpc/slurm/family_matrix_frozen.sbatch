#!/bin/bash
# One Blender Cycles render of one L-Py tree under the plan's lighting presets.
# The array size and the tree list come from the frozen plan, never from this
# file, so a batch cannot render more or fewer trees than it registered.
#SBATCH --job-name=prune-family-matrix
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:15:00
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?requires the submitted render array}"
BATCH_DIR="${1:?frozen batch directory}"
export PYTHONDONTWRITEBYTECODE=1 COMPUTER_VISION_ROOT=/nfs/hpc/share/$USER/Computer_Vision
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/tmp"
export BLENDER_USER_CONFIG="$TMPDIR/blender-config" BLENDER_USER_SCRIPTS="$TMPDIR/blender-scripts"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
# Everything below is read from the frozen plan and preflight. The plan names
# the tree for this index; the preflight must have passed when the batch was made.
read -r TREE_ID WIDTH SAMPLES LIGHTS < <(python3 - "$BATCH_DIR" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
from pathlib import Path
batch, index = Path(sys.argv[1]), int(sys.argv[2])
if not json.loads((batch / "storage_preflight.json").read_text())["ok"]:
    raise SystemExit("storage preflight did not pass")
plan = json.loads((batch / "plan.json").read_text())
trees = [t for t in plan["trees"] if t["status"] == "to_render"]
if not 0 <= index < len(trees):
    raise SystemExit(f"array index {index} is outside the {len(trees)} registered renders")
print(trees[index]["tree_id"], plan["width"], plan["samples"], " ".join(plan["lights"]))
PY
)
OUT="$BATCH_DIR/matrix_${SLURM_ARRAY_TASK_ID}_${TREE_ID}"
exec /usr/local/apps/blender/4.2.19/blender --background "$COMPUTER_VISION_ROOT/orchard_template.blend" \
 --threads 4 --python-exit-code 1 --python "$BATCH_DIR/code/tools/render_family_lighting.py" -- \
 --companion "$COMPUTER_VISION_ROOT" --tree-id "$TREE_ID" \
 --output "$OUT" --width "$WIDTH" --samples "$SAMPLES" --lights $LIGHTS
