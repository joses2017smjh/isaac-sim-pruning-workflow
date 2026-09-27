#!/bin/bash
# One Envy training tree re-rendered at its recorded poses under seeded lighting
# draws. The tree for this task and the array it belongs to come from the frozen
# plan (arrays[$2].trees[$SLURM_ARRAY_TASK_ID]); the time limit is set per array
# by the launcher from the plan. Depth and mask checks go to node-local scratch
# and are deleted; only RGB and the manifest reach the share.
#SBATCH --job-name=prune-train-lighting
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?requires the submitted render array}"
BATCH_DIR="${1:?frozen batch directory}"
ARRAY_INDEX="${2:?array index in the plan}"
export PYTHONDONTWRITEBYTECODE=1 COMPUTER_VISION_ROOT=/nfs/hpc/share/$USER/Computer_Vision
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
LOCAL="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
export XDG_CACHE_HOME="$LOCAL/cache" TMPDIR="$LOCAL/tmp"
export BLENDER_USER_CONFIG="$TMPDIR/blender-config" BLENDER_USER_SCRIPTS="$TMPDIR/blender-scripts"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
read -r TREE_ID CHECKS < <(python3 - "$BATCH_DIR" "$ARRAY_INDEX" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
from pathlib import Path
batch, array_index, task = Path(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
if not json.loads((batch / "storage_preflight.json").read_text())["ok"]:
    raise SystemExit("storage preflight did not pass")
plan = json.loads((batch / "plan.json").read_text())
trees = plan["arrays"][array_index]["trees"]
if not 0 <= task < len(trees):
    raise SystemExit(f"task {task} is outside the {len(trees)} trees of array {array_index}")
print(trees[task], plan["source_checks"])
PY
)
exec /usr/local/apps/blender/4.2.19/blender --background "$COMPUTER_VISION_ROOT/orchard_template.blend" \
 --threads 8 --python-exit-code 1 --python "$BATCH_DIR/code/tools/render_training_lighting.py" -- \
 --companion "$COMPUTER_VISION_ROOT" --plan "$BATCH_DIR/plan.json" --tree-id "$TREE_ID" \
 --output "$BATCH_DIR/render/$TREE_ID" --scratch "$TMPDIR/checks" --source-checks "$CHECKS"
