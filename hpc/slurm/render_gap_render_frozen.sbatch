#!/bin/bash
# One matrix tree re-rendered at the matrix camera poses and lighting presets with the
# training renderer's settings (render_family_lighting.py --renderer training: world fix,
# generate_tree2.render_tree, the template's 4096-sample adaptive sampler asserted, never
# changed), plus each view's depth and mask once more on the 512x288 matrix grid with the
# matrix sampler, which the evaluation's pose-identity check compares with the published
# ground truth. The tree for this task and the lights come from the frozen plan, never from
# this file, so a batch cannot render more or fewer trees than it registered; the launcher
# sets the partition and time limit from the plan. Never a smoke render.
#SBATCH --job-name=prune-render-gap
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=00:15:00
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?requires the submitted render array}"
BATCH_DIR="${1:?frozen batch directory}"
export PYTHONDONTWRITEBYTECODE=1 COMPUTER_VISION_ROOT=/nfs/hpc/share/$USER/Computer_Vision
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
LOCAL="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
export XDG_CACHE_HOME="$LOCAL/cache" TMPDIR="$LOCAL/tmp"
# Keep GPU driver kernel caches off the nearly full home directory.
export CUDA_CACHE_PATH="$TMPDIR/cuda-cache"
export BLENDER_USER_CONFIG="$TMPDIR/blender-config" BLENDER_USER_SCRIPTS="$TMPDIR/blender-scripts"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
# Everything below is read from the frozen plan and preflight. The plan names the tree
# for this index; the preflight must have passed when the batch was made.
read -r TREE_ID REVISION LIGHTS < <(python3 - "$BATCH_DIR" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
from pathlib import Path
batch, index = Path(sys.argv[1]), int(sys.argv[2])
if not json.loads((batch / "storage_preflight.json").read_text())["ok"]:
    raise SystemExit("storage preflight did not pass")
plan = json.loads((batch / "plan.json").read_text())
if plan["renderer"] != "training":
    raise SystemExit("this batch does not register the training renderer")
trees = [t for t in plan["trees"] if t["status"] == "to_render"]
if not 0 <= index < len(trees):
    raise SystemExit(f"array index {index} is outside the {len(trees)} registered renders")
print(trees[index]["tree_id"], plan["code_revision"], " ".join(plan["lights"]))
PY
)
export PRUNING_CODE_REVISION="$REVISION"
exec /usr/local/apps/blender/4.2.19/blender --background "$COMPUTER_VISION_ROOT/orchard_template.blend" \
 --threads 8 --python-exit-code 1 --python "$BATCH_DIR/code/tools/render_family_lighting.py" -- \
 --companion "$COMPUTER_VISION_ROOT" --tree-id "$TREE_ID" --renderer training \
 --output "$BATCH_DIR/render/$TREE_ID" --lights $LIGHTS
