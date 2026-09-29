#!/bin/bash
# Resume one stopped rendered-lighting tree inside the batch that registered it:
# only the frames its task never rendered, with this batch's frozen code. The
# tree comes from this batch's plan (arrays[0].trees[$SLURM_ARRAY_TASK_ID]), the
# frames from the target batch's plan. The task refuses to start once any
# fine-tune that reads the target batch has built its manifest, so a manifest
# already consumed is never changed underneath it.
#SBATCH --job-name=prune-train-lighting-resume
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?requires the submitted resume array}"
BATCH_DIR="${1:?frozen resume batch directory}"
export PYTHONDONTWRITEBYTECODE=1 COMPUTER_VISION_ROOT=/nfs/hpc/share/$USER/Computer_Vision
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
LOCAL="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}"
export XDG_CACHE_HOME="$LOCAL/cache" TMPDIR="$LOCAL/tmp"
export BLENDER_USER_CONFIG="$TMPDIR/blender-config" BLENDER_USER_SCRIPTS="$TMPDIR/blender-scripts"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
read -r TREE_ID TARGET REVISION < <(python3 - "$BATCH_DIR" "$SLURM_ARRAY_TASK_ID" <<'PY'
import json, sys
from pathlib import Path
batch, task = Path(sys.argv[1]), int(sys.argv[2])
if not json.loads((batch / "storage_preflight.json").read_text())["ok"]:
    raise SystemExit("storage preflight did not pass")
plan = json.loads((batch / "plan.json").read_text())
if plan.get("mode") != "resume":
    raise SystemExit("not a resume plan")
built = [c["batch"] for c in plan["consumers"] if (Path(c["batch"]) / "manifests/rendered").exists()]
if built:
    raise SystemExit(f"refusing to change renders a fine-tune already read: {built}")
trees = plan["arrays"][0]["trees"]
if not 0 <= task < len(trees):
    raise SystemExit(f"task {task} is outside the {len(trees)} trees of this resume")
print(trees[task], plan["target_batch"], plan["code_revision"])
PY
)
export PRUNING_CODE_REVISION="$REVISION"
exec /usr/local/apps/blender/4.2.19/blender --background "$COMPUTER_VISION_ROOT/orchard_template.blend" \
 --threads 8 --python-exit-code 1 --python "$BATCH_DIR/code/tools/render_training_lighting.py" -- \
 --companion "$COMPUTER_VISION_ROOT" --plan "$TARGET/plan.json" --tree-id "$TREE_ID" \
 --output "$TARGET/render/$TREE_ID" --scratch "$TMPDIR/checks" --resume
