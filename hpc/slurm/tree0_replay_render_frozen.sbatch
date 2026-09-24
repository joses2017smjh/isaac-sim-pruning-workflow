#!/bin/bash
# One Blender Cycles job: the original orchard tree0 at the recorded Isaac wrist
# poses of the Stage A runs the frozen plan names, two bark textures per pose.
#SBATCH --job-name=prune-tree0-replay
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:20:00
set -euo pipefail
BATCH_DIR="${1:?frozen batch directory}"
export PYTHONDONTWRITEBYTECODE=1 COMPUTER_VISION_ROOT=/nfs/hpc/share/$USER/Computer_Vision
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/tmp"
export BLENDER_USER_CONFIG="$TMPDIR/blender-config" BLENDER_USER_SCRIPTS="$TMPDIR/blender-scripts"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
read -r SAMPLES STRIDE MAXFRAME < <(python3 - "$BATCH_DIR" <<'PY'
import json, sys
from pathlib import Path
batch = Path(sys.argv[1])
if not json.loads((batch / "storage_preflight.json").read_text())["ok"]:
    raise SystemExit("storage preflight did not pass")
plan = json.loads((batch / "plan.json").read_text())
print(plan["samples"], plan["frame_stride"], plan["max_frame"])
PY
)
RUNS=$(python3 -c "import json,sys; print(' '.join('--run-dir '+r['path'] for r in json.load(open(sys.argv[1]))['runs']))" "$BATCH_DIR/plan.json")
exec /usr/local/apps/blender/4.2.19/blender --background "$COMPUTER_VISION_ROOT/orchard_template.blend" \
 --threads 4 --python-exit-code 1 --python "$BATCH_DIR/code/tools/render_tree0_replay.py" -- \
 --companion "$COMPUTER_VISION_ROOT" $RUNS \
 --manifest "$BATCH_DIR/code/artifacts/blender_scene/orchard_two_trees_v1/manifest.json" \
 --output "$BATCH_DIR/render" --samples "$SAMPLES" --frame-stride "$STRIDE" --max-frame "$MAXFRAME"
