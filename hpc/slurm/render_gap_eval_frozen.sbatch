#!/bin/bash
# Score the five registered checkpoints on the render-gap renders in two arms. Runs after
# the render array with afterany, so a failed render leaves a recorded gap rather than a
# job pending forever: prepare_render_gap_evaluation.py writes render_status.json (which
# registered trees rendered), pose_identity.json (the per-view check that admits a tree
# to MATCHED) and one plan per arm in prepare_family_evaluation.py's format. An arm with
# no plan is skipped and its status.json says why; the tree count never shrinks silently.
# Each checkpoint is scored exactly as the family and fine-tune evaluations score it:
# depth_generalization.py with its plan hash, into eval/<arm>/<model>/evaluation.json.
#SBATCH --job-name=prune-render-gap-eval
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
set -euo pipefail
BATCH_DIR="${1:?frozen batch directory}"
CODE="$BATCH_DIR/code"
OUT="$BATCH_DIR/eval"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_HOME=/nfs/hpc/share/$USER/.cache/torch
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/tmp"
# Keep GPU driver kernel caches off the nearly full home directory.
export CUDA_CACHE_PATH="$TMPDIR/cuda-cache"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
mkdir "$OUT"
PY=/nfs/hpc/share/$USER/miniforge3/envs/depth-env/bin/python
COMPANION=/nfs/hpc/share/$USER/Computer_Vision
"$PY" "$CODE/tools/prepare_render_gap_evaluation.py" --batch "$BATCH_DIR" --output "$OUT"
MODELS=$("$PY" -c "import json,sys;print(' '.join(json.load(open(sys.argv[1]))['models']))" "$BATCH_DIR/plan.json")
for MODEL in $MODELS; do
  CKPT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['models'][sys.argv[2]]['checkpoint'])" "$BATCH_DIR/plan.json" "$MODEL")
  SHA=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['models'][sys.argv[2]]['checkpoint_sha256'])" "$BATCH_DIR/plan.json" "$MODEL")
  for ARM in native matched; do
    if [ ! -f "$OUT/$ARM/plan.json" ]; then
      echo "$ARM has no evaluation plan; $OUT/$ARM/status.json says why"
      continue
    fi
    "$PY" "$CODE/tools/depth_generalization.py" --manifest "$OUT/$ARM/plan.json" \
     --companion "$COMPANION" \
     --checkpoint "$CKPT" --checkpoint-sha256 "$SHA" \
     --output "$OUT/$ARM/$MODEL" --save-predictions
  done
done
