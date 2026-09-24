#!/bin/bash
# Frozen DA2 metric and relative-head scoring of the tree0 replay renders.
# Runs after the render with afterany; missing frames stay in the plan.
#SBATCH --job-name=prune-tree0-eval
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
BATCH_DIR="${1:?frozen batch directory}"
CODE="$BATCH_DIR/code"
OUT="$BATCH_DIR/eval"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_HOME=/nfs/hpc/share/$USER/.cache/torch
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$OUT"
PY=/nfs/hpc/share/$USER/miniforge3/envs/depth-env/bin/python
EXPECTED=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['registered_frames'])" "$BATCH_DIR/plan.json")
"$PY" "$CODE/tools/prepare_replay_evaluation.py" --manifest "$BATCH_DIR/render/render_manifest.json" --output "$OUT/plan.json" --expected-frames "$EXPECTED"
DA2_SHA=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['models']['da2']['checkpoint_sha256'])" "$BATCH_DIR/plan.json")
DA2_CKPT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['models']['da2']['checkpoint'])" "$BATCH_DIR/plan.json")
"$PY" "$CODE/tools/depth_generalization.py" --manifest "$OUT/plan.json" \
 --companion /nfs/hpc/share/$USER/Computer_Vision \
 --checkpoint "$DA2_CKPT" --checkpoint-sha256 "$DA2_SHA" \
 --output "$OUT/da2" --save-predictions
REL_SHA=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['relative_head']['checkpoint_sha256'])" "$BATCH_DIR/plan.json")
REL_CKPT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['relative_head']['checkpoint'])" "$BATCH_DIR/plan.json")
exec "$PY" "$CODE/tools/relative_depth_ceiling.py" --companion /nfs/hpc/share/$USER/Computer_Vision \
 --checkpoint "$REL_CKPT" --checkpoint-sha256 "$REL_SHA" \
 --frames "tree0_replay=$OUT/plan.json" --output "$OUT/relative"
