#!/bin/bash
# Frozen DA2 scoring over every registered control condition, the test-time
# photometric variants derived in this job from the plain evening cell, the
# relative DA2 head on the same frames plus the pinned matrix and Stage A
# frames, and six-view DINO on the cells that match its rig. Runs after the
# render array with afterany: a missing tree or condition is recorded in
# render_status.json and plan.json and stays in the denominator; nothing is
# scored twice. Steps run in order of evidential weight so a late failure
# cannot lose an earlier result.
#SBATCH --job-name=prune-controls-eval
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
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
mkdir "$OUT"
PY=/nfs/hpc/share/$USER/miniforge3/envs/depth-env/bin/python
"$PY" "$CODE/tools/prepare_controls_evaluation.py" --batch "$BATCH_DIR" --output "$OUT"
DA2_SHA=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['models']['da2']['checkpoint_sha256'])" "$BATCH_DIR/plan.json")
DA2_CKPT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['models']['da2']['checkpoint'])" "$BATCH_DIR/plan.json")
"$PY" "$CODE/tools/depth_generalization.py" --manifest "$OUT/plan.json" \
 --companion /nfs/hpc/share/$USER/Computer_Vision \
 --checkpoint "$DA2_CKPT" --checkpoint-sha256 "$DA2_SHA" \
 --output "$OUT/da2" --save-predictions
REL_SHA=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['relative_head']['checkpoint_sha256'])" "$BATCH_DIR/plan.json")
REL_CKPT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['relative_head']['checkpoint'])" "$BATCH_DIR/plan.json")
read -r STAGE_A MATRIX < <("$PY" -c "import json,sys;e=json.load(open(sys.argv[1]))['relative_head']['extra_frames'];print(e['stage_a_isaac']['path'], e['family_matrix']['path'])" "$BATCH_DIR/plan.json")
"$PY" "$CODE/tools/relative_depth_ceiling.py" --companion /nfs/hpc/share/$USER/Computer_Vision \
 --checkpoint "$REL_CKPT" --checkpoint-sha256 "$REL_SHA" \
 --frames "controls=$OUT/plan.json" --frames "family_matrix=$MATRIX" --frames "stage_a_isaac=$STAGE_A" \
 --output "$OUT/relative"
exec "$PY" "$CODE/tools/dino_generalization.py" --companion /nfs/hpc/share/$USER/Computer_Vision \
 --da2-result "$OUT/da2/evaluation.json" --output "$OUT/dino"
