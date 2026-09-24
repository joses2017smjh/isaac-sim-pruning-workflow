#!/bin/bash
# Score one fine-tuned arm's best checkpoint on the three frozen evaluation
# plans the frozen model was scored on: the family matrix, the controls and
# Isaac Stage A. Runs afterany on the training job; a missing checkpoint is
# recorded, never substituted.
#SBATCH --job-name=prune-finetune-eval
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
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME" "$OUT"
PY=/nfs/hpc/share/$USER/miniforge3/envs/depth-env/bin/python
CKPT="$BATCH_DIR/train/best.pth"
if [ ! -f "$CKPT" ]; then
  echo '{"ok": false, "reason": "no best.pth from training"}' > "$OUT/status.json"
  exit 0
fi
SHA=$(sha256sum "$CKPT" | cut -c1-64)
echo "{\"checkpoint\": \"$CKPT\", \"checkpoint_sha256\": \"$SHA\"}" > "$OUT/status.json"
for NAME in family_matrix controls stage_a; do
  PLAN=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['evaluation_plans'][sys.argv[2]]['path'])" "$BATCH_DIR/plan.json" "$NAME")
  "$PY" "$CODE/tools/depth_generalization.py" --manifest "$PLAN" \
   --companion /nfs/hpc/share/$USER/Computer_Vision \
   --checkpoint "$CKPT" --checkpoint-sha256 "$SHA" \
   --output "$OUT/$NAME" --save-predictions
done
