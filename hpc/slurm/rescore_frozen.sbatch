#!/bin/bash
# Score every named checkpoint on every named plan with the frozen scorer.
# Checkpoint hashes are computed here and recorded; a missing checkpoint is
# recorded as missing, never substituted.
#SBATCH --job-name=prune-rescore
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
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
"$PY" - "$BATCH_DIR" > "$OUT/pairs.txt" <<'PYEOF'
import json, sys
plan = json.load(open(sys.argv[1] + "/plan.json"))
for c, ce in plan["checkpoints"].items():
    for p, pe in plan["plans"].items():
        print(c, ce["path"], p, pe["path"])
PYEOF
: > "$OUT/status.jsonl"
while read -r CNAME CKPT PNAME PLAN; do
  if [ ! -f "$CKPT" ]; then echo "{\"checkpoint\": \"$CNAME\", \"plan\": \"$PNAME\", \"status\": \"checkpoint_missing\"}" >> "$OUT/status.jsonl"; continue; fi
  SHA=$(sha256sum "$CKPT" | cut -c1-64)
  if "$PY" "$CODE/tools/depth_generalization.py" --manifest "$PLAN" --companion /nfs/hpc/share/$USER/Computer_Vision \
      --checkpoint "$CKPT" --checkpoint-sha256 "$SHA" --output "$OUT/${CNAME}__${PNAME}" --save-predictions; then
    echo "{\"checkpoint\": \"$CNAME\", \"checkpoint_sha256\": \"$SHA\", \"plan\": \"$PNAME\", \"status\": \"scored\"}" >> "$OUT/status.jsonl"
  else
    echo "{\"checkpoint\": \"$CNAME\", \"checkpoint_sha256\": \"$SHA\", \"plan\": \"$PNAME\", \"status\": \"failed\"}" >> "$OUT/status.jsonl"
  fi
done < "$OUT/pairs.txt"
