#!/bin/bash
# One warm-start fine-tune arm of the DA2 metric head on an A40, from the
# frozen plan: data manifests, checkpoint hash, epochs and the arm come from
# plan.json, never from this file.
#SBATCH --job-name=prune-finetune-da2
#SBATCH --account=eecs
#SBATCH --partition=ampere
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail
BATCH_DIR="${1:?frozen batch directory}"
CODE="$BATCH_DIR/code"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME=/nfs/hpc/share/$USER/.cache/torch
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
PY=/nfs/hpc/share/$USER/miniforge3/envs/depth-env/bin/python
read -r ARM EPOCHS CKPT SHA SECONDS_LIMIT < <("$PY" - "$BATCH_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
batch = Path(sys.argv[1])
if not json.loads((batch / "storage_preflight.json").read_text())["ok"]:
    raise SystemExit("storage preflight did not pass")
plan = json.loads((batch / "plan.json").read_text())
print(plan["arm"], plan["epochs"], plan["warm_start"]["checkpoint"], plan["warm_start"]["checkpoint_sha256"], plan["max_runtime_seconds"])
PYEOF
)
exec "$PY" "$CODE/tools/finetune_da2.py" --companion /nfs/hpc/share/$USER/Computer_Vision \
 --checkpoint "$CKPT" --checkpoint-sha256 "$SHA" --arm "$ARM" --epochs "$EPOCHS" \
 --train-manifest "$BATCH_DIR/manifests/train.csv" --val-manifest "$BATCH_DIR/manifests/val.csv" \
 --extra-val-manifest "$BATCH_DIR/manifests/family_matrix.csv" \
 --output "$BATCH_DIR/train" --max-runtime-seconds "$SECONDS_LIMIT" --num-workers 4
