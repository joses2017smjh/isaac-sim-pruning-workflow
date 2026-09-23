#!/bin/bash
#SBATCH --job-name=prune-depth-a
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:15:00
set -euo pipefail
BATCH_DIR="${1:?frozen batch directory}"
export TORCH_HOME=/nfs/hpc/share/$USER/.cache/torch
export HF_HOME=/nfs/hpc/share/$USER/.cache/huggingface
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/tmp"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
PY=/nfs/hpc/share/$USER/miniforge3/envs/depth-env/bin/python
"$PY" - "$BATCH_DIR" <<'PY'
import hashlib,json,sys
from pathlib import Path
p=Path(sys.argv[1]);pins=json.loads((p/'source_pins.json').read_text())
for name,h in pins.items():
 assert hashlib.sha256((p/name).read_bytes()).hexdigest()==h,name
s=json.loads((p/'storage_preflight.json').read_text());assert s['ok']
PY
exec "$PY" "$BATCH_DIR/code/tools/depth_generalization.py" \
  --manifest "$BATCH_DIR/plans/stage_a.json" \
  --companion /nfs/hpc/share/$USER/Computer_Vision \
  --checkpoint /nfs/hpc/share/$USER/Computer_Vision/checkpoints/full_spur_2tex_all_3view_seed1/best.pth \
  --checkpoint-sha256 5ecc5182a2717e14f65dbb9d2d400aac6927435f6e92e87b7e891fb8f7a82834 \
  --output "$BATCH_DIR/stage_a_job_$SLURM_JOB_ID" --save-predictions
