#!/bin/bash
#SBATCH --job-name=prune-family-eval
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:20:00
set -euo pipefail
BATCH_DIR="${1:?batch}"
INPUT_PLAN="${2:?plan}"
CODE="$BATCH_DIR/family_code_v3/code"
OUT="$BATCH_DIR/family_eval_job_$SLURM_JOB_ID"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_HOME=/nfs/hpc/share/$USER/.cache/torch
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/tmp"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
mkdir "$OUT"
PY=/nfs/hpc/share/$USER/miniforge3/envs/depth-env/bin/python
"$PY" "$CODE/tools/prepare_family_evaluation.py" --inputs "$INPUT_PLAN" --output "$OUT/plan.json"
"$PY" "$CODE/tools/depth_generalization.py" --manifest "$OUT/plan.json" \
 --companion /nfs/hpc/share/$USER/Computer_Vision \
 --checkpoint /nfs/hpc/share/$USER/Computer_Vision/checkpoints/full_spur_2tex_all_3view_seed1/best.pth \
 --checkpoint-sha256 5ecc5182a2717e14f65dbb9d2d400aac6927435f6e92e87b7e891fb8f7a82834 \
 --output "$OUT/da2" --save-predictions
exec "$PY" "$CODE/tools/dino_generalization.py" --companion /nfs/hpc/share/$USER/Computer_Vision \
 --da2-result "$OUT/da2/evaluation.json" --output "$OUT/dino"
