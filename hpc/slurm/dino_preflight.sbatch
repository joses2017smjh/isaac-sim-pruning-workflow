#!/bin/bash
#SBATCH --job-name=prune-dino-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:10:00
set -euo pipefail
BATCH_DIR="${1:?batch}"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_HOME=/nfs/hpc/share/$USER/.cache/torch
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/tmp"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
exec /nfs/hpc/share/$USER/miniforge3/envs/depth-env/bin/python \
 "$BATCH_DIR/family_code_v1/code/tools/dino_generalization.py" \
 --companion /nfs/hpc/share/$USER/Computer_Vision \
 --da2-result "$BATCH_DIR/stage_a_job_21370005/evaluation.json" \
 --output "$BATCH_DIR/dino_smoke_job_$SLURM_JOB_ID" --smoke
