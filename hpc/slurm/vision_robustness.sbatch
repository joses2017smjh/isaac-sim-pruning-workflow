#!/bin/bash
#SBATCH --job-name=prune-vision-pilot
#SBATCH --account=eecs
#SBATCH --partition=ampere
#SBATCH --constraint=a40
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:25:00
#SBATCH --array=0-5%1

set -euo pipefail
: "${SLURM_ARRAY_TASK_ID:?requires the submitted experiment array}"
export PRUNING_BATCH_DIR="${1:?requires frozen batch directory}"
export PRUNING_RUN_INDEX="$SLURM_ARRAY_TASK_ID"
export PRUNING_ROOT="$PRUNING_BATCH_DIR/code"
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export PYTHONPATH="$PRUNING_ROOT/source/isaaclab_pruning"
export ENABLE_CAMERAS=1
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PRUNING_ROOT PRUNING_BATCH_DIR PRUNING_RUN_INDEX SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURMD_NODENAME OPENBLAS_NUM_THREADS MKL_NUM_THREADS"
bhl_exec "$PRUNING_ROOT/hpc/inner/vision_robustness.sh"
