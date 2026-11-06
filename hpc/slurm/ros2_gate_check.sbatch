#!/bin/bash
#SBATCH --job-name=ros2-gate-check
#SBATCH --partition=share
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
# CPU only. Builds and tests the ROS 2 packages in the SIL container (node-local
# build), then runs tools/validate_safety_gate.py on the exported cycles.
#   sbatch hpc/slurm/ros2_gate_check.sbatch <cycles.jsonl.gz|-> <evidence.json> [panel]
set -euo pipefail
REPO=${PRUNING_ROOT:-/nfs/hpc/share/$USER/isaac-sim-pruning-workflow}
IMAGE=${PRUNING_SIL_IMAGE:-/nfs/hpc/share/$USER/containers/pruning_sil_humble_qt.sif}
CYCLES=${1:?cycles file or -}; OUT=${2:?evidence output}; PANEL=${3:-}
# A private DDS domain on loopback, so no other graph on this node can interfere.
export ROS_DOMAIN_ID=$(( (${SLURM_JOB_ID:-$$} % 90) + 10 ))
echo "node $(hostname) job ${SLURM_JOB_ID:-none} domain $ROS_DOMAIN_ID commit $(git -C $REPO rev-parse --short HEAD) image $(basename $IMAGE)"
git -C $REPO status --short | head -20
# --no-home keeps ~/.local out of the container, so ROS logs go to node-local /tmp.
LOGS=$(mktemp -d -p /tmp ros2-gate-logs.XXXXXX)
trap 'rm -rf "$LOGS"' EXIT
apptainer exec --cleanenv --no-home \
  --env PYTHONNOUSERSITE=1,ROS_LOCALHOST_ONLY=1,ROS_DOMAIN_ID=$ROS_DOMAIN_ID,ROS_LOG_DIR=$LOGS/log,ROS_HOME=$LOGS,HOME=$LOGS,QT_QPA_PLATFORM=offscreen,QT_QUICK_BACKEND=software \
  --env REHEARSE_CAPTURE=${REHEARSE_CAPTURE:-},REHEARSE_OUT=${REHEARSE_OUT:-} \
  --bind /nfs/hpc/share/$USER \
  $IMAGE bash $REPO/hpc/inner/ros2_gate_check.sh $REPO "$CYCLES" "$OUT" $PANEL
