#!/bin/bash
#SBATCH --job-name=prune-ros2-ws
#SBATCH --partition=share
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=00:30:00
# CPU only. Builds the frozen batch's ROS 2 packages (messages, C++ gate, controller,
# operator panel) into <batch>/ros2_ws/install for the GPU tasks that depend on this job.
set -euo pipefail
BATCH="${1:?requires frozen batch directory}"
IMAGE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sil_image"])' "$BATCH/plan.json")
BUILD=$(mktemp -d -p /tmp ros2-ws-build.XXXXXX)
trap 'rm -rf "$BUILD"' EXIT
test ! -e "$BATCH/ros2_ws" || { echo "workspace already exists; refusing to overwrite" >&2; exit 2; }
mkdir -p "$BATCH/ros2_ws"
echo "node $(hostname) image $IMAGE start $(date -Is)"
apptainer exec --cleanenv --no-home --bind /nfs/hpc/share/$USER --env PYTHONNOUSERSITE=1,HOME=$BUILD "$IMAGE" bash -c "
  set -eo pipefail
  source /opt/ros/humble/setup.bash
  cd $BUILD
  colcon build --base-paths $BATCH/code/ros2 --build-base $BUILD/build --install-base $BATCH/ros2_ws/install \
    --event-handlers console_direct- --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo -DBUILD_TESTING=OFF
"
test -x "$BATCH/ros2_ws/install/pruning_sil_cpp/lib/pruning_sil_cpp/safety_gate_node"
# Hash what was built; the operator panel exists only in batches whose code includes it.
find "$BATCH/ros2_ws/install" -path "*/lib/pruning_sil_cpp/safety_gate_node" -o -path "*/lib/pruning_operator_panel/operator_panel" \
  | sort | xargs sha256sum | tee "$BATCH/ros2_ws/executables.sha256"
echo "end $(date -Is)"
