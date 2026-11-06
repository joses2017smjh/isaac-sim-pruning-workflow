#!/bin/bash
#SBATCH --job-name=prune-ros2-loop
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --constraint=a40|rtx8000
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:35:00
# One frozen ROS 2 live-loop trial per array task (tools/queue_ros2_live_loop.py submits it).
#
# The ROS 2 side (pruning_sil live_bringup: the controller, the C++ safety gate and,
# for a panel row, the Qt/QML operator panel) runs in the SIL container on this node.
# Isaac Sim runs in bhl.sif through the usual vision-experiment runner with
# PRUNING_ROS2_LOOP=1, so the simulated UR5e moves only on the gate's command.
# Both share a private loopback DDS domain; nothing leaves the node.
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
export ROS_LOCALHOST_ONLY=1
export ROS_DOMAIN_ID=$(( (${SLURM_JOB_ID} % 90) + 10 ))

WS="$PRUNING_BATCH_DIR/ros2_ws/install"
IMAGE=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["sil_image"])' "$PRUNING_BATCH_DIR/plan.json")
HOLD_AFTER=$(python3 -c 'import json,sys; r=json.load(open(sys.argv[1]))["runs"][int(sys.argv[2])]; v=r["ros2_loop"].get("operator_hold_after_cycles"); print("" if v is None else int(v))' "$PRUNING_BATCH_DIR/plan.json" "$PRUNING_RUN_INDEX")
test -x "$WS/pruning_sil_cpp/lib/pruning_sil_cpp/safety_gate_node" || { echo "ROS 2 workspace not built: $WS" >&2; exit 2; }
PANEL=()
if [[ -n "$HOLD_AFTER" ]]; then
  PANEL=(--panel "$WS/pruning_operator_panel/lib/pruning_operator_panel/operator_panel"
         --auto-hold-after-cycles "$HOLD_AFTER" --record "{output_dir}/operator_panel_record.jsonl"
         --screenshot-dir "{output_dir}/panel")
fi

ROSLOG=$(mktemp -d -p "${TMPDIR:-/tmp}" ros2-loop.XXXXXX)
BRINGUP_LOG="$PRUNING_BATCH_DIR/logs/bringup-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.out"
echo "node $(hostname) domain $ROS_DOMAIN_ID image $IMAGE hold_after=${HOLD_AFTER:-none}"
apptainer exec --cleanenv --no-home --bind /nfs/hpc/share/$USER \
  --env PYTHONNOUSERSITE=1,ROS_LOCALHOST_ONLY=1,ROS_DOMAIN_ID=$ROS_DOMAIN_ID,ROS_LOG_DIR=$ROSLOG/log,ROS_HOME=$ROSLOG,HOME=$ROSLOG \
  --env QT_QPA_PLATFORM=offscreen,QT_QUICK_BACKEND=software \
  "$IMAGE" bash -c "source /opt/ros/humble/setup.bash && source $WS/setup.bash && \
    export PYTHONPATH=$PRUNING_ROOT/source/isaaclab_pruning:\$PYTHONPATH && \
    exec python3 -m pruning_sil.live_bringup --gate $WS/pruning_sil_cpp/lib/pruning_sil_cpp/safety_gate_node \
      --input-timeout-s 30 --config-timeout-s 1800 ${PANEL[*]}" > "$BRINGUP_LOG" 2>&1 &
BRINGUP=$!

export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PRUNING_ROOT PRUNING_BATCH_DIR PRUNING_RUN_INDEX SLURM_JOB_ID SLURM_ARRAY_JOB_ID SLURM_ARRAY_TASK_ID SLURMD_NODENAME OPENBLAS_NUM_THREADS MKL_NUM_THREADS ROS_DOMAIN_ID ROS_LOCALHOST_ONLY"
set +e
bhl_exec "$PRUNING_ROOT/hpc/inner/vision_robustness.sh"
status=$?
set -e
# The simulator sends an end marker on exit; give the ROS 2 side a moment to write its summary.
for _ in $(seq 60); do kill -0 $BRINGUP 2>/dev/null || break; sleep 1; done
kill $BRINGUP 2>/dev/null || true
wait $BRINGUP 2>/dev/null || true
rm -rf "$ROSLOG"
exit $status
