#!/bin/bash
# Inside the SIL container: build the ROS 2 packages in node-local scratch, run their
# tests, then drive the C++ safety gate with recorded cycles and faults.
#   $1  repository checkout (read-only use)
#   $2  cycles_all.jsonl.gz from tools/export_gate_parity.py ("-" to skip the harness)
#   $3  evidence output path for tools/validate_safety_gate.py
#   $4  optional: "panel" to also build and test ros2/pruning_operator_panel
# Optional: REHEARSE_CAPTURE=<capture dir> REHEARSE_OUT=<json> runs tools/rehearse_ros2_loop.py.
set -eo pipefail
REPO=$1; CYCLES=$2; OUT=$3; WITH_PANEL=${4:-}
WS=$(mktemp -d -p /tmp pruning-ws.XXXXXX)
trap 'rm -rf "$WS"' EXIT
mkdir -p $WS/src
PACKAGES="pruning_sil_msgs pruning_sil_cpp pruning_sil"
[ "$WITH_PANEL" = panel ] && PACKAGES="$PACKAGES pruning_operator_panel"
for package in $PACKAGES; do cp -r $REPO/ros2/$package $WS/src/; done
cd $WS
source /opt/ros/humble/setup.bash
echo "== build ($PACKAGES) $(date -Is)"
colcon build --event-handlers console_direct- --cmake-args -DCMAKE_BUILD_TYPE=RelWithDebInfo 2>&1 | grep -E "warning:|error:|Finished|Failed|Summary" || true
test -x install/pruning_sil_cpp/lib/pruning_sil_cpp/safety_gate_node
source install/setup.bash
export PYTHONPATH="$REPO/source/isaaclab_pruning:$REPO/ros2/pruning_sil:$PYTHONPATH"
echo "== pytest ros2/pruning_sil/test $(date -Is)"
python3 -m pytest -q -p no:cacheprovider $REPO/ros2/pruning_sil/test 2>&1 | tail -15
echo "== colcon test $(date -Is)"
# pruning_sil's pytest suite ran above; colcon runs the C++ (and panel) gtests.
colcon test --packages-skip pruning_sil --event-handlers console_direct- >/dev/null 2>&1 || true
colcon test-result --verbose 2>&1 | tail -25
grep -h "\[parity\]" build/pruning_sil_cpp/test_results/pruning_sil_cpp/*.xml build/pruning_sil_cpp/Testing/Temporary/LastTest.log 2>/dev/null | head -3 || true
colcon test-result >/dev/null 2>&1 || { echo "colcon tests failed"; exit 3; }
if [ "$CYCLES" != "-" ]; then
  echo "== safety gate harness $(date -Is)"
  python3 $REPO/tools/validate_safety_gate.py --cycles "$CYCLES" \
    --gate $WS/install/pruning_sil_cpp/lib/pruning_sil_cpp/safety_gate_node --output "$OUT" 2>&1 | tail -60
fi
if [ -n "${REHEARSE_CAPTURE:-}" ]; then
  echo "== live-loop rehearsal on $REHEARSE_CAPTURE $(date -Is)"
  PANEL_ARGS=()
  [ "$WITH_PANEL" = panel ] && PANEL_ARGS=(--panel $WS/install/pruning_operator_panel/lib/pruning_operator_panel/operator_panel)
  python3 $REPO/tools/rehearse_ros2_loop.py --capture "$REHEARSE_CAPTURE" \
    --gate $WS/install/pruning_sil_cpp/lib/pruning_sil_cpp/safety_gate_node --output "$REHEARSE_OUT" "${PANEL_ARGS[@]}" 2>&1 | tail -60
fi
echo "== done $(date -Is)"
