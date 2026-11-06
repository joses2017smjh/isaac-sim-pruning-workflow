#!/bin/bash
# Inside the Qt SIL image: build the operator panel and capture it in review mode on a
# recorded live-loop run, at wide, medium and compact sizes.
#   $1 repository  $2 run directory  $3 output directory  $4 decision index (-1: last)
set -eo pipefail
REPO=$1; RUN=$2; OUT=$3; INDEX=${4:-80}
WS=$(mktemp -d -p /tmp panel-shots.XXXXXX)
trap 'rm -rf "$WS"' EXIT
mkdir -p $WS/src $OUT
cp -r $REPO/ros2/pruning_sil_msgs $REPO/ros2/pruning_operator_panel $WS/src/
cd $WS
source /opt/ros/humble/setup.bash
colcon build --event-handlers console_direct- --cmake-args -DBUILD_TESTING=OFF 2>&1 | grep -E "error|Finished|Failed" || true
source install/setup.bash
PANEL=install/pruning_operator_panel/lib/pruning_operator_panel/operator_panel
export QT_QPA_PLATFORM=offscreen QT_QUICK_BACKEND=software
$PANEL --review "$RUN" --review-index $INDEX --size 1440x860 --screenshot-all-tabs --screenshot-dir $OUT/wide
$PANEL --review "$RUN" --review-index $INDEX --size 960x720 --screenshot-dir $OUT/medium
$PANEL --review "$RUN" --review-index $INDEX --size 420x860 --screenshot-dir $OUT/compact
find $OUT -name "*.png" | sort
