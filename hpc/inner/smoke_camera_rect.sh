#!/bin/bash
set -euo pipefail
cd "${OV_CACHE:-/tmp}"
exec "$PY" "$PRUNING_ROOT/hpc/inner/smoke_camera_rect.py"
