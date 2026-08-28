#!/bin/bash
set -uo pipefail
cd "${OV_CACHE:-/tmp}"
exec "$PY" "$PRUNING_ROOT/hpc/inner/import_urdf.py"
