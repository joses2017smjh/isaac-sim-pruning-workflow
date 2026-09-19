#!/bin/bash
set -euo pipefail
cd "$PRUNING_ROOT"
exec "$PY" tools/run_vision_experiment.py --batch-dir "$PRUNING_BATCH_DIR" --index "$PRUNING_RUN_INDEX"
