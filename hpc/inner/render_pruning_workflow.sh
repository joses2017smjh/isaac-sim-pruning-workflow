#!/bin/bash
set -euo pipefail
cd "$PRUNING_ROOT"
# Resolve the promoted asset explicitly so both report paths retain provenance.
readarray -t asset_fields < <("$PY" -c 'from isaaclab_pruning.robot import load_ur5e_pruner_config, repository_root; c=load_ur5e_pruner_config()["usd"]; print(c["asset_id"]); print(repository_root()/c["relative_path"]); print(repository_root()/c["import_evidence"])')
export PRUNING_ASSET_ID="${asset_fields[0]}"
export PRUNING_USD="${asset_fields[1]}"
export PRUNING_USD_EVIDENCE="${asset_fields[2]}"
"$PY" tools/check_isaac_render_stack.py \
    --lock docs/ENVIRONMENT.lock.json --output "$PRUNING_RENDER_DIR/preflight.json"

# Diagnose control and contacts in the same allocation. A failed smoke remains
# failed; it does not prohibit a separately labeled rendered demonstration.
if [[ "${PRUNING_RUN_ENV_SMOKE:-1}" == 1 ]]; then
    "$PY" hpc/inner/smoke_env.py || echo "Environment smoke failed; preserving its evidence."
fi
if [[ "${PRUNING_SMOKE_ONLY:-0}" == 1 ]]; then
    "$PY" -c 'import json,os,sys; r=json.load(open(os.environ["BENCH_OUT"])); print("Environment smoke ok:",r.get("ok")); sys.exit(0 if r.get("ok") is True else 1)'
    exit 0
fi
"$PY" hpc/inner/render_pruning_workflow.py

# Kit can close the interpreter with exit 0 after an application error.
# This independent process validates fresh output instead of trusting that exit.
"$PY" tools/validate_isaac_render.py "$PRUNING_RENDER_DIR"
