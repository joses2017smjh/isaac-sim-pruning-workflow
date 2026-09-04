#!/bin/bash
#SBATCH --job-name=prune-urdf-import
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/hpc/share/sanchej7/isaac-sim-pruning-workflow/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/isaac-sim-pruning-workflow/logs/%x-%j.out

# Content-addressed GPU import. The caller must select one generated asset and
# its matching provenance explicitly; there are intentionally no stale input
# defaults. The Python preflight writes pending evidence before Kit starts.

set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export PRUNING_ROOT=/nfs/hpc/share/$USER/isaac-sim-pruning-workflow
: "${PRUNING_ASSET_ID:?set PRUNING_ASSET_ID to the generated provenance asset_id}"
: "${PRUNING_URDF:?set PRUNING_URDF to the generated absolute URDF path}"
: "${PRUNING_URDF_PROVENANCE:?set PRUNING_URDF_PROVENANCE to its generation evidence JSON}"
: "${SLURM_JOB_ID:?this importer must run as a SLURM job}"

export PRUNING_ASSET_ID PRUNING_URDF PRUNING_URDF_PROVENANCE
export BENCH_OUT=${BENCH_OUT:-$PRUNING_ROOT/docs/evidence/urdf_import_${SLURM_JOB_ID}.json}
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PRUNING_ROOT PRUNING_ASSET_ID PRUNING_URDF PRUNING_URDF_PROVENANCE BENCH_OUT SLURM_JOB_ID SLURM_JOB_NAME SLURM_JOB_PARTITION SLURMD_NODENAME"
mkdir -p "$PRUNING_ROOT/logs" "$PRUNING_ROOT/docs/evidence"

if [[ ! -f "$PRUNING_URDF" ]]; then
    echo "missing generated absolute URDF: $PRUNING_URDF"
    exit 1
fi
if [[ ! -f "$PRUNING_URDF_PROVENANCE" ]]; then
    echo "missing URDF generation provenance: $PRUNING_URDF_PROVENANCE"
    exit 1
fi

echo "=== asset: $PRUNING_ASSET_ID ==="
echo "=== urdf: $PRUNING_URDF ==="
echo "=== provenance: $PRUNING_URDF_PROVENANCE ==="
echo "=== evidence: $BENCH_OUT ==="
echo "=== node: $(hostname) gpu=$(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | head -1) | $(date) ==="
bhl_exec "$PRUNING_ROOT/hpc/inner/import_urdf.sh"

# Apptainer/Kit shutdown has historically obscured an inner non-zero exit code
# on this cluster.  Treat the job-specific evidence as the application result
# and fail the Slurm allocation unless every import gate is explicitly green.
# This keeps a scheduler-level COMPLETED state from being mistaken for a valid
# robot asset (as happened for diagnostic job 21125352).
/usr/bin/python3 - "$BENCH_OUT" "$PRUNING_ROOT" <<'PY'
import json
import sys
from pathlib import Path

report_path = Path(sys.argv[1])
repository_root = Path(sys.argv[2])
if not report_path.is_file():
    raise SystemExit(f"import postflight: missing evidence: {report_path}")

try:
    report = json.loads(report_path.read_text(encoding="utf-8"))
except (OSError, json.JSONDecodeError) as error:
    raise SystemExit(f"import postflight: unreadable evidence: {error}") from error

required = {
    "status": report.get("status") == "complete",
    "ok": report.get("ok") is True,
    "imported": report.get("imported") is True,
    "stage_validation": isinstance(report.get("stage_validation"), dict)
    and report["stage_validation"].get("ok") is True,
}
failed = [name for name, passed in required.items() if not passed]
if failed:
    reason = report.get("reason", "no failure reason recorded")
    raise SystemExit(f"import postflight: failed gates {failed}: {reason}")

root_value = report.get("output", {}).get("root_layer")
if not isinstance(root_value, str) or not root_value:
    raise SystemExit("import postflight: output.root_layer is missing")
root_layer = Path(root_value)
if not root_layer.is_absolute():
    root_layer = repository_root / root_layer
if not root_layer.is_file():
    raise SystemExit(f"import postflight: root layer is missing: {root_layer}")

print(f"import postflight: PASS ({root_layer})")
PY
echo "=== done: $(date) ==="
