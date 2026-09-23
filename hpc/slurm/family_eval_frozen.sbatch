#!/bin/bash
# Frozen DA2 and six-view DINO scoring over every tree the plan registered.
# Runs after the render array with afterany, so a failed render does not
# leave this job pending forever: the missing tree is recorded as incomplete
# and the rest is scored. N never shrinks silently; render_status.json says
# which registered trees are present.
#SBATCH --job-name=prune-family-eval
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
BATCH_DIR="${1:?frozen batch directory}"
CODE="$BATCH_DIR/code"
OUT="$BATCH_DIR/eval"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2
export TORCH_HOME=/nfs/hpc/share/$USER/.cache/torch
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/tmp"
export MPLCONFIGDIR="$XDG_CACHE_HOME/matplotlib"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
mkdir "$OUT"
PY=/nfs/hpc/share/$USER/miniforge3/envs/depth-env/bin/python
"$PY" - "$BATCH_DIR" "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
batch, out = Path(sys.argv[1]), Path(sys.argv[2])
plan = json.loads((batch / "plan.json").read_text())
manifests, status = [], []
for tree in plan["trees"]:
    if tree["status"] == "rendered_pilot":
        path = Path(tree["render_manifest"])
    else:
        index = [t["tree_id"] for t in plan["trees"] if t["status"] == "to_render"].index(tree["tree_id"])
        path = batch / f"matrix_{index}_{tree['tree_id']}" / "render_manifest.json"
    present = path.is_file()
    ok = present and bool(json.loads(path.read_text()).get("ok"))
    status.append({"tree_id": tree["tree_id"], "family": tree["family"], "manifest": str(path),
                   "present": present, "render_ok": ok})
    if ok:
        manifests.append(str(path))
(out / "render_status.json").write_text(json.dumps({"registered": len(plan["trees"]), "evaluated": len(manifests),
                                                    "trees": status}, indent=2) + "\n")
(out / "inputs.json").write_text(json.dumps({"scope": plan["scope"], "render_manifests": manifests}, indent=2) + "\n")
print(json.dumps({"registered": len(plan["trees"]), "evaluated": len(manifests)}))
if not manifests:
    raise SystemExit("no registered tree rendered; nothing to score")
PYEOF
"$PY" "$CODE/tools/prepare_family_evaluation.py" --inputs "$OUT/inputs.json" --output "$OUT/plan.json"
DA2_SHA=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['models']['da2']['checkpoint_sha256'])" "$BATCH_DIR/plan.json")
DA2_CKPT=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1]))['models']['da2']['checkpoint'])" "$BATCH_DIR/plan.json")
"$PY" "$CODE/tools/depth_generalization.py" --manifest "$OUT/plan.json" \
 --companion /nfs/hpc/share/$USER/Computer_Vision \
 --checkpoint "$DA2_CKPT" --checkpoint-sha256 "$DA2_SHA" \
 --output "$OUT/da2" --save-predictions
exec "$PY" "$CODE/tools/dino_generalization.py" --companion /nfs/hpc/share/$USER/Computer_Vision \
 --da2-result "$OUT/da2/evaluation.json" --output "$OUT/dino"
