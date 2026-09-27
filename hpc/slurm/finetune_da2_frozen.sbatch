#!/bin/bash
# One warm-start fine-tune arm of the DA2 metric head on an A40, from the
# frozen plan: data manifests, checkpoint hash, epochs and the arm come from
# plan.json, never from this file.
#SBATCH --job-name=prune-finetune-da2
#SBATCH --account=eecs
#SBATCH --partition=ampere
#SBATCH --constraint=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=48G
#SBATCH --time=08:00:00
set -euo pipefail
BATCH_DIR="${1:?frozen batch directory}"
CODE="$BATCH_DIR/code"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
export TORCH_HOME=/nfs/hpc/share/$USER/.cache/torch
export XDG_CACHE_HOME="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/cache"
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/$SLURM_JOB_ID}/tmp"
mkdir -p "$TMPDIR" "$XDG_CACHE_HOME"
PY=/nfs/hpc/share/$USER/miniforge3/envs/depth-env/bin/python
read -r ARM EPOCHS CKPT SHA SECONDS_LIMIT < <("$PY" - "$BATCH_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
batch = Path(sys.argv[1])
if not json.loads((batch / "storage_preflight.json").read_text())["ok"]:
    raise SystemExit("storage preflight did not pass")
plan = json.loads((batch / "plan.json").read_text())
print(plan.get("trainer_arm", plan["arm"]), plan["epochs"], plan["warm_start"]["checkpoint"],
      plan["warm_start"]["checkpoint_sha256"], plan["max_runtime_seconds"])
PYEOF
)
DESCRIPTION=$("$PY" -c "import json,sys;print(json.load(open(sys.argv[1])).get('data_description','6,270 surviving frames (box and box_cam1-4, bark_brown_02) instead of 68,400'))" "$BATCH_DIR/plan.json")
# A rendered arm builds its training manifest here, after the renders it waited
# for, from the batches its plan registered; zero verified rendered rows is a
# failure, never a silent fall-back to the surviving frames alone.
RENDERED=$("$PY" -c "import json,sys;print(' '.join('--batch '+b for b in (json.load(open(sys.argv[1])).get('rendered_batches') or [])))" "$BATCH_DIR/plan.json")
if [ -n "$RENDERED" ]; then
  "$PY" "$CODE/tools/build_training_lighting_manifest.py" $RENDERED \
    --surviving-manifest /nfs/hpc/share/$USER/Computer_Vision/manifests/full_spur_2tex_all_3view_seed1/spur_train.csv \
    --output "$BATCH_DIR/manifests/rendered"
  "$PY" - "$BATCH_DIR" <<'PYEOF'
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]) / "code/tools"))
from finetune_da2 import filter_manifest, sha256
batch = Path(sys.argv[1])
summary = json.loads((batch / "manifests/rendered/summary.json").read_text())
if summary["kept_frames"] == 0:
    raise SystemExit("no rendered frame passed the geometry check; refusing to train on the surviving rows alone")
report = filter_manifest(batch / "manifests/rendered/combined.csv", batch / "manifests/train.csv")
report["sha256"] = sha256(batch / "manifests/train.csv")
report["rendered_summary_sha256"] = sha256(batch / "manifests/rendered/summary.json")
(batch / "manifests/train_built_in_job.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report))
PYEOF
fi
exec "$PY" "$CODE/tools/finetune_da2.py" --companion /nfs/hpc/share/$USER/Computer_Vision \
 --checkpoint "$CKPT" --checkpoint-sha256 "$SHA" --arm "$ARM" --epochs "$EPOCHS" \
 --train-manifest "$BATCH_DIR/manifests/train.csv" --val-manifest "$BATCH_DIR/manifests/val.csv" \
 --extra-val-manifest "$BATCH_DIR/manifests/family_matrix.csv" \
 --output "$BATCH_DIR/train" --max-runtime-seconds "$SECONDS_LIMIT" --num-workers 4 \
 --data-description "$DESCRIPTION"
