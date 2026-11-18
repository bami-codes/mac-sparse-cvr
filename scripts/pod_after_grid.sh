#!/usr/bin/env bash
# Study E after the grid (docs/DESIGN.md §18.10, amendment 2), on the main pod, with the settings chosen locally:
#   bash scripts/pod_after_grid.sh <MoAE lr> <MoAE lambda> <BASE lr>
# 1. seeds 2027 and 2028 of both chosen settings (four runs, one per GPU)
# 2. predictions of the six research checkpoints (train-17 = trained on train-00..17) on V1 = train-18 and V2 = train-19,
#    one folder per checkpoint under runs/e_runs/research (found later through each checkpoint's SHA-256)
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
MOAE_LR=$1 MOAE_LAM=$2 BASE_LR=$3
uv run python scripts/pod_queue.py seeds --moae "$MOAE_LR $MOAE_LAM" --base "$BASE_LR" > runs/e_runs/jobs_seeds.json
uv run python scripts/pod_queue.py run runs/e_runs/jobs_seeds.json --gpus 4
ids=$(uv run python - "$MOAE_LR" "$MOAE_LAM" "$BASE_LR" <<'PY'
import sys
from cvr.study_e_eval import run_id
m_lr, m_lam, b_lr = map(float, sys.argv[1:])
print(" ".join(run_id(m, lr, lam, s) for m, lr, lam in (("MoAE", m_lr, m_lam), ("BASE", b_lr, 0.3)) for s in (2026, 2027, 2028)))
PY
)
args=()
for id in $ids; do args+=(--ckpt "runs/e_runs/$id/model_after_train-17.pt"); done
uv run python -m cvr.pymal_run predict --out runs/e_runs/research "${args[@]}" train-18.parquet train-19.parquet
echo "after grid done: $ids"
