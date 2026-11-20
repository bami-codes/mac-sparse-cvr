#!/usr/bin/env bash
# Locally, after the 15 checkpoints are here (the pod may be gone): verify them, make each missing
# V1/V2 prediction one checkpoint and one file at a time, then calibrate, freeze and dry-run (docs/DESIGN.md §18).
#   bash scripts/calibrate_and_freeze.sh
# The continuation of scripts/pod_autopilot.sh once the pod is terminated (that script starts by reaching the pod).
# Every step is skipped when its output exists; the official test is never run here.
set -euo pipefail
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }

log "1. the 15 local checkpoints against their run records"
python3 - <<'PY'
import hashlib, json, sys
from pathlib import Path
c = json.load(open("runs/eval/config_choice.json"))
bad = 0
for m in ("MoAE", "BASE", "MoAE-g1w2"):
    for s in (2026, 2027, 2028):
        rid = f"{m}_lr{c[m]['lr']:g}_lam{c[m]['lamda']:g}_s{s}"
        rec = json.loads(Path(f"runs/train/{rid}/run.json").read_text())["checkpoints"]
        for k in ((17,) if m == "MoAE-g1w2" else (17, 19)):
            f = Path(f"runs/train/{rid}/model_after_train-{k}.pt")
            ok = f.exists() and hashlib.sha256(f.read_bytes()).hexdigest() == rec[f"train-{k}.parquet"]["sha256"]
            bad += not ok
            print(("OK  " if ok else "BAD ") + str(f))
sys.exit(1 if bad else 0)
PY

log "2. missing V1/V2 predictions, one checkpoint and one file at a time"
python3 - <<'PY' > runs/eval/missing_predictions.txt
import json
from pathlib import Path
c = json.load(open("runs/eval/config_choice.json"))
for m in ("MoAE", "BASE", "MoAE-g1w2"):
    for s in (2026, 2027, 2028):
        rid = f"{m}_lr{c[m]['lr']:g}_lam{c[m]['lamda']:g}_s{s}"
        sha = json.loads(Path(f"runs/train/{rid}/run.json").read_text())["checkpoints"]["train-17.parquet"]["sha256"]
        for name in ("train-18", "train-19"):
            d = Path(f"runs/train/research/model_after_train-17_{sha[:12]}")
            if not ((d / f"{name}.npz").exists() and (d / f"{name}.json").exists()):
                print(rid, f"{name}.parquet")
PY
while read -r rid file; do
  d=$(python3 -c "import json; print(json.load(open('runs/train/$rid/run.json'))['checkpoints']['train-17.parquet']['sha256'][:12])")
  rm -f "runs/train/research/model_after_train-17_$d/${file%.parquet}.npz"   # an interrupted write leaves no json
  log "predict $rid $file"
  uv run python -m cvr.pymal_run predict --device cpu --num_workers 0 --out runs/train/research \
    --ckpt "runs/train/$rid/model_after_train-17.pt" "$file"
done < runs/eval/missing_predictions.txt

log "3. calibration on V1, choice on V2, freeze, dry run"
[ -f runs/eval/q2/selection.json ] || uv run python -m cvr.evaluation calibrate runs/eval/config_choice.json
[ -f runs/eval/frozen/freeze_q2.json ] || uv run python -m cvr.final_test freeze
uv run python -m cvr.final_test dryrun
log "done: freeze and dry run ready for review; the official test is not run"
