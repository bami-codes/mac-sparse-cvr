#!/usr/bin/env bash
# Study E: copy from the main pod to this Mac (docs/DESIGN.md §18.1).
#   POD="-p <port> root@<host>" bash scripts/pod_fetch.sh records      # every job's records (small files only)
#   POD="-p <port> root@<host>" bash scripts/pod_fetch.sh checkpoints  # the 12 chosen checkpoints, then their SHA-256
# The chosen runs are read from runs/e_study/config_choice.json (both models, seeds 2026-2028); for each run the
# train-17 checkpoint (Q2) and the train-19 checkpoint (Q1) are copied six at a time (one connection to the pod is
# slow) and checked against the run record before the pod is terminated.
set -euo pipefail
: "${POD:?set POD to the ssh port and address, e.g. POD='-p 22138 root@69.30.85.238'}"
port=$(echo "$POD" | awk '{print $2}'); addr=$(echo "$POD" | awk '{print $3}')
SSH="ssh -i $HOME/.ssh/runpod_naver_ed25519 -p $port -o UserKnownHostsFile=$HOME/.ssh/known_hosts_runpod"
mkdir -p runs/e_runs
case "$1" in
  records)
    rsync -a -e "$SSH" --include='*/' --include='run.json' --include='progress.json' --include='loss_*.npy' \
      --include='*.log' --include='val/**.json' --include='jobs_*.json' --exclude='*' --prune-empty-dirs \
      "$addr:/root/naver/runs/e_runs/" runs/e_runs/ ;;
  checkpoints)
    ids=$(python3 - <<'PY'
import json
c = json.load(open("runs/e_study/config_choice.json"))
fmt = lambda x: f"{x:g}"
print(" ".join(f"{m}_lr{fmt(c[m]['lr'])}_lam{fmt(c[m]['lamda'])}_s{s}" for m in ("MoAE", "BASE") for s in (2026, 2027, 2028)))
PY
)
    for id in $ids; do for k in 17 19; do echo "$id/model_after_train-$k.pt"; done; done \
      | xargs -P 6 -I{} rsync -a -e "$SSH" "$addr:/root/naver/runs/e_runs/{}" "runs/e_runs/{}"
    python3 - $ids <<'PY'
import hashlib, json, sys
from pathlib import Path
bad = 0
for rid in sys.argv[1:]:
    rec = json.loads(Path(f"runs/e_runs/{rid}/run.json").read_text())["checkpoints"]
    for k in (17, 19):
        f = Path(f"runs/e_runs/{rid}/model_after_train-{k}.pt")
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        ok = h == rec[f"train-{k}.parquet"]["sha256"]
        bad += not ok
        print(("OK  " if ok else "BAD ") + str(f))
sys.exit(1 if bad else 0)
PY
    ;;
  *) echo "records | checkpoints"; exit 1 ;;
esac
