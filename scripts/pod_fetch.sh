#!/usr/bin/env bash
# Study E: copy from the main pod to this Mac (docs/DESIGN.md §18.1).
#   POD="-p <port> root@<host>" bash scripts/pod_fetch.sh records      # every job's records (small files only)
#   POD="-p <port> root@<host>" bash scripts/pod_fetch.sh checkpoints  # the 12 chosen + 3 weighted checkpoints, then SHA-256
#   POD="-p <port> root@<host>" bash scripts/pod_fetch.sh devicecheck <run id>   # GPU vs Mac CPU on the same V2 rows
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
    # '**/val/**/train-19.json' matches the same files in GNU rsync and macOS openrsync ('val/**.json' does not);
    # the copy is then checked against the pod's own list of validation records
    rsync -a -e "$SSH" --include='*/' --include='run.json' --include='progress.json' --include='loss_*.npy' \
      --include='*.log' --include='**/val/**/train-19.json' --include='jobs_*.json' --exclude='*' --prune-empty-dirs \
      "$addr:/root/naver/runs/e_runs/" runs/e_runs/
    pod_list=$($SSH "$addr" 'cd /root/naver/runs/e_runs && find . -path "*/val/*" -name train-19.json | sort')
    here_list=$(cd runs/e_runs && find . -path "*/val/*" -name train-19.json | sort)
    if [ "$pod_list" != "$here_list" ]; then echo "validation records differ between the pod and this Mac"; exit 1; fi
    echo "validation records: $(echo "$here_list" | grep -c . ) on both sides" ;;
  devicecheck)
    # one validated run (argument: its id): its train-18 checkpoint scored on train-19 on this Mac's CPU against the
    # pod's GPU predictions of the same rows (validation data only; the official test is not used)
    rid=$2
    sub=$($SSH "$addr" "cd /root/naver/runs/e_runs/$rid && ls -d val/model_after_train-18_*")
    mkdir -p "runs/e_runs/$rid/$sub"
    rsync -a -e "$SSH" "$addr:/root/naver/runs/e_runs/$rid/model_after_train-18.pt" "runs/e_runs/$rid/"
    rsync -a -e "$SSH" "$addr:/root/naver/runs/e_runs/$rid/$sub/train-19.npz" "runs/e_runs/$rid/$sub/"
    rm -rf "runs/e_local/devicecheck/$rid"
    uv run python -m cvr.pymal_run predict --device cpu --num_workers 0 --out "runs/e_local/devicecheck/$rid" \
      --ckpt "runs/e_runs/$rid/model_after_train-18.pt" train-19.parquet
    cpu=$(find "runs/e_local/devicecheck/$rid" -name train-19.npz | head -1)
    uv run python scripts/device_compare.py "runs/e_runs/$rid/$sub/train-19.npz" "$cpu" \
      "runs/e_local/devicecheck/$rid/comparison.json"
    ;;
  checkpoints)
    ids=$(python3 - <<'PY'
import json
c = json.load(open("runs/e_study/config_choice.json"))
fmt = lambda x: f"{x:g}"
print(" ".join(f"{m}_lr{fmt(c[m]['lr'])}_lam{fmt(c[m]['lamda'])}_s{s}" for m in ("MoAE", "BASE") for s in (2026, 2027, 2028)))
PY
)
    wids=$(echo "$ids" | tr ' ' '\n' | grep '^MoAE_' | sed 's/^MoAE_/MoAE-g1w2_/' | tr '\n' ' ')
    { for id in $ids; do for k in 17 19; do echo "$id/model_after_train-$k.pt"; done; done
      for id in $wids; do echo "$id/model_after_train-17.pt"; echo "$id/run.json"; done; } \
      | xargs -n 1 -P 6 sh -c 'mkdir -p "runs/e_runs/$(dirname "$2")" && rsync -a -e "$0" "$1:/root/naver/runs/e_runs/$2" "runs/e_runs/$2"' "$SSH" "$addr"
    # each path goes in as a positional argument ($2): macOS xargs -I{} cannot build the long weighted-run paths
    python3 - $ids $wids <<'PY'
import hashlib, json, sys
from pathlib import Path
bad = 0
for rid in sys.argv[1:]:
    rec = json.loads(Path(f"runs/e_runs/{rid}/run.json").read_text())["checkpoints"]
    for k in ((17,) if rid.startswith("MoAE-g1w2_") else (17, 19)):
        f = Path(f"runs/e_runs/{rid}/model_after_train-{k}.pt")
        h = hashlib.sha256(f.read_bytes()).hexdigest()
        ok = h == rec[f"train-{k}.parquet"]["sha256"]
        bad += not ok
        print(("OK  " if ok else "BAD ") + str(f))
sys.exit(1 if bad else 0)
PY
    ;;
  *) echo "records | checkpoints | devicecheck <run id>"; exit 1 ;;
esac
