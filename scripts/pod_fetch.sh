#!/usr/bin/env bash
# Study E: copy from the main pod to this Mac (docs/DESIGN.md §18.1).
#   POD="-p <port> root@<host>" bash scripts/pod_fetch.sh records      # every job's records (small files only)
#   POD="-p <port> root@<host>" bash scripts/pod_fetch.sh checkpoints  # the 12 chosen checkpoints, then their SHA-256
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
    # '**/val/**/train-19.json' means the same in GNU rsync and macOS openrsync ('val/**.json' did not: review);
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
    uv run python - "$rid" "$sub" <<'PY'
import json, sys
from pathlib import Path
import numpy as np
rid, sub = sys.argv[1:]
gpu = np.load(Path("runs/e_runs") / rid / sub / "train-19.npz")
cpu = np.load(next(Path("runs/e_local/devicecheck", rid).rglob("train-19.npz")))
same_rows = np.array_equal(gpu["user"], cpu["user"]) and np.array_equal(gpu["y_last"], cpu["y_last"])
d = {k: float(np.abs(gpu[k].astype(np.float64) - cpu[k].astype(np.float64)).max()) for k in gpu.files if k.startswith("p_")}
rec = {"run": rid, "rows": int(gpu["p_last"].size), "same_rows_and_labels": bool(same_rows), "max_abs_diff": d}
Path("runs/e_local/devicecheck", rid, "comparison.json").write_text(json.dumps(rec, indent=2))
print(json.dumps(rec))
sys.exit(0 if same_rows and max(d.values()) <= 1e-5 else 1)
PY
    ;;
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
  *) echo "records | checkpoints | devicecheck <run id>"; exit 1 ;;
esac
