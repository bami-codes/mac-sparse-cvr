#!/usr/bin/env bash
# Study E from the end of the grid to the V2 dry run, unattended (docs/DESIGN.md §18). Run on the Mac:
#   POD="-p <port> root@<host>" POD_ID=<runpod id> bash scripts/e_autopilot.sh
# Every step is skipped when its output exists, so the script can be run again after a stop. It stops at the first
# failure (set -e) and never runs the official test: that is done by hand after the freeze and the dry run are
# reviewed. The pod is terminated only after all 15 checkpoints are copied and match their run records.
set -euo pipefail
: "${POD:?}" "${POD_ID:?}"
port=$(echo "$POD" | awk '{print $2}'); addr=$(echo "$POD" | awk '{print $3}')
SSHOPT="-i $HOME/.ssh/runpod_naver_ed25519 -p $port -o UserKnownHostsFile=$HOME/.ssh/known_hosts_runpod -o ServerAliveInterval=30"
on_pod() { ssh $SSHOPT "$addr" "$@"; }
log() { echo "$(date '+%m-%d %H:%M:%S') $*"; }
left_on_pod() { on_pod "cd /root/naver && .venv/bin/python scripts/pod_left.py $1"; }
wait_jobs() {  # $1 job file on the pod, $2 the number of jobs it must hold
  while true; do
    read -r n total state < <(left_on_pod "$1")
    log "$1: $n of $total left, queue $state"
    if [ "$n" != missing ] && [ "$total" != "$2" ]; then log "expected $2 jobs in $1"; return 1; fi
    [ "$n" = 0 ] && return 0
    if [ "$state" = down ]; then log "the queue is not running and $n jobs are left"; return 1; fi
    sleep 600
  done
}

rsync -a -e "ssh $SSHOPT" scripts/pod_left.py "$addr:/root/naver/scripts/"  # new file only; the grid code stays

log "1. wait for the grid"
wait_jobs runs/e_runs/jobs_grid.json 45

log "2. records and the complete-grid choice"
bash scripts/pod_fetch.sh records
python3 scripts/pod_queue.py summary runs/e_runs/jobs_grid.json > runs/e_runs/summary_grid.jsonl
[ -f runs/e_study/config_choice.json ] || uv run python -m cvr.final_eval_e choose runs/e_runs/summary_grid.jsonl
read -r MLR MLAM BLR < <(python3 -c "import json; c=json.load(open('runs/e_study/config_choice.json')); print(c['MoAE']['lr'], c['MoAE']['lamda'], c['BASE']['lr'])")
log "chosen: MoAE lr $MLR lambda $MLAM, BASE lr $BLR"

log "3. seeds and the weighted runs on the pod"
if ! on_pod "test -f /root/naver/runs/e_runs/jobs_seeds.json"; then
  rsync -az --exclude ".secrets/" --exclude ".venv/" --exclude "data/" --exclude "runs/" --exclude "__pycache__/" \
    --exclude ".pytest_cache/" --exclude ".ruff_cache/" --exclude ".DS_Store" -e "ssh $SSHOPT" ./ "$addr:/root/naver/"
  on_pod "mkdir -p /root/naver/runs/e_study"
  rsync -a -e "ssh $SSHOPT" runs/e_study/group_lists.json runs/e_study/groups.json "$addr:/root/naver/runs/e_study/"
  on_pod "chown -R root:root /root/naver && cd /root/naver && (setsid nohup bash scripts/pod_after_grid.sh $MLR $MLAM $BLR \
          > runs/e_after.log 2>&1 < /dev/null &) ; sleep 20; tail -2 runs/e_after.log"
fi
sleep 60
wait_jobs runs/e_runs/jobs_seeds.json 7

log "4. records and the 15 checkpoints, then the pod is terminated"
bash scripts/pod_fetch.sh records
rsync -a -e "ssh $SSHOPT" "$addr:/root/naver/runs/e_runs/jobs_seeds.json" runs/e_runs/
python3 scripts/pod_queue.py summary runs/e_runs/jobs_seeds.json > runs/e_runs/summary_seeds.jsonl
bash scripts/pod_fetch.sh checkpoints
python3 scripts/runpod.py terminate "$POD_ID"

log "5. V1/V2 predictions of the nine research checkpoints on this Mac"
ids=$(python3 - <<'PY'
import json
c = json.load(open("runs/e_study/config_choice.json"))
print(" ".join(f"{m}_lr{c[m]['lr']:g}_lam{c[m]['lamda']:g}_s{s}" for m in ("MoAE", "BASE", "MoAE-g1w2") for s in (2026, 2027, 2028)))
PY
)
todo=()
for id in $ids; do
  sha=$(python3 -c "import json,sys; print(json.load(open('runs/e_runs/$id/run.json'))['checkpoints']['train-17.parquet']['sha256'][:12])")
  [ -f "runs/e_runs/research/model_after_train-17_$sha/train-19.npz" ] || todo+=(--ckpt "runs/e_runs/$id/model_after_train-17.pt")
done
if [ ${#todo[@]} -gt 0 ]; then
  uv run python -m cvr.pymal_run predict --device cpu --num_workers 0 --out runs/e_runs/research "${todo[@]}" \
    train-18.parquet train-19.parquet
fi

log "6. calibration on V1, choice on V2, freeze, dry run"
[ -f runs/e_study/q2/selection.json ] || uv run python -m cvr.study_e_eval calibrate runs/e_study/config_choice.json
[ -f runs/e_study/frozen/freeze_q2.json ] || uv run python -m cvr.final_eval_e freeze
uv run python -m cvr.final_eval_e dryrun
log "done: freeze and dry run ready for review; the official test is not run"
