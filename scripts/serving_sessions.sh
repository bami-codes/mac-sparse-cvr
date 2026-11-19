#!/usr/bin/env bash
# Latency protocol v3 on one machine (docs/DESIGN.md §18.12.1): the consistency check, then sessions 20 minutes apart
# until three are valid, at most six attempts per machine in all, then the summary. Nothing else of this repository may
# run meanwhile. A restart continues after the sessions already recorded (none is measured again or overwritten), and
# the script fails unless the machine ends with three valid sessions.
#   bash scripts/serving_sessions.sh mac      # on the Mac
#   bash scripts/serving_sessions.sh linux    # on the Linux CPU server
set -euo pipefail
name=$1
gap=${GAP_S:-1200}
dir=runs/e_study/serving_v3/sessions
export PYTHONPATH=.
mkdir -p "$dir"
count() {  # "<sessions recorded> <valid sessions>" of this machine (no ls or grep failure when there is none yet)
  local n=0 v=0 f
  for f in "$dir/${name}"_*.json; do
    [ -e "$f" ] || continue
    n=$((n + 1))
    if grep -q '"valid": true' "$f"; then v=$((v + 1)); fi
  done
  echo "$n $v"
}
[ -f "runs/e_study/serving_v3/check_${name}.json" ] || uv run python scripts/serving_e.py check "$name" > /dev/null
echo "$(date '+%H:%M:%S') check $name passed"
read -r done valid < <(count)
while [ "$valid" -lt 3 ] && [ "$done" -lt 6 ]; do
  if [ "$done" -gt 0 ]; then sleep "$gap"; fi
  out=$(uv run python scripts/serving_e.py session "${name}_$((done + 1))")
  echo "$(date '+%H:%M:%S') $(echo "$out" | tr -d '\n ')"
  before=$done
  read -r done valid < <(count)
  if [ "$done" -le "$before" ]; then echo "session ${name}_$((before + 1)) left no record"; exit 1; fi
done
uv run python scripts/serving_e.py summary > /dev/null
echo "$(date '+%H:%M:%S') $valid valid of $done sessions; summary written"
[ "$valid" -ge 3 ]
