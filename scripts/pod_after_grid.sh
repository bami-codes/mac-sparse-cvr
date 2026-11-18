#!/usr/bin/env bash
# Study E after the grid (docs/DESIGN.md §18.10, amendment 2), on the main pod, with the settings chosen locally:
#   bash scripts/pod_after_grid.sh <MoAE lr> <MoAE lambda> <BASE lr>
# Seeds 2027 and 2028 of both chosen settings and the three weighted MoAE runs (seven runs on four GPUs). Nothing
# else runs on the pod: the chosen
# checkpoints are copied to the Mac (scripts/pod_fetch.sh) and every evaluation prediction is made there.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# with amendment 3's three weighted MoAE runs (Q4); runs/e_study/group_lists.json is copied to the pod beforehand
test -f runs/e_study/group_lists.json
uv run python scripts/pod_queue.py seeds --moae "$1 $2" --base "$3" --with-weighted > runs/e_runs/jobs_seeds.json
uv run python scripts/pod_queue.py run runs/e_runs/jobs_seeds.json --gpus 4
uv run python scripts/pod_queue.py summary runs/e_runs/jobs_seeds.json
echo "seeds done"
