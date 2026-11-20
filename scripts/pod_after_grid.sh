#!/usr/bin/env bash
# After the grid, on the GPU machine, with the settings chosen on V2:
#   bash scripts/pod_after_grid.sh <MoAE lr> <MoAE lambda> <BASE lr>
# Seeds 2027 and 2028 of both chosen settings and the three weighted MoAE runs (seven runs on four GPUs). Nothing
# else runs on the pod: the checkpoints are copied to the local machine (scripts/pod_fetch.sh) and every evaluation
# prediction is made there.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
# the weighted runs need the fixed group lists, copied to the pod beforehand
test -f runs/eval/group_lists.json
uv run python scripts/pod_queue.py seeds --moae "$1 $2" --base "$3" --with-weighted > runs/train/jobs_seeds.json
uv run python scripts/pod_queue.py run runs/train/jobs_seeds.json --gpus 4
uv run python scripts/pod_queue.py summary runs/train/jobs_seeds.json
echo "seeds done"
