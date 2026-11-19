#!/usr/bin/env bash
# Set up the study E GPU pod: environment, every official data file, the official preprocessing.
#   bash scripts/pod_setup.sh        # then: python scripts/pod_queue.py run runs/e_runs/jobs_grid.json --gpus 4
set -euo pipefail
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv
uv run python -c "import torch; assert torch.cuda.is_available(), 'CUDA 13 wheels need driver >= 580'; print(torch.cuda.device_count(), 'GPUs')"
mkdir -p runs/e_runs
# downloads: four files at a time, each verified against the Hugging Face API
python3 scripts/fetch_mac.py vocabs
{ for i in $(seq -w 0 19); do echo "train/train-$i.parquet"; done; echo test/test.parquet; } \
  | xargs -P 4 -I{} python3 scripts/fetch_mac.py {}
# official preprocessing: six files at a time (about 23 GB of memory each)
{ for i in $(seq -w 0 19); do echo "train-$i.parquet"; done; echo test.parquet; } \
  | xargs -P 6 -I{} uv run python -m cvr.pymal_run preprocess {}
ls data/pymal_cache/*.pkl | wc -l
uv run python scripts/pod_queue.py grid > runs/e_runs/jobs_grid.json
echo "setup done"
