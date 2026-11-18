#!/usr/bin/env bash
# Study E pilot on a GPU pod (docs/DESIGN.md §18.7): sizes and speed only, no model metric is computed.
# Run from the repository root on the pod after copying the repository there (data/ and runs/ are not copied).
#
#   bash scripts/pod_pilot.sh            # writes runs/e_pilot/{env.txt,preprocess.json,BASE/,MoAE/}
#
# 1. environment: uv, the locked dependencies, the GPU seen by PyTorch, the test suites on Linux
# 2. data: the official vocabularies and one training file, verified against the Hugging Face API
# 3. preprocessing of that file with the official functions: peak RAM, cache size, time
# 4. training BASE and MoAE on that file only, no checkpoint and no prediction: VRAM after the first Adam step
#    with the full official vocabularies loaded, peak VRAM, rows per second
set -euo pipefail
OUT=runs/e_pilot
mkdir -p "$OUT"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
{
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  free -g; df -h .; nproc
  uv run python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
} | tee "$OUT/env.txt"
uv run pytest -q | tail -2 | tee -a "$OUT/env.txt"
uv run pytest -m e2e -q -rs | tail -3 | tee -a "$OUT/env.txt"   # study C's latency step skips off macOS

python3 scripts/fetch_mac.py vocabs train/train-00.parquet
uv run python -m cvr.pymal_run preprocess train-00.parquet
cp "$(ls -t data/pymal_cache/preprocess_*.json | head -1)" "$OUT/preprocess.json"

uv run python -m cvr.pymal_run train --out "$OUT/BASE" --model BASE --lr 0.003 --lamda 0.3 \
  --train_files train-00.parquet --ckpt_after
uv run python -m cvr.pymal_run train --out "$OUT/MoAE" --model MoAE --lr 0.003 --lamda 0.3 \
  --aux_views first mta linear cartesian --train_files train-00.parquet --ckpt_after
echo "pilot done: $OUT"
