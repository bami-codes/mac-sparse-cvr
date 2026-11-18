#!/usr/bin/env bash
# Study E pilot on a GPU pod (docs/DESIGN.md §18.7): sizes and speed only, no model metric is computed.
# Run from the repository root on the pod after copying the repository there (data/ and runs/ are not copied).
#
#   bash scripts/pod_pilot.sh            # writes runs/e_pilot/{env.txt,tests.txt,preprocess.json,BASE/,MoAE/,predict/}
#
# 1. environment: uv, the locked dependencies, and a GPU that PyTorch can use. The lock installs PyTorch 2.15.0
#    with CUDA 13 wheels on Linux, which need an NVIDIA driver >= 580; the pilot stops here if CUDA is unusable.
# 2. the test suites on Linux, recorded in tests.txt (a failure is reported but does not stop the measurements)
# 3. the official vocabularies and one training file, verified against the Hugging Face API
# 4. preprocessing of that file with the official functions: time, cache size, peak RAM of the process and container
# 5. BASE and MoAE trained on that file: VRAM after the first Adam step with the full official vocabularies, steady
#    rows per second after the first 50 steps, peak VRAM, container memory with the 8 DataLoader workers
# 6. inference speed of the MoAE checkpoint on the same file, keeping no labels or metric (--no_metrics)
set -euo pipefail
OUT=runs/e_pilot
mkdir -p "$OUT"

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen
{
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
  free -g; df -h .; nproc
  cat /sys/fs/cgroup/memory.max 2>/dev/null || cat /sys/fs/cgroup/memory/memory.limit_in_bytes 2>/dev/null || true
} | tee "$OUT/env.txt"
uv run python - <<'EOF' | tee -a "$OUT/env.txt"
import torch
print("torch", torch.__version__, "cuda build", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not usable: the CUDA 13 wheels need an NVIDIA driver >= 580 on the host")
print("device", torch.cuda.get_device_name(0))
EOF

{ uv run pytest -q 2>&1 | tail -15; uv run pytest -m e2e -q -rs 2>&1 | tail -6; } > "$OUT/tests.txt" || true
grep -E "passed|failed|error" "$OUT/tests.txt" || echo "TESTS: no summary line, see $OUT/tests.txt"

python3 scripts/fetch_mac.py vocabs train/train-00.parquet
uv run python -m cvr.pymal_run preprocess train-00.parquet
cp "$(ls -t data/pymal_cache/preprocess_*.json | head -1)" "$OUT/preprocess.json"

uv run python -m cvr.pymal_run train --out "$OUT/BASE" --model BASE --lr 0.003 --lamda 0.3 \
  --train_files train-00.parquet --ckpt_after
uv run python -m cvr.pymal_run train --out "$OUT/MoAE" --model MoAE --lr 0.003 --lamda 0.3 \
  --aux_views first mta linear cartesian --train_files train-00.parquet --ckpt_after train-00.parquet
uv run python -m cvr.pymal_run predict --out "$OUT/predict" --no_metrics \
  --ckpt "$OUT/MoAE/model_after_train-00.pt" train-00.parquet
free -g | tee -a "$OUT/env.txt"
echo "pilot done: $OUT"
