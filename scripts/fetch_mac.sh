#!/usr/bin/env bash
# Replaced by scripts/fetch_mac.py (2026-11-18), which runs on macOS and Linux and also verifies the
# vocabulary files that are plain git files (no SHA-256 in the API). Kept so old commands keep working.
#
# Usage: scripts/fetch_mac.sh train/train-18.parquet test/test.parquet vocabs ...
set -euo pipefail
exec python3 "$(dirname "$0")/fetch_mac.py" "$@"
