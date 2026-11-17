#!/usr/bin/env bash
# Download MAC files from Hugging Face with resume, then verify size and SHA-256
# against the values published by the Hugging Face API.
#
# Usage: scripts/fetch_mac.sh train/train-18.parquet train/train-19.parquet ...
# Files are saved under data/raw/mac/<path>. Raw data is never committed.
set -euo pipefail

REPO="alimamaTech/MAC"
DEST_ROOT="data/raw/mac"

for rel in "$@"; do
  dir=$(dirname "$rel")
  dest="$DEST_ROOT/$rel"
  mkdir -p "$DEST_ROOT/$dir"
  meta=$(curl -s "https://huggingface.co/api/datasets/$REPO/tree/main/$dir")
  read -r size sha < <(python3 -c "
import sys, json
items = json.loads(sys.argv[1]); rel = sys.argv[2]
f = [i for i in items if i['path'] == rel][0]
print(f['size'], f['lfs']['oid'])" "$meta" "$rel")

  for attempt in $(seq 1 12); do
    # abort a stalled transfer (< 100 KB/s for 60 s) or a connection that never opens (30 s), so the loop
    # resumes it instead of hanging (a 0-byte connect stall happened twice for train-05 / train-07)
    curl -sL --retry 3 --connect-timeout 30 --speed-limit 100000 --speed-time 60 -C - -o "$dest" \
      "https://huggingface.co/datasets/$REPO/resolve/main/$rel" || true
    [ "$(stat -f %z "$dest")" = "$size" ] && break
    echo "$rel: attempt $attempt incomplete ($(stat -f %z "$dest") / $size bytes)"
  done

  local_sha=$(shasum -a 256 "$dest" | cut -d' ' -f1)
  if [ "$local_sha" = "$sha" ]; then
    echo "$local_sha  $(basename "$dest")" > "$dest.sha256"
    echo "OK $rel ($size bytes, sha256 matches)"
  else
    echo "FAIL $rel: sha256 $local_sha != $sha" >&2
    exit 1
  fi
done
