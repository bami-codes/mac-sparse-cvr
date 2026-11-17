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

  for attempt in 1 2 3 4 5 6; do
    curl -sL --retry 3 -C - -o "$dest" "https://huggingface.co/datasets/$REPO/resolve/main/$rel" || true
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
