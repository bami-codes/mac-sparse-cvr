"""Download MAC files from Hugging Face with resume, then verify them against the Hugging Face API (any OS).

    python3 scripts/fetch_mac.py train/train-18.parquet test/test.parquet vocabs   # a folder name = every file in it

Files are saved under data/raw/mac/<path>. Raw data is never committed. Verification:
- LFS files (the parquet files and the large vocabularies): size and SHA-256 equal the API's `lfs.oid`;
- plain git files (13 of the 17 vocabulary files): size and git blob SHA-1 equal the API's `oid`
  (the API gives no SHA-256 for them; the earlier shell script could not verify them).
Every verified file gets a `<file>.sha256` sidecar with its SHA-256, which the runners record.
Only the standard library is used, so this runs before any environment is installed (a fresh GPU pod).
Replaces scripts/fetch_mac.sh, which used BSD `stat -f` (wrong on Linux) and `shasum`.
"""

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

REPO = "alimamaTech/MAC"
API = f"https://huggingface.co/api/datasets/{REPO}/tree/main"
RESOLVE = f"https://huggingface.co/datasets/{REPO}/resolve/main"
DEST = Path("data/raw/mac")
ATTEMPTS, TIMEOUT, CHUNK = 12, 60, 1 << 20
UA = {"User-Agent": "mac-fetch/1.0"}


def listing(folder: str) -> list[dict]:
    with urllib.request.urlopen(urllib.request.Request(f"{API}/{folder}", headers=UA), timeout=TIMEOUT) as r:
        return [i for i in json.loads(r.read()) if i.get("type") == "file"]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path: Path) -> str:
    """The id git gives a file's content: sha1(b"blob <size>\\0" + content)."""
    h = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def verify(path: Path, meta: dict) -> str:
    """SHA-256 of a file that matches the API record; raises otherwise."""
    size = path.stat().st_size if path.exists() else -1
    if size != int(meta["size"]):
        raise ValueError(f"{path}: {size} bytes, the API says {meta['size']}")
    digest = sha256(path)
    if "lfs" in meta:
        if digest != meta["lfs"]["oid"]:
            raise ValueError(f"{path}: SHA-256 {digest} differs from the API {meta['lfs']['oid']}")
    elif git_blob_sha1(path) != meta["oid"]:
        raise ValueError(f"{path}: git blob SHA-1 differs from the API {meta['oid']}")
    return digest


def download(rel: str, dest: Path, size: int) -> None:
    """Resume with a Range request; a stalled read raises after TIMEOUT seconds and the next attempt resumes."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, ATTEMPTS + 1):
        have = dest.stat().st_size if dest.exists() else 0
        if have == size:
            return
        if have > size:  # a different file under this name: start again
            dest.unlink()
            have = 0
        req = urllib.request.Request(f"{RESOLVE}/{rel}", headers=UA | ({"Range": f"bytes={have}-"} if have else {}))
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(dest, "ab" if have else "wb") as f:
                if have and r.status != 206:  # the server ignored the range: start again
                    f.truncate(0)
                while chunk := r.read(CHUNK):
                    f.write(chunk)
        except OSError as e:
            print(
                f"{rel}: attempt {attempt} stopped at {dest.stat().st_size if dest.exists() else 0}/{size}: {e}",
                flush=True,
            )
            time.sleep(min(30, 2 * attempt))
    if not dest.exists() or dest.stat().st_size != size:
        raise RuntimeError(f"{rel}: incomplete after {ATTEMPTS} attempts")


def fetch(rel: str, meta: dict) -> None:
    dest = DEST / rel
    download(rel, dest, int(meta["size"]))
    digest = verify(dest, meta)
    dest.with_name(dest.name + ".sha256").write_text(f"{digest}  {dest.name}\n")
    print(f"OK {rel} ({meta['size']} bytes, {'sha256' if 'lfs' in meta else 'git blob sha1'} matches)", flush=True)


def main(args: list[str]) -> None:
    folders: dict[str, list[dict]] = {}
    for arg in args:
        arg = arg.strip("/")
        folder = arg if "/" not in arg else arg.rsplit("/", 1)[0]
        items = folders.setdefault(folder, listing(folder))
        todo = items if arg == folder else [i for i in items if i["path"] == arg]
        if not todo:
            raise SystemExit(f"{arg}: not found in {API}/{folder}")
        for meta in todo:
            fetch(meta["path"], meta)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    main(sys.argv[1:])
