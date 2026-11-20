"""Machine facts and the data download work the same on macOS and Linux (a GPU pod runs Linux)."""

import hashlib
import io
import subprocess
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, "scripts")
import fetch_mac as F

from cvr import sysinfo


def test_peak_memory_units_follow_the_os(monkeypatch):
    monkeypatch.setattr(sysinfo.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=2_000_000_000))
    monkeypatch.setattr(sysinfo.sys, "platform", "darwin")  # bytes
    assert sysinfo.peak_rss_gb() == 2.0
    monkeypatch.setattr(sysinfo.resource, "getrusage", lambda _: SimpleNamespace(ru_maxrss=2_000_000))
    monkeypatch.setattr(sysinfo.sys, "platform", "linux")  # kilobytes
    assert sysinfo.peak_rss_gb() == 2.05


def test_cpu_name_is_recorded():
    assert sysinfo.cpu_name()


def test_git_blob_id_matches_git(tmp_path):
    f = tmp_path / "ad_feat5.json"
    f.write_bytes(b'{"dbee74ee5fbf": 0, "a": 1}\n')
    want = subprocess.run(["git", "hash-object", str(f)], capture_output=True, text=True, check=True).stdout.strip()
    assert F.git_blob_sha1(f) == want


def test_verify_uses_sha256_for_lfs_and_the_git_id_otherwise(tmp_path):
    f = tmp_path / "x.bin"
    f.write_bytes(b"official content")
    size, sha = f.stat().st_size, hashlib.sha256(b"official content").hexdigest()
    assert F.verify(f, {"size": size, "lfs": {"oid": sha}}) == sha
    assert F.verify(f, {"size": size, "oid": F.git_blob_sha1(f)}) == sha  # plain git file (most vocabularies)
    with pytest.raises(ValueError, match="SHA-256"):
        F.verify(f, {"size": size, "lfs": {"oid": "0" * 64}})
    with pytest.raises(ValueError, match="git blob"):
        F.verify(f, {"size": size, "oid": "0" * 40})
    with pytest.raises(ValueError, match="bytes"):
        F.verify(f, {"size": size + 1, "lfs": {"oid": sha}})


class _Resp(io.BytesIO):
    def __init__(self, data: bytes, status: int, fail_after: int | None = None):
        super().__init__(data)
        self.status, self.fail_after, self.sent = status, fail_after, 0

    def read(self, n=-1):
        if self.fail_after is not None and self.sent >= self.fail_after:
            raise TimeoutError("stalled")
        chunk = super().read(min(n, 4) if n and n > 0 else -1)
        self.sent += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_download_resumes_after_a_stall_and_restarts_if_the_range_is_ignored(tmp_path, monkeypatch):
    content = b"0123456789abcdefghij"
    calls = []

    def urlopen(req, timeout):
        rng = req.headers.get("Range")
        calls.append(rng)
        if len(calls) == 1:
            return _Resp(content, 200, fail_after=8)  # stalls after 8 bytes
        if len(calls) == 2:
            return _Resp(content, 200)  # ignores the range: the file must start again, not be appended to
        start = int(rng.split("=")[1].rstrip("-")) if rng else 0
        return _Resp(content[start:], 206 if rng else 200)

    monkeypatch.setattr(F.urllib.request, "urlopen", urlopen)
    monkeypatch.setattr(F.time, "sleep", lambda s: None)
    dest = tmp_path / "train-00.parquet"
    F.download("train/train-00.parquet", dest, len(content))
    assert dest.read_bytes() == content
    assert calls[0] is None and calls[1] == "bytes=8-"


def test_fetch_writes_the_sha256_sidecar(tmp_path, monkeypatch):
    content = b"vocab"
    monkeypatch.setattr(F, "DEST", tmp_path)
    monkeypatch.setattr(
        F,
        "download",
        lambda rel, dest, size: (dest.parent.mkdir(parents=True, exist_ok=True), dest.write_bytes(content)),
    )
    (tmp_path / "vocabs").mkdir()
    probe = tmp_path / "probe"
    probe.write_bytes(content)
    F.fetch("vocabs/user_feat4.json", {"size": len(content), "oid": F.git_blob_sha1(probe)})
    side = (tmp_path / "vocabs" / "user_feat4.json.sha256").read_text().split()
    assert side == [hashlib.sha256(content).hexdigest(), "user_feat4.json"]
