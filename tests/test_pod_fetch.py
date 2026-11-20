"""scripts/pod_fetch.sh checkpoints with rsync replaced by a local copy: all 15 checkpoints arrive and are verified."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SHIM = """#!/bin/sh
# rsync stand-in: copy <addr>:/root/naver/<path> from $FAKE_POD/<path> (the last two arguments are source and target)
for last; do :; done
src=$(eval echo \\${$(($#-1))})
cp "$FAKE_POD/${src#*:/root/naver/}" "$last"
"""


def test_all_fifteen_checkpoints_are_copied_and_verified(tmp_path):
    pod, work, bin_ = tmp_path / "pod", tmp_path / "work", tmp_path / "bin"
    for d in (pod, work / "runs" / "eval", bin_):
        d.mkdir(parents=True)
    (bin_ / "rsync").write_text(SHIM)
    (bin_ / "rsync").chmod(0o755)
    choice = {"MoAE": {"lr": 0.00035, "lamda": 0.4}, "BASE": {"lr": 0.0035, "lamda": 0.3}}  # long run ids
    (work / "runs" / "eval" / "config_choice.json").write_text(json.dumps(choice))
    ids = [f"{m}_lr{c['lr']:g}_lam{c['lamda']:g}_s{s}" for m, c in choice.items() for s in (2026, 2027, 2028)]
    ids += [f"MoAE-g1w2_lr0.00035_lam0.4_s{s}" for s in (2026, 2027, 2028)]
    for rid in ids:
        d = pod / "runs" / "train" / rid
        d.mkdir(parents=True)
        cks = {}
        for k in (17,) if rid.startswith("MoAE-g1w2") else (17, 19):
            blob = f"{rid}-{k}".encode() * 1000
            (d / f"model_after_train-{k}.pt").write_bytes(blob)
            cks[f"train-{k}.parquet"] = {"sha256": hashlib.sha256(blob).hexdigest()}
        (d / "run.json").write_text(json.dumps({"checkpoints": cks}))
        if not rid.startswith("MoAE-g1w2"):  # the chosen runs' records were fetched with `records` before
            (work / "runs" / "train" / rid).mkdir(parents=True)
            (work / "runs" / "train" / rid / "run.json").write_text((d / "run.json").read_text())
    env = os.environ | {"PATH": f"{bin_}:{os.environ['PATH']}", "FAKE_POD": str(pod), "POD": "-p 1 root@pod"}
    p = subprocess.run(
        ["bash", str(REPO / "scripts" / "pod_fetch.sh"), "checkpoints"],
        cwd=work,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.count("OK  ") == 15 and "BAD" not in p.stdout
