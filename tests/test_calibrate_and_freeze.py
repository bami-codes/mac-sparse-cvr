"""scripts/calibrate_and_freeze.sh with `uv` replaced by a recorder: only the missing V1/V2 predictions are made."""

import hashlib
import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FAKE_UV = """#!/bin/sh
# record the call; for predict, write the .npz and .json of the one file asked for
echo "$*" >> "$CALLS"
case "$*" in
  *pymal_run\\ predict*)
    for a; do last=$a; done
    ck=$(echo "$*" | sed 's/.*--ckpt \\([^ ]*\\).*/\\1/')
    sha=$(python3 -c "import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest()[:12])" "$ck")
    d="runs/e_runs/research/model_after_train-17_$sha"; mkdir -p "$d"
    touch "$d/${last%.parquet}.npz" "$d/${last%.parquet}.json" ;;
  *cvr.evaluation\\ calibrate*) mkdir -p runs/e_study/q2 && touch runs/e_study/q2/selection.json ;;
  *cvr.final_test\\ freeze*) mkdir -p runs/e_study/frozen && touch runs/e_study/frozen/freeze_q2.json ;;
esac
"""


def test_only_missing_predictions_are_made(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "uv").write_text(FAKE_UV)
    (bin_ / "uv").chmod(0o755)
    choice = {
        "MoAE": {"lr": 0.003, "lamda": 0.3},
        "BASE": {"lr": 0.0035, "lamda": 0.3},
        "MoAE-g1w2": {"lr": 0.003, "lamda": 0.3},
    }
    (tmp_path / "runs" / "e_study").mkdir(parents=True)
    (tmp_path / "runs" / "e_study" / "config_choice.json").write_text(json.dumps(choice))
    shas = {}
    for m, c in choice.items():
        for s in (2026, 2027, 2028):
            rid = f"{m}_lr{c['lr']:g}_lam{c['lamda']:g}_s{s}"
            d = tmp_path / "runs" / "e_runs" / rid
            d.mkdir(parents=True)
            cks = {}
            for k in (17,) if m == "MoAE-g1w2" else (17, 19):
                blob = f"{rid}{k}".encode()
                (d / f"model_after_train-{k}.pt").write_bytes(blob)
                cks[f"train-{k}.parquet"] = {"sha256": hashlib.sha256(blob).hexdigest()}
            (d / "run.json").write_text(json.dumps({"checkpoints": cks}))
            shas[rid] = cks["train-17.parquet"]["sha256"][:12]
    done = tmp_path / "runs" / "e_runs" / "research"
    first = "MoAE_lr0.003_lam0.3_s2026"
    (done / f"model_after_train-17_{shas[first]}").mkdir(parents=True)
    for f in ("train-18.npz", "train-18.json", "train-19.npz", "train-19.json"):  # this run is complete
        (done / f"model_after_train-17_{shas[first]}" / f).touch()
    half = "BASE_lr0.0035_lam0.3_s2027"
    (done / f"model_after_train-17_{shas[half]}").mkdir(parents=True)
    (done / f"model_after_train-17_{shas[half]}" / "train-18.npz").touch()  # interrupted: no record yet
    (done / f"model_after_train-17_{shas[half]}" / "train-18.json").touch()
    (done / f"model_after_train-17_{shas[half]}" / "train-19.npz").touch()  # cut before its record
    env = os.environ | {"PATH": f"{bin_}:{os.environ['PATH']}", "CALLS": str(tmp_path / "calls.txt")}
    p = subprocess.run(
        ["bash", str(REPO / "scripts" / "calibrate_and_freeze.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode == 0, p.stdout + p.stderr
    calls = (tmp_path / "calls.txt").read_text().splitlines()
    predicts = [c for c in calls if "pymal_run predict" in c]
    assert len(predicts) == 9 * 2 - 2 - 1  # complete run skipped, the half run's finished V1 skipped
    assert all(c.count("--ckpt") == 1 and c.split()[-1] in ("train-18.parquet", "train-19.parquet") for c in predicts)
    assert any(half in c and c.endswith("train-19.parquet") for c in predicts)
    assert not any(first in c for c in predicts)
    assert [c.split()[4] for c in calls if "cvr.final_test" in c] == ["freeze", "dryrun"]


def test_a_changed_local_checkpoint_stops_before_any_prediction(tmp_path):
    bin_ = tmp_path / "bin"
    bin_.mkdir()
    (bin_ / "uv").write_text(FAKE_UV)
    (bin_ / "uv").chmod(0o755)
    choice = {
        "MoAE": {"lr": 0.003, "lamda": 0.3},
        "BASE": {"lr": 0.0035, "lamda": 0.3},
        "MoAE-g1w2": {"lr": 0.003, "lamda": 0.3},
    }
    (tmp_path / "runs" / "e_study").mkdir(parents=True)
    (tmp_path / "runs" / "e_study" / "config_choice.json").write_text(json.dumps(choice))
    for m, c in choice.items():
        for s in (2026, 2027, 2028):
            rid = f"{m}_lr{c['lr']:g}_lam{c['lamda']:g}_s{s}"
            d = tmp_path / "runs" / "e_runs" / rid
            d.mkdir(parents=True)
            cks = {}
            for k in (17,) if m == "MoAE-g1w2" else (17, 19):
                (d / f"model_after_train-{k}.pt").write_bytes(b"x")
                cks[f"train-{k}.parquet"] = {"sha256": "0" * 64}  # the record does not match
            (d / "run.json").write_text(json.dumps({"checkpoints": cks}))
    env = os.environ | {"PATH": f"{bin_}:{os.environ['PATH']}", "CALLS": str(tmp_path / "calls.txt")}
    p = subprocess.run(
        ["bash", str(REPO / "scripts" / "calibrate_and_freeze.sh")],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert p.returncode != 0 and "BAD" in p.stdout
    assert not (tmp_path / "calls.txt").exists()
