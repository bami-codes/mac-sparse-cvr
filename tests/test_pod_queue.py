"""The study E queue on CPU with synthetic files: train -> validate on train-19, a diverged run, the summary."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from synthetic_mac import make_file, make_world, write_official_vocabs  # noqa: E402

from cvr import pymal_run as R  # noqa: E402

REPO = Path(__file__).resolve().parents[1]


@pytest.mark.e2e
def test_the_queue_trains_validates_and_records_a_diverged_run(tmp_path):
    mac = tmp_path / "data" / "raw" / "mac"
    world = make_world(seed=3, n_users=200, n_items=100)
    names = [f"train-{i:02d}.parquet" for i in range(20)]
    for i, n in enumerate(names):
        make_file(world, mac / "train" / n, seed=i, rows=300)
    make_file(world, mac / "test" / "test.parquet", seed=99, rows=300)
    write_official_vocabs(mac)
    R.preprocess(mac, tmp_path / "data" / "pymal_cache", names)
    sys.path.insert(0, str(REPO / "scripts"))
    import pod_queue as Q
    # MoAE at 1e6 reaches a non-finite loss; BASE at 1e6 only collapses (the official loss adds 1e-8 inside the log,
    # so it stays finite) and is validated like any run, losing the selection with AUC 0.5
    jobs = [Q.job("MoAE", 0.003, 0.3, 2026), Q.job("MoAE", 1e6, 0.3, 2026)]
    (tmp_path / "jobs.json").write_text(json.dumps(jobs))
    env = os.environ | {"PYTHONPATH": str(REPO)}
    q = [sys.executable, str(REPO / "scripts" / "pod_queue.py")]
    subprocess.run(q + ["run", "jobs.json", "--gpus", "2", "--device", "cpu", "--num_workers", "0"], cwd=tmp_path,
                   env=env, check=True, capture_output=True, timeout=900)
    out = subprocess.run(q + ["summary", "jobs.json"], cwd=tmp_path, env=env, check=True, capture_output=True,
                         text=True).stdout
    rows = {r["id"]: r for r in map(json.loads, out.splitlines())}
    good, bad = rows[jobs[0]["id"]], rows[jobs[1]["id"]]
    assert good["state"] == "done" and 0 <= good["val_gauc"] <= 1
    assert bad["state"] == "diverged" and "val_gauc" not in bad
    run = tmp_path / "runs" / "e_runs" / jobs[0]["id"]
    assert {p.name for p in run.glob("model_after_*.pt")} == {f"model_after_train-{k}.pt" for k in (17, 18, 19)}
    rerun = subprocess.run(q + ["run", "jobs.json", "--gpus", "2", "--device", "cpu", "--num_workers", "0"],
                           cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert "GPU" not in rerun.stdout  # finished jobs are skipped
