"""The predictor (cvr/predictor.py) and its example (examples/): a request in, calibrated probabilities out."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cvr import predictor as P

ROOT = Path(__file__).resolve().parents[1]
REQUEST = json.loads((ROOT / "examples" / "request.json").read_text())


@pytest.fixture(scope="module")
def server():
    return P.load()


def test_the_example_answers_its_saved_response(server):
    got = P.predict(server, REQUEST)
    want = json.loads((ROOT / "examples" / "response.json").read_text())
    assert [c["ad_feat0"] for c in got["candidates"]] == [c["ad_feat0"] for c in want["candidates"]]
    for key in ("cvr", "cvr_uncalibrated"):
        assert np.allclose([c[key] for c in got["candidates"]], [c[key] for c in want["candidates"]], atol=1e-6)
    assert got["values_not_in_vocabulary"] == {}


def test_the_response_is_the_server_request_path(server):
    got = P.predict(server, REQUEST)
    direct = server.predict_request(REQUEST["user"], REQUEST["candidates"])
    assert np.allclose([c["cvr"] for c in got["candidates"]], direct, atol=1e-8)
    per_row = server.predict([c | REQUEST["user"] for c in REQUEST["candidates"]])  # the same as one row per candidate
    assert np.allclose(direct, per_row, atol=1e-7)


def test_the_command_line(tmp_path):
    out = subprocess.run(
        [sys.executable, "-m", "cvr.predictor", str(ROOT / "examples" / "request.json")],
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    assert len(json.loads(out.stdout)["candidates"]) == 3
    bad = copy.deepcopy(REQUEST)
    del bad["user"]["user_feat0"]
    lines = "\n".join([json.dumps(REQUEST), json.dumps(bad), json.dumps(REQUEST)])
    out = subprocess.run(
        [sys.executable, "-m", "cvr.predictor", "-", "--jsonl"],
        input=lines,
        capture_output=True,
        text=True,
        cwd=ROOT,
        check=True,
    )
    answers = [json.loads(x) for x in out.stdout.splitlines()]
    assert len(answers) == 3 and "error" in answers[1] and answers[0] == answers[2]


def test_line_by_line_answers_each_request_without_waiting_for_the_input_to_end():
    import select

    p = subprocess.Popen(
        [sys.executable, "-m", "cvr.predictor", "-", "--jsonl"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=ROOT,
    )
    try:
        answers = []
        for line in (REQUEST, {"user": REQUEST["user"], "candidates": [None]}, REQUEST):
            p.stdin.write(json.dumps(line) + "\n")
            p.stdin.flush()
            ready, _, _ = select.select([p.stdout], [], [], 60)
            assert ready, "no answer while the input is still open"
            answers.append(json.loads(p.stdout.readline()))
        assert "candidate 0 must be an object" in answers[1]["error"]
        assert answers[0] == answers[2] and p.poll() is None  # the bad line did not stop the server
    finally:
        p.stdin.close()
        p.wait(timeout=60)


@pytest.mark.parametrize(
    "break_it, message",
    [
        (lambda r: r["user"].pop("ad_feat0_seq"), "user lacks"),
        (lambda r: r["user"]["ad_feat1_seq"].pop(), "one length"),
        (lambda r: r["user"].update(user_feat0=3), "must be strings"),
        (lambda r: r["user"]["ad_feat2_seq"].__setitem__(0, None), "lists of strings"),
        (lambda r: r["candidates"][1].pop("context_feat2"), "candidate 1 lacks"),
        (lambda r: r["candidates"].__setitem__(2, None), "candidate 2 must be an object"),
        (lambda r: r["candidates"][0].update(ad_feat3=["x"]), "must be strings"),
        (lambda r: r["candidates"][0]["mm_feat0_seq"].pop(), "one number per history item"),
        (lambda r: r["candidates"][0]["mm_feat1_seq"].__setitem__(0, "0.1"), "one number per history item"),
        (lambda r: r.update(candidates=[]), "non-empty"),
    ],
)
def test_invalid_requests_are_refused_with_the_reason(server, break_it, message):
    r = copy.deepcopy(REQUEST)
    break_it(r)
    with pytest.raises(P.RequestError, match=message):
        P.predict(server, r)


def test_new_values_are_served_and_counted(server):
    r = copy.deepcopy(REQUEST)
    r["candidates"][0]["ad_feat0"] = "never-seen-ad"
    r["candidates"][0]["ad_feat1"] = "never-seen-group"
    r["user"]["user_id"] = "never-seen-user"
    got = P.predict(server, r)
    assert got["values_not_in_vocabulary"] == {"ad_feat0": 1, "ad_feat1": 1}
    assert all(0 < c["cvr"] < 1 for c in got["candidates"])


def test_a_bundle_refuses_another_checkpoint(tmp_path):
    import shutil

    b = tmp_path / "bundle"
    shutil.copytree(P.EXAMPLE, b)
    cfg = json.loads((b / "config.json").read_text())
    (b / "config.json").write_text(json.dumps(cfg | {"checkpoint_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="not the checkpoint"):
        P.load(b)


def test_the_example_is_rebuilt_identically(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_example_bundle as M
    import torch

    try:
        rebuilt = M.build(tmp_path)
    finally:
        torch.use_deterministic_algorithms(False)  # the official set_seed switches it on for the whole process
    want = json.loads((ROOT / "examples" / "response.json").read_text())
    assert np.allclose([c["cvr"] for c in rebuilt["candidates"]], [c["cvr"] for c in want["candidates"]], atol=1e-6)


def test_a_bundle_without_calibrator_serves_raw_probabilities(tmp_path):
    import shutil

    b = tmp_path / "bundle"
    shutil.copytree(P.EXAMPLE, b)
    cfg = json.loads((b / "config.json").read_text())
    (b / "config.json").write_text(json.dumps(cfg | {"calibrator": None, "calibration": "none"}))
    (b / "calibrator.npz").unlink()
    got = P.predict(P.load(b), REQUEST)
    assert got["calibration"] == "none"
    assert all(c["cvr"] == c["cvr_uncalibrated"] for c in got["candidates"])


def test_export_bundles_a_calibrator_only_with_the_checkpoint_it_was_fitted_on(tmp_path, monkeypatch):
    """A study E layout in miniature: two checkpoints of one run, the calibrator fitted on train-17's predictions."""
    import polars as pl

    from cvr import pymal_run as R
    from cvr import study_e as E
    from cvr import study_e_eval as Q

    shutil = __import__("shutil")
    monkeypatch.setattr(Q, "RUNS", tmp_path / "e_runs")
    monkeypatch.setattr(E, "OUT", tmp_path / "e_study")
    rid = "BASE_lr0.003_lam0.3_s2026"
    d = Q.RUNS / rid
    d.mkdir(parents=True)
    shas = {}
    for k in (17, 19):
        (d / f"model_after_train-{k}.pt").write_bytes(f"checkpoint {k}".encode())
        shas[f"train-{k}.parquet"] = {"sha256": R.sha256(d / f"model_after_train-{k}.pt")}
    (d / "run.json").write_text(json.dumps({"args": {"model": "BASE"}, "checkpoints": shas}))
    pred = Q.prediction_file("research", Q.CKPT, shas["train-17.parquet"]["sha256"], Q.V1).with_suffix(".json")
    pred.parent.mkdir(parents=True)
    pred.write_text(json.dumps({"checkpoint_sha256": shas["train-17.parquet"]["sha256"]}))
    cal = Q.calibrator_path(rid, "iii", "main")
    cal.parent.mkdir(parents=True)
    shutil.copy(P.EXAMPLE / "calibrator.npz", cal)
    pl.DataFrame({"ad_feat1": ["a", "b"], "clicks": [1, 2], "pos": [0, 1]}).write_parquet(
        E.OUT / "train_stats_ad_feat1.parquet"
    )

    cfg = P.export(rid, tmp_path / "b17")
    assert cfg["calibrator"] == "calibrator.npz" and (tmp_path / "b17" / "calibrator.npz").exists()
    with pytest.raises(SystemExit, match="no-calibration"):
        P.export(rid, tmp_path / "b19", after="train-19.parquet")
    assert not (tmp_path / "b19").exists()
    cfg = P.export(rid, tmp_path / "b19", after="train-19.parquet", calibrated=False)
    assert cfg["calibrator"] is None and not (tmp_path / "b19" / "calibrator.npz").exists()
