"""The predictor (cvr/predictor.py) and its example (examples/): a request in, calibrated probabilities out."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from cvr import predictor

ROOT = Path(__file__).resolve().parents[1]
REQUEST = json.loads((ROOT / "examples" / "request.json").read_text())


@pytest.fixture(scope="module")
def server():
    return predictor.load()


def test_the_example_answers_its_saved_response(server):
    got = predictor.predict(server, REQUEST)
    want = json.loads((ROOT / "examples" / "response.json").read_text())
    assert [c["ad_feat0"] for c in got["candidates"]] == [c["ad_feat0"] for c in want["candidates"]]
    for key in ("cvr", "cvr_uncalibrated"):
        assert np.allclose([c[key] for c in got["candidates"]], [c[key] for c in want["candidates"]], atol=1e-6)
    assert got["values_not_in_vocabulary"] == {}


def test_the_response_is_the_server_request_path(server):
    got = predictor.predict(server, REQUEST)
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


def test_jsonl_answers_each_line_at_once():
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
        (lambda r: r["user"]["ad_feat1_seq"].pop(), "must each hold 20 items"),
        (lambda r: [r["user"].update({c: []}) for c in ("ad_feat0_seq", "ad_feat1_seq", "ad_feat2_seq")], "hold 20"),
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
    with pytest.raises(predictor.RequestError, match=message):
        predictor.predict(server, r)


def test_new_values_are_served_and_counted(server):
    r = copy.deepcopy(REQUEST)
    r["candidates"][0]["ad_feat0"] = "never-seen-ad"
    r["candidates"][0]["ad_feat1"] = "never-seen-group"
    r["user"]["user_id"] = "never-seen-user"
    got = predictor.predict(server, r)
    assert got["values_not_in_vocabulary"] == {"ad_feat0": 1, "ad_feat1": 1}
    assert all(0 < c["cvr"] < 1 for c in got["candidates"])


def test_a_bundle_refuses_another_checkpoint(tmp_path):
    import shutil

    b = tmp_path / "bundle"
    shutil.copytree(predictor.EXAMPLE, b)
    cfg = json.loads((b / "config.json").read_text())
    (b / "config.json").write_text(json.dumps(cfg | {"checkpoint_sha256": "0" * 64}))
    with pytest.raises(SystemExit, match="not the checkpoint"):
        predictor.load(b)


def test_the_example_is_rebuilt_identically(tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import make_example_bundle
    import torch

    try:
        rebuilt = make_example_bundle.build(tmp_path)
    finally:
        torch.use_deterministic_algorithms(False)  # the official set_seed switches it on for the whole process
    want = json.loads((ROOT / "examples" / "response.json").read_text())
    assert np.allclose([c["cvr"] for c in rebuilt["candidates"]], [c["cvr"] for c in want["candidates"]], atol=1e-6)


def test_a_bundle_without_calibrator_serves_raw_probabilities(tmp_path):
    import shutil

    b = tmp_path / "bundle"
    shutil.copytree(predictor.EXAMPLE, b)
    cfg = json.loads((b / "config.json").read_text())
    (b / "config.json").write_text(json.dumps(cfg | {"calibrator": None, "calibration": "none"}))
    (b / "calibrator.npz").unlink()
    got = predictor.predict(predictor.load(b), REQUEST)
    assert got["calibration"] == "none"
    assert all(c["cvr"] == c["cvr_uncalibrated"] for c in got["candidates"])


def test_export_pairs_calibrator_with_its_checkpoint(tmp_path, monkeypatch):
    """A training-run layout in miniature: two checkpoints of one run, the calibrator fitted on train-17's predictions."""
    import polars as pl

    from cvr import evaluation, groups, pymal_run

    shutil = __import__("shutil")
    monkeypatch.setattr(evaluation, "RUNS", tmp_path / "train")
    monkeypatch.setattr(groups, "OUT", tmp_path / "eval")
    rid = "BASE_lr0.003_lam0.3_s2026"
    d = evaluation.RUNS / rid
    d.mkdir(parents=True)
    shas = {}
    for k in (17, 19):
        (d / f"model_after_train-{k}.pt").write_bytes(f"checkpoint {k}".encode())
        shas[f"train-{k}.parquet"] = {"sha256": pymal_run.sha256(d / f"model_after_train-{k}.pt")}
    (d / "run.json").write_text(json.dumps({"args": {"model": "BASE"}, "checkpoints": shas}))
    pred = evaluation.prediction_file(
        "research", evaluation.CKPT, shas["train-17.parquet"]["sha256"], evaluation.V1
    ).with_suffix(".json")
    pred.parent.mkdir(parents=True)
    pred.write_text(json.dumps({"checkpoint_sha256": shas["train-17.parquet"]["sha256"]}))
    cal = evaluation.calibrator_path(rid, "iii", "main")
    cal.parent.mkdir(parents=True)
    shutil.copy(predictor.EXAMPLE / "calibrator.npz", cal)
    pl.DataFrame({"ad_feat1": ["a", "b"], "clicks": [1, 2], "pos": [0, 1]}).write_parquet(
        groups.OUT / "train_stats_ad_feat1.parquet"
    )

    cfg = predictor.export(rid, tmp_path / "b17")
    assert cfg["calibrator"] == "calibrator.npz" and (tmp_path / "b17" / "calibrator.npz").exists()
    with pytest.raises(SystemExit, match="no-calibration"):
        predictor.export(rid, tmp_path / "b19", after="train-19.parquet")
    assert not (tmp_path / "b19").exists()
    cfg = predictor.export(rid, tmp_path / "b19", after="train-19.parquet", calibrated=False)
    assert cfg["calibrator"] is None and not (tmp_path / "b19" / "calibrator.npz").exists()


def test_empty_history_is_refused(server):
    """An empty purchase history used to pass and give another probability than MAC's padded history (0.0365 vs 0.0482
    for the example's first candidate); MAC always stores 20 slots, so the request must too."""
    r = copy.deepcopy(REQUEST)
    for c in ("ad_feat0_seq", "ad_feat1_seq", "ad_feat2_seq"):
        r["user"][c] = []
    for cand in r["candidates"]:
        cand["mm_feat0_seq"], cand["mm_feat1_seq"] = [], []
    with pytest.raises(predictor.RequestError, match="hold 20"):
        predictor.predict(server, r)


def test_request_from_a_user_with_one_click_and_with_several(tmp_path):
    import polars as pl
    from synthetic_mac import make_file, make_world

    src = make_file(make_world(seed=5, n_users=40, n_items=60), tmp_path / "src.parquet", seed=5, rows=200)
    rows = pl.read_parquet(src)
    counts = rows["user_id"].value_counts()
    many = counts.filter(pl.col("count") >= 3)["user_id"][0]
    one = pl.concat([rows.filter(pl.col("user_id") == many), rows.filter(pl.col("user_id") != many).head(1)])
    path = tmp_path / "rows.parquet"
    one.write_parquet(path)
    single = predictor.request_from(path, one.height - 1, 100)  # its user has this one click only
    assert len(single["candidates"]) == 1
    several = predictor.request_from(path, 1, 2)
    assert len(several["candidates"]) == 2 and several["user"]["user_id"] == many
    assert several["candidates"][0]["ad_feat0"] == one["ad_feat0"][1]  # the given row comes first
