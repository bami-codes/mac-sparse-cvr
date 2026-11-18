"""Study E Q2/Q3 on synthetic predictions with a known group bias (no MAC data, no GPU)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.special import expit

from cvr import metrics as M
from cvr import study_e as E
from cvr import study_e_eval as Q

RNG = np.random.default_rng(7)
TOKENS = [f"tok{i:03d}" for i in range(30)]
G1, C = TOKENS[:10], TOKENS[10:]


def frame(n: int, seed: int):
    rng = np.random.default_rng(seed)
    tok = rng.choice(TOKENS, n)
    base = np.where(np.isin(tok, G1), -4.0, -2.5) + rng.normal(0, 0.5, n)
    y = (rng.random(n) < expit(base)).astype(np.int8)
    user = rng.integers(0, 2000, n)
    return tok, base, y, user


def write_case(tmp: Path, monkeypatch, shuffle_v2: bool = False) -> Path:
    out, data, runs = tmp / "e_study", tmp / "train", tmp / "e_runs"
    for d in (out, data, runs):
        d.mkdir(parents=True)
    monkeypatch.setattr(E, "OUT", out)
    monkeypatch.setattr(E, "DATA", data)
    monkeypatch.setattr(Q, "RUNS", runs)
    pl.DataFrame({"ad_feat1": TOKENS, "clicks": [5000] * 30, "pos": [1] * 30}).write_parquet(out / "train_stats_ad_feat1.parquet")
    (out / "group_lists.json").write_text(json.dumps({"G1": G1, "C": C}))
    (out / "groups.json").write_text(json.dumps({"group_lists_sha256": "x"}))
    files = {}
    for name, seed in ((Q.V1, 1), (Q.V2, 2)):
        tok, base, y, user = frame(20_000, seed)
        pl.DataFrame({"ad_feat1": tok, "last": y.astype(float)}).write_parquet(data / name)
        files[name] = (tok, base, y, user)
    choice = {}
    for model, shift in (("MoAE", 0.6), ("BASE", 0.9)):  # each model over-predicts the G1 groups by `shift` logits
        choice[model] = {"lr": 0.003, "lamda": 0.3}
        for s in Q.SEEDS:
            rid = Q.run_id(model, 0.003, 0.3, s)
            d = runs / rid / "research" / f"{Q.CKPT}_abcdef123456"
            d.mkdir(parents=True)
            for name, (tok, base, y, user) in files.items():
                noise = np.random.default_rng(s).normal(0, 0.05, y.size)
                p = expit(base + noise + np.where(np.isin(tok, G1), shift, 0.0))
                yy = y.copy()
                if shuffle_v2 and name == Q.V2:
                    yy = yy[::-1].copy()  # predictions saved in another row order
                np.savez(d / f"{Path(name).stem}.npz", p_last=p.astype(np.float32), y_last=yy, user=user, row=np.arange(y.size))
                sha = hashlib.sha256((d / f"{Path(name).stem}.npz").read_bytes()).hexdigest()
                (d / f"{Path(name).stem}.json").write_text(json.dumps({"sha256": sha, "checkpoint_sha256": f"ck{rid}"}))
    path = tmp / "choice.json"
    path.write_text(json.dumps(choice))
    return path


def test_calibration_choice_bootstrap_and_q3(tmp_path, monkeypatch):
    rec = Q.calibrate(write_case(tmp_path, monkeypatch))
    rows = rec["v2_rows"]
    for model in ("MoAE", "BASE"):
        rids = [Q.run_id(model, 0.003, 0.3, s) for s in Q.SEEDS]
        means = {lam: np.mean([rows[f"{r}|iii_lam{lam:g}"]["G1"]["macro_logloss"] for r in rids]) for lam in Q.LAMBDAS}
        assert rec["lambda_chosen"][model]["iii"]["main"] == min(means, key=means.get)
        # the known bias is corrected: group calibration beats the global mapping on G1 here
        assert rec["v2_bootstrap"][f"{model}: iii - ii"]["diff"] < 0
        for r in rids:  # one file per run (a dot in the run id used to make them overwrite each other)
            assert Q.calibrator_path(r, "iii", "main").exists()
        q_ii, q_iii = Q.load_calibrated(rids[1], "iii", "main", np.full(4, 0.05), np.array([0, 1, 15, -1]))
        assert np.all((q_iii > 0) & (q_iii < 1)) and q_ii[0] == q_ii[3]
    assert len(list((tmp_path / "e_study" / "q2" / "calibrators").glob("*.npz"))) == 6 * 2 * 2  # runs x methods x rules
    d = rec["v2_bootstrap"]["Q3 ii: (MoAE - BASE) on G1 minus on C"]
    assert abs(d["did"] - (d["diff_set_1"] - d["diff_set_2"])) < 1e-12
    assert d["ci95"][0] <= d["did"] <= d["ci95"][1]


def test_did_point_equals_two_group_reports():
    tok, base, y, user = frame(5000, 3)
    g = np.array([TOKENS.index(t) for t in tok])
    a, b = expit(base + 0.3), expit(base - 0.2)
    r = Q.did_bootstrap(y, [a], [b], g, user, np.arange(10), np.arange(10, 30), reps=50, seed=0)
    m1, m2 = g < 10, g >= 10
    want = ((M.group_report(y[m1], a[m1], g[m1])["macro_logloss"] - M.group_report(y[m1], b[m1], g[m1])["macro_logloss"])
            - (M.group_report(y[m2], a[m2], g[m2])["macro_logloss"] - M.group_report(y[m2], b[m2], g[m2])["macro_logloss"]))
    assert abs(r["did"] - want) < 1e-12


def test_predictions_in_another_row_order_are_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="labels differ"):
        Q.calibrate(write_case(tmp_path, monkeypatch, shuffle_v2=True))
