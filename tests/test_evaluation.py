"""Q2/Q3 of the official test on synthetic predictions with a known group bias (no MAC data, no GPU)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from scipy.special import expit

from cvr import evaluation, groups, metrics

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
    out, data, runs = tmp / "eval", tmp / "data", tmp / "train"
    for d in (out, data, runs):
        d.mkdir(parents=True)
    monkeypatch.setattr(groups, "OUT", out)
    monkeypatch.setattr(groups, "DATA", data)
    monkeypatch.setattr(evaluation, "RUNS", runs)
    pl.DataFrame({"ad_feat1": TOKENS, "clicks": [5000] * 30, "pos": [1] * 30}).write_parquet(
        out / "train_stats_ad_feat1.parquet"
    )
    (out / "group_lists.json").write_text(json.dumps({"G1": G1, "C": C}))
    (out / "groups.json").write_text(json.dumps({"group_lists_sha256": "x"}))
    files = {}
    for name, seed in ((evaluation.V1, 1), (evaluation.V2, 2)):
        tok, base, y, user = frame(20_000, seed)
        pl.DataFrame({"ad_feat1": tok, "last": y.astype(float)}).write_parquet(data / name)
        files[name] = (tok, base, y, user)
    choice = {}
    # each model over-predicts the G1 groups by `shift` logits; the weighted MoAE less than the plain one
    for model, shift in (("MoAE", 0.6), ("BASE", 0.9), (evaluation.WEIGHTED, 0.3)):
        choice[model] = {"lr": 0.003, "lamda": 0.3}
        for s in evaluation.SEEDS:
            rid = evaluation.run_id(model, 0.003, 0.3, s)
            (runs / rid).mkdir(parents=True)
            ck17, ck19 = (
                hashlib.sha256(f"17{rid}".encode()).hexdigest(),
                hashlib.sha256(f"19{rid}".encode()).hexdigest(),
            )
            cks = {"train-17.parquet": {"sha256": ck17}} | (
                {} if model == evaluation.WEIGHTED else {"train-19.parquet": {"sha256": ck19}}
            )
            (runs / rid / "run.json").write_text(json.dumps({"checkpoints": cks}))  # the weighted runs stop at train-17
            d = runs / "research" / f"{evaluation.CKPT}_{ck17[:12]}"
            d.mkdir(parents=True)
            for name, (tok, base, y, user) in files.items():
                noise = np.random.default_rng(s).normal(0, 0.05, y.size)
                p = expit(base + noise + np.where(np.isin(tok, G1), shift, 0.0))
                yy = y.copy()
                if shuffle_v2 and name == evaluation.V2:
                    yy = yy[::-1].copy()  # predictions saved in another row order
                np.savez(
                    d / f"{Path(name).stem}.npz",
                    p_last=p.astype(np.float32),
                    y_last=yy,
                    user=user,
                    row=np.arange(y.size),
                )
                sha = hashlib.sha256((d / f"{Path(name).stem}.npz").read_bytes()).hexdigest()
                (d / f"{Path(name).stem}.json").write_text(json.dumps({"sha256": sha, "checkpoint_sha256": ck17}))
    path = tmp / "choice.json"
    path.write_text(json.dumps(choice))
    return path


def test_calibration_choice_bootstrap_and_q3(tmp_path, monkeypatch):
    rec = evaluation.calibrate(write_case(tmp_path, monkeypatch))
    rows = rec["v2_rows"]
    for model in ("MoAE", "BASE"):
        rids = [evaluation.run_id(model, 0.003, 0.3, s) for s in evaluation.SEEDS]
        means = {
            lam: np.mean([rows[f"{r}|iii_lam{lam:g}"]["G1"]["macro_logloss"] for r in rids])
            for lam in evaluation.LAMBDAS
        }
        assert rec["lambda_chosen"][model]["iii"]["main"] == min(means, key=means.get)
        # the known bias is corrected: group calibration beats the global mapping on G1 here
        assert rec["v2_bootstrap"][f"{model}: iii - ii"]["diff"] < 0
        for r in rids:  # one file per run, even with dots in the run id
            assert evaluation.calibrator_path(r, "iii", "main").exists()
        q_ii, q_iii = evaluation.load_calibrated(rids[1], "iii", "main", np.full(4, 0.05), np.array([0, 1, 15, -1]))
        assert np.all((q_iii > 0) & (q_iii < 1)) and q_ii[0] == q_ii[3]
    assert len(list((tmp_path / "eval" / "q2" / "calibrators").glob("*.npz"))) == 9 * 2 * 2  # runs x methods x rules
    q4 = rec["v2_bootstrap"]
    assert q4["Q4: weighted ii - plain ii"]["diff"] < 0  # less G1 bias in the weighted model before group calibration
    assert {
        "Q4: weighted iii - plain iii",
        "Q4 guard: weighted iii - plain iii, all clicks",
        "Q4 guard: weighted iii - plain iii, G1 Poisson deviance",
    } <= set(q4)
    d = rec["v2_bootstrap"]["Q3 ii: (MoAE - BASE) on G1 minus on C"]
    assert abs(d["did"] - (d["diff_set_1"] - d["diff_set_2"])) < 1e-12
    assert d["ci95"][0] <= d["did"] <= d["ci95"][1]


def test_did_point_equals_two_group_reports():
    tok, base, y, user = frame(5000, 3)
    g = np.array([TOKENS.index(t) for t in tok])
    a, b = expit(base + 0.3), expit(base - 0.2)
    r = evaluation.did_bootstrap(y, [a], [b], g, user, np.arange(10), np.arange(10, 30), reps=50, seed=0)
    m1, m2 = g < 10, g >= 10
    want = (
        metrics.group_report(y[m1], a[m1], g[m1])["macro_logloss"]
        - metrics.group_report(y[m1], b[m1], g[m1])["macro_logloss"]
    ) - (
        metrics.group_report(y[m2], a[m2], g[m2])["macro_logloss"]
        - metrics.group_report(y[m2], b[m2], g[m2])["macro_logloss"]
    )
    assert abs(r["did"] - want) < 1e-12


def test_predictions_in_another_row_order_are_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="labels differ"):
        evaluation.calibrate(write_case(tmp_path, monkeypatch, shuffle_v2=True))


# ------------------------------------------------------------------ choose -> calibrate -> freeze -> dry run -> test
def add_test_and_records(tmp: Path, monkeypatch, choice_path: Path):
    from sklearn.metrics import roc_auc_score

    from cvr import final_test

    tok, base, y, user = frame(20_000, 9)
    test = tmp / "test" / "test.parquet"
    test.parent.mkdir()
    pl.DataFrame({"ad_feat1": tok, "last": y.astype(float)}).write_parquet(test)
    test_sha = hashlib.sha256(test.read_bytes()).hexdigest()
    (test.parent / "test.parquet.sha256").write_text(f"{test_sha}  test.parquet\n")
    monkeypatch.setattr(evaluation, "TEST_PATH", test)
    monkeypatch.setattr(final_test, "FROZEN", groups.OUT / "frozen")
    monkeypatch.setattr(final_test, "git_state", lambda: {"commit": "c" * 40, "dirty_files": []})
    choice = json.loads(choice_path.read_text())
    for model, shift in (("MoAE", 0.6), ("BASE", 0.9), (evaluation.WEIGHTED, 0.3)):
        for s in evaluation.SEEDS:
            rid = evaluation.run_id(model, choice[model]["lr"], choice[model]["lamda"], s)
            ck17 = evaluation.checkpoint_sha(rid)
            p = expit(base + np.random.default_rng(s).normal(0, 0.05, y.size) + np.where(np.isin(tok, G1), shift, 0.0))
            d = evaluation.prediction_file("research", evaluation.CKPT, ck17, "test.parquet").parent
            np.savez(d / "test.npz", p_last=p.astype(np.float32), y_last=y, user=user, row=np.arange(y.size))
            sha = hashlib.sha256((d / "test.npz").read_bytes()).hexdigest()
            (d / "test.json").write_text(
                json.dumps({"sha256": sha, "checkpoint_sha256": ck17, "source_sha256": test_sha})
            )
            if model == evaluation.WEIGHTED:  # Q1 reproduces the paper's two models only
                continue
            ck19 = evaluation.checkpoint_sha(rid, "train-19.parquet")
            d1 = evaluation.prediction_file("q1", "model_after_train-19", ck19, "test.parquet").parent
            d1.mkdir(parents=True)
            q = np.clip(p * 0.9, 1e-6, 1 - 1e-6).astype(np.float32)
            np.savez(d1 / "test.npz", p_last=q, y_last=y, user=user, row=np.arange(y.size))
            (d1 / "test.json").write_text(
                json.dumps(
                    {
                        "sha256": hashlib.sha256((d1 / "test.npz").read_bytes()).hexdigest(),
                        "checkpoint_sha256": ck19,
                        "source_sha256": test_sha,
                        "auc": float(roc_auc_score(y, q)),
                        "gauc": metrics.gauc(y, q.astype(np.float64), user)["gauc"],
                    }
                )
            )
    return final_test


def done_predicting(final_test):
    """What cvr.pymal_run predict writes next to each manifest when its single claim has finished."""
    for name in ("q1", "q2"):
        (final_test.FROZEN / f"freeze_{name}.official_test_status.json").write_text(json.dumps({"state": "done"}))


def test_choose_freeze_dryrun_and_one_test_per_question(tmp_path, monkeypatch):
    path = write_case(tmp_path, monkeypatch)
    final_test = add_test_and_records(tmp_path, monkeypatch, path)
    groups.OUT.joinpath("config_choice.json").write_text(path.read_text())
    evaluation.calibrate(path)
    final_test.freeze()
    with pytest.raises(final_test.CheckFailed, match="never rewritten"):
        final_test.freeze()
    with pytest.raises(final_test.CheckFailed, match="dry run must pass"):  # refused before the single run is claimed
        final_test.test("q2")
    assert not list((groups.OUT / "frozen").glob("*.lock"))
    dry = final_test.dryrun()
    assert dry["dryrun_passed"] and dry["values_reproduced"] > 50
    assert set(dry["v2_values_added_after_selection"]) == evaluation.ADDED_AFTER_SELECTION  # recorded, not reproduced
    with pytest.raises(final_test.CheckFailed, match="not complete"):  # the prediction step has not finished
        final_test.test("q2")
    assert not list((groups.OUT / "frozen").glob("evaluation_*.lock"))
    done_predicting(final_test)
    r = final_test.test("q2")
    assert (
        set(r["verdict"]["conditions"]) == {"diff<0", "ci_upper<0", "all_pairs<0"} and r["primary"]["seed_pairs"] == 3
    )
    assert set(r["q4"]["hold"]) == {"all_click_logloss_worse", "g1_poisson_worse"}
    frozen = json.loads((groups.OUT / "frozen" / "freeze_q2.json").read_text())
    assert len(frozen["checkpoints_sha256"]) == 9  # Q2 and Q4 in one claim
    assert len(json.loads((groups.OUT / "frozen" / "freeze_q1.json").read_text())["checkpoints_sha256"]) == 6
    with pytest.raises(final_test.CheckFailed, match="evaluated once"):
        final_test.test("q2")
    q1 = final_test.test("q1")
    assert set(q1["summary"]) == {"MoAE", "BASE"} and q1["summary"]["MoAE"]["judgement"]["auc"] in (
        "reproduced",
        "close",
        "not reproduced",
    )


def test_a_changed_calibrator_stops_the_dry_run(tmp_path, monkeypatch):
    path = write_case(tmp_path, monkeypatch)
    final_test = add_test_and_records(tmp_path, monkeypatch, path)
    groups.OUT.joinpath("config_choice.json").write_text(path.read_text())
    evaluation.calibrate(path)
    final_test.freeze()
    f = next((groups.OUT / "q2" / "calibrators").glob("*.npz"))
    f.write_bytes(f.read_bytes() + b" ")
    with pytest.raises(final_test.CheckFailed, match="changed since freeze"):
        final_test.dryrun()


def grid_rows(**over):
    """Summary rows for the full planned grid, every run validated at GAUC 0.70 unless overridden by id."""
    from cvr import final_test

    rows = []
    for j in final_test.planned_grid():
        r = {k: j[k] for k in ("id", "model", "lr", "lamda", "seed")} | {
            "state": "done",
            "val_gauc": 0.70,
            "val_auc": 0.80,
        }
        rows.append(r | over.get(j["id"], {}))
    return rows


def test_choice_breaks_ties_by_auc(tmp_path, monkeypatch):
    from cvr import final_test

    monkeypatch.setattr(groups, "OUT", tmp_path)
    rows = grid_rows(
        **{
            "MoAE_lr0.004_lam0.1_s2026": {"val_gauc": 0.76, "val_auc": 0.86},
            "MoAE_lr0.0035_lam0.3_s2026": {"val_gauc": 0.7600000004, "val_auc": 0.87},
            "MoAE_lr0.003_lam0.3_s2026": {"val_gauc": 0.7600000002, "val_auc": 0.87},
            "MoAE_lr0.04_lam0.3_s2026": {
                "state": "diverged",
                "val_gauc": None,
                "val_auc": None,
                "diverged": {"loss": "nan"},
            },
            "BASE_lr0.003_lam0.3_s2026": {"val_gauc": 0.74, "val_auc": 0.85},
            "BASE_lr0.0004_lam0.3_s2026": {"val_gauc": 0.75, "val_auc": 0.84},
        }
    )
    s = tmp_path / "summary.jsonl"
    s.write_text("\n".join(json.dumps(r) for r in rows))
    out = final_test.choose(s)
    assert out["completeness"] == {"jobs": 45, "validated": 44, "diverged": 1}
    saved = json.loads((tmp_path / "config_choice.json").read_text())
    assert (
        saved[evaluation.WEIGHTED]["lr"] == saved["MoAE"]["lr"]
        and saved[evaluation.WEIGHTED]["lamda"] == saved["MoAE"]["lamda"]
    )
    assert (out["MoAE"]["lr"], out["MoAE"]["lamda"]) == (0.003, 0.3)  # tie on GAUC and AUC: closest to 0.003 / 0.3
    assert out["BASE"]["lr"] == 0.0004  # GAUC decides first
    with pytest.raises(final_test.CheckFailed, match="chosen once"):
        final_test.choose(s)


@pytest.mark.parametrize("case", ["missing", "duplicate", "failed", "unvalidated", "settings", "finite_stop"])
def test_the_settings_are_chosen_only_from_the_complete_grid(tmp_path, monkeypatch, case):
    from cvr import final_test

    monkeypatch.setattr(groups, "OUT", tmp_path)
    rows = grid_rows()
    if case == "missing":  # one MoAE and one BASE row must not be enough to choose
        rows = rows[:-1]
    elif case == "duplicate":
        rows = rows + [rows[0]]
    elif case == "failed":
        rows[3] = rows[3] | {"state": "resume", "val_gauc": None}
    elif case == "unvalidated":
        rows[4] = rows[4] | {"state": "done", "val_gauc": None}
    elif case == "settings":
        rows[5] = rows[5] | {"lr": 0.005}
    else:  # a stop that is not the planned non-finite rule
        rows[6] = rows[6] | {"state": "diverged", "diverged": {"loss": "2.33"}}
    s = tmp_path / "summary.jsonl"
    s.write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(final_test.CheckFailed, match="not complete"):
        final_test.choose(s)
    assert not (tmp_path / "config_choice.json").exists()


def test_the_freeze_needs_all_nine_research_runs(tmp_path, monkeypatch):
    path = write_case(tmp_path, monkeypatch)
    final_test = add_test_and_records(tmp_path, monkeypatch, path)
    choice = json.loads(path.read_text())
    choice.pop(evaluation.WEIGHTED)  # a freeze without the weighted runs
    path.write_text(json.dumps(choice))
    groups.OUT.joinpath("config_choice.json").write_text(path.read_text())
    evaluation.calibrate(path)
    with pytest.raises(final_test.CheckFailed, match="needs MoAE, BASE and"):
        final_test.freeze()


def test_the_dry_run_reproduces_every_seed_pair(tmp_path, monkeypatch):
    path = write_case(tmp_path, monkeypatch)
    final_test = add_test_and_records(tmp_path, monkeypatch, path)
    groups.OUT.joinpath("config_choice.json").write_text(path.read_text())
    evaluation.calibrate(path)
    sel_path = groups.OUT / "q2" / "selection.json"
    sel = json.loads(sel_path.read_text())
    sel["v2_bootstrap"]["Q4: weighted iii - plain iii"]["per_pair_diff"] = [
        123.0,
        123.0,
        123.0,
    ]  # the mean alone would pass
    sel_path.write_text(json.dumps(sel))
    final_test.freeze()
    with pytest.raises(final_test.CheckFailed, match="seed pair"):
        final_test.dryrun()


def test_a_missing_test_prediction_is_found_before_the_claim(tmp_path, monkeypatch):
    path = write_case(tmp_path, monkeypatch)
    final_test = add_test_and_records(tmp_path, monkeypatch, path)
    groups.OUT.joinpath("config_choice.json").write_text(path.read_text())
    evaluation.calibrate(path)
    final_test.freeze()
    final_test.dryrun()
    done_predicting(final_test)
    rid = evaluation.run_id("BASE", 0.003, 0.3, 2027)
    evaluation.prediction_file(
        "q1", "model_after_train-19", evaluation.checkpoint_sha(rid, "train-19.parquet"), "test.parquet"
    ).unlink()
    with pytest.raises(final_test.CheckFailed, match="no test predictions"):  # caught before the claim
        final_test.test("q1")
    assert not (final_test.FROZEN / "evaluation_q1.lock").exists()


def test_click_weighted_did_matches_definition():
    tok, base, y, user = frame(5000, 4)
    g = np.array([TOKENS.index(t) for t in tok])
    a, b = expit(base + 0.3), expit(base - 0.2)
    r = evaluation.did_click_bootstrap(y, [a], [b], g, user, np.arange(10), np.arange(10, 30), reps=50, seed=0)
    m1, m2 = g < 10, g >= 10
    want = (metrics.bce(y[m1], a[m1]).mean() - metrics.bce(y[m1], b[m1]).mean()) - (
        metrics.bce(y[m2], a[m2]).mean() - metrics.bce(y[m2], b[m2]).mean()
    )
    assert (
        abs(r["did"] - want) < 1e-12
        and abs(r["diff_set_1"] - (metrics.bce(y[m1], a[m1]).mean() - metrics.bce(y[m1], b[m1]).mean())) < 1e-12
    )


@pytest.mark.parametrize("case", ["q1_no_gauc", "q2_no_p_last", "q2_nan"])
def test_a_malformed_test_prediction_is_found_before_the_claim(tmp_path, monkeypatch, case):
    path = write_case(tmp_path, monkeypatch)
    final_test = add_test_and_records(tmp_path, monkeypatch, path)
    groups.OUT.joinpath("config_choice.json").write_text(path.read_text())
    evaluation.calibrate(path)
    final_test.freeze()
    final_test.dryrun()
    done_predicting(final_test)
    rid = evaluation.run_id("MoAE", 0.003, 0.3, 2026)
    name = "q1" if case.startswith("q1") else "q2"
    if name == "q1":
        f = evaluation.prediction_file(
            "q1", "model_after_train-19", evaluation.checkpoint_sha(rid, "train-19.parquet"), "test.parquet"
        )
        meta = json.loads(f.with_suffix(".json").read_text())
        meta.pop("gauc")  # caught before the claim, not as a KeyError after it
        f.with_suffix(".json").write_text(json.dumps(meta))
    else:
        f = evaluation.prediction_file("research", evaluation.CKPT, evaluation.checkpoint_sha(rid), "test.parquet")
        z = dict(np.load(f))
        if case == "q2_no_p_last":
            z.pop("p_last")
        else:
            z["p_last"] = z["p_last"].copy()
            z["p_last"][5] = np.nan
        np.savez(f, **z)
        meta = json.loads(f.with_suffix(".json").read_text())
        meta["sha256"] = hashlib.sha256(f.read_bytes()).hexdigest()  # the record matches; only the content is wrong
        f.with_suffix(".json").write_text(json.dumps(meta))
    with pytest.raises(final_test.CheckFailed):
        final_test.test(name)
    assert not (final_test.FROZEN / f"evaluation_{name}.lock").exists()
