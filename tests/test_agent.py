"""Hypothesis agent on synthetic data with a scripted LLM (no API key, no LightGBM)."""

import json

import numpy as np
import polars as pl

from cvr.agent.experiments import FEATURE_GROUPS
from cvr.agent.llm import AuditLog
from cvr.agent.scripted import scripted_llm
from cvr.agent.session import REPORT_SCHEMA, run_session, verify


def _menu(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n = 20_000
    users = rng.integers(0, 2_000, n) * 10_000
    hist_empty = (rng.uniform(size=n) < 0.1).astype(np.int8)
    match1 = np.where(hist_empty == 1, 0, rng.poisson(0.3, n)).astype(np.int16)
    p_full = np.where(match1 > 0, 0.12, 0.01)  # a strong history signal by construction
    y = (rng.uniform(size=n) < p_full).astype(np.int8)
    df = {"user_key": users, "y": y, "ad_feat1": rng.integers(2, 50, n).astype(np.int32),
          "group_class": rng.choice(["G1", "dense", "mid"], n), "hist_empty": hist_empty,
          "match_ad_feat0": np.zeros(n, np.int16), "match_ad_feat1": match1, "match_ad_feat2": np.zeros(n, np.int16),
          "high_visual": rng.uniform(size=n) < 0.2, "p_full": p_full.astype(np.float32)}
    for g in FEATURE_GROUPS:  # removing history_match loses the signal; other groups change nothing
        df[f"p_no_{g}"] = (np.full(n, y.mean()) if g == "history_match" else p_full).astype(np.float32)
    df["p_full_noise"] = np.clip(p_full + rng.normal(0, 0.002, n), 0.001, 0.5).astype(np.float32)  # noise adds nothing
    pl.DataFrame(df).write_parquet(tmp_path / "confirmation_predictions.parquet")
    (tmp_path / "precompute.json").write_text(json.dumps({"config": {"data": {"confirmation": ["a", "b"]}}}))
    return tmp_path


def test_scripted_session_enforces_rules(tmp_path):
    menu = _menu(tmp_path / "menu")
    out = tmp_path / "s"
    summary = run_session(menu, out, scripted_llm(AuditLog(out / "audit.jsonl")))
    rec = {r["hypothesis"]["id"]: r for r in json.loads((out / "records.json").read_text())}
    assert rec[1]["verdict"] == "confirmed"  # history_match helps by construction
    assert rec[4]["verdict"] == "needs_experiment" and rec[4]["result"]["tool"] == "ab_test_design"
    assert any("rerouted" in i for i in rec[4]["issues"])
    assert rec[5]["verdict"] == "invalid" and any("E99" in i for i in rec[5]["issues"])
    assert summary["verification"]["passed"], summary["verification"]
    assert (out / "audit.jsonl").read_text().count("\n") == 2  # planner + reporter


def test_verifier_flags_wrong_verdict_and_invented_number(tmp_path):
    menu = _menu(tmp_path / "menu")
    out = tmp_path / "s"
    run_session(menu, out, scripted_llm(AuditLog(out / "audit.jsonl")))
    records = json.loads((out / "records.json").read_text())
    report = json.loads((out / "report.json").read_text())
    code_verdict = next(r["verdict"] for r in records if r["hypothesis"]["id"] == report["items"][0]["id"])
    report["items"][0]["verdict"] = "opposite_direction" if code_verdict != "opposite_direction" else "confirmed"
    report["items"][1]["summary"] += " 전환율이 37.25% 올랐다"  # a number that exists nowhere
    check = verify(records, report)
    assert not check["passed"]
    assert check["verdict_mismatches"][0]["id"] == 1
    assert any(u["number"] == 37.25 for u in check["ungrounded_numbers"])
    assert set(REPORT_SCHEMA["required"]) == {"items", "overall"}


def test_negative_control_noise_group_is_not_confirmed(tmp_path):
    from cvr.agent.experiments import NEGATIVE_GROUP, Menu

    menu = Menu(_menu(tmp_path / "menu"))
    r = menu.ablation_effect(NEGATIVE_GROUP, "sparse_G1")
    assert r["estimate"] >= 0 or r["ci95"][1] >= 0  # adding pure noise never shows up as a clear gain
