"""Hypothesis agent on synthetic data with a scripted LLM test double (no API key, no LightGBM).

Each test mirrors a failure found in review: a zero-row arm produced NaN yet "confirmed"; an unknown
evidence id was still confirmed; report numbers were checked by "exists somewhere" (sign flips and invented
numbers passed); per-session multiple testing was the only family.
"""

import json

import numpy as np
import polars as pl

from cvr.agent.experiments import FEATURE_GROUPS, NEGATIVE_GROUP, PLACEBO_REPLICATES, Menu
from cvr.agent.llm import AuditLog
from cvr.agent.scripted import scripted_factory
from cvr.agent.session import check_text, render_result, run


def _menu(tmp_path, n=20_000):
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    users = rng.integers(0, 2_000, n) * 10_000
    hist_empty = (rng.uniform(size=n) < 0.1).astype(np.int8)
    match1 = np.where(hist_empty == 1, 0, rng.poisson(0.3, n)).astype(np.int16)
    p_full = np.where(match1 > 0, 0.12, 0.01)  # a strong history signal by construction
    y = (rng.uniform(size=n) < p_full).astype(np.int8)
    df = {"user_key": users, "y": y, "ad_feat1": rng.integers(2, 50, n).astype(np.int32),
          "group_class": rng.choice(["G1", "dense", "mid"], n), "hist_empty": hist_empty,
          # same_item_in_history never happens -> the "with feature" arm of that association is EMPTY
          "match_ad_feat0": np.zeros(n, np.int16), "match_ad_feat1": match1, "match_ad_feat2": np.zeros(n, np.int16),
          "high_visual": rng.uniform(size=n) < 0.2, "p_full": p_full.astype(np.float32)}
    for g in FEATURE_GROUPS:
        df[f"p_no_{g}"] = (np.full(n, y.mean()) if g == "history_match" else p_full).astype(np.float32)
    for r in range(PLACEBO_REPLICATES):
        df[f"p_full_noise_{r}"] = np.clip(p_full + rng.normal(0, 0.002, n), 0.001, 0.5).astype(np.float32)
    pl.DataFrame(df).write_parquet(tmp_path / "confirmation_predictions.parquet")
    (tmp_path / "precompute.json").write_text(json.dumps({"config": {"data": {"confirmation": ["a", "b"]}}}))
    return tmp_path


def _run(tmp_path, sessions=2, negative=None):
    menu = _menu(tmp_path / "menu")
    out = tmp_path / "run"
    summary = run(menu, out, scripted_factory(AuditLog(out / "audit.jsonl"), negative), sessions, negative)
    records = json.loads((out / "records.json").read_text())
    return summary, records, out


def test_empty_arm_is_insufficient_data_not_a_verdict(tmp_path):
    menu = Menu(_menu(tmp_path / "menu"))
    r = menu.association("same_item_in_history", "all")
    assert r["status"] == "insufficient_data" and "estimate" not in r and "p_value" not in r
    summary, records, _ = _run(tmp_path)
    h5 = [x for x in records if x["hypothesis"]["id"] == 5]
    assert all(x["verdict"] == "insufficient_data" and "bh_adjusted_p" not in x for x in h5)


def test_unknown_evidence_id_invalidates_the_hypothesis(tmp_path):
    _, records, _ = _run(tmp_path)
    h4 = [x for x in records if x["hypothesis"]["id"] == 4]
    assert all(x["verdict"] == "invalid" and "result" not in x for x in h4)
    assert any("E999" in i for i in h4[0]["issues"])


def test_causal_claim_is_routed_and_valid_claim_confirmed(tmp_path):
    summary, records, _ = _run(tmp_path)
    by = {(x["session"], x["hypothesis"]["id"]): x for x in records}
    assert by[(1, 3)]["verdict"] == "needs_experiment" and by[(1, 3)]["result"]["tool"] == "ab_test_design"
    assert by[(1, 3)]["result"]["design_effect"] >= 1.0
    assert by[(1, 1)]["verdict"] == "confirmed"


def test_one_bh_family_over_deduplicated_tests_and_frozen_plans(tmp_path):
    summary, records, out = _run(tmp_path, sessions=2)
    # both sessions propose the same tests: each unique test runs once and enters the family once
    assert summary["hypotheses"] == 10 and summary["unique_tests"] == 4
    assert summary["tested_in_bh_family"] == 2  # H1 (ablation) and H2 (association); H3 causal, H4 invalid, H5 empty
    assert (out / "plans_frozen.json").exists() and len(summary["plans_sha256"]) == 64
    assert summary["stability_mean_jaccard"] == 1.0


def test_numbers_come_from_code_and_llm_text_is_checked(tmp_path):
    summary, records, out = _run(tmp_path)
    rec = next(x for x in records if x["hypothesis"]["id"] == 1)
    sentence = render_result(rec)
    assert f"{rec['result']['estimate']:+.6f}" in sentence  # sign and value come from the tool output
    assert summary["llm_text_with_digits"] == 0 and summary["llm_text_causal_word_flags"] == 0
    # what the review injected into a report: a sign flip / invented number and a causal claim
    bad = check_text("광고 이미지를 바꾸면 전환율이 15% 상승한다는 인과 효과가 입증됐다", "not_confirmed")
    assert bad["digits"] > 0 and {"인과", "입증"} <= set(bad["causal_words"])
    assert check_text("효과 추정치는 +0.012345였다", "confirmed")["digits"] > 0


def test_negative_control_status_and_placebo_calibration(tmp_path):
    summary, records, _ = _run(tmp_path, sessions=2, negative=2)
    nc = summary["negative_control"]
    assert nc["selected_by_planner"] and nc["status"] in ("not confirmed", "wrongly confirmed")
    pc = summary["placebo_calibration"]
    assert pc["tests"] == 2 * PLACEBO_REPLICATES and pc["computable"] == pc["tests"]
    # a run without the control session reports "not run", never "passed"
    s2, _, _ = _run(tmp_path / "again", sessions=1, negative=None)
    assert s2["negative_control"]["status"] == "not run"
    assert NEGATIVE_GROUP == "user_score"
