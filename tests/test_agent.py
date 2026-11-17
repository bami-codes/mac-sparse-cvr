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


def test_expected_direction_must_fit_the_tool(tmp_path):
    # run 1 with HyperCLOVA X: an association labelled "group_helps" was judged "opposite_direction" although the
    # data agreed with the statement; such plans are now invalid instead of silently flipped
    from cvr.agent.session import validate

    h = {"claim_type": "associational", "tool": "association", "feature": "same_category_in_history",
         "segment": "all", "expected": "group_helps", "evidence": ["E3"]}
    assert any("does not fit tool" in i for i in validate(h, {"E3"}))
    h2 = dict(h, expected="higher_cvr_with_feature")
    assert validate(h2, {"E3"}) == []
    h3 = {"claim_type": "predictive", "tool": "ablation_effect", "group": "none", "segment": "all",
          "expected": "group_helps", "evidence": ["E3"]}
    assert any("needs argument group" in i for i in validate(h3, {"E3"}))


def test_identifiers_are_not_numbers():
    assert check_text("ad_feat1과 sparse_G1, E3 근거에서 G1에서는 다르다", "not_confirmed")["digits"] == 0
    assert check_text("효과는 0.0123이다", "not_confirmed")["digits"] > 0


def test_model_comparisons_are_judged_against_the_placebo_null(tmp_path):
    menu = Menu(_menu(tmp_path / "menu"))
    r = menu.ablation_effect("history_match", "sparse_G1")
    assert r["test"].startswith("training-variance") and r["placebo_replicates"] == PLACEBO_REPLICATES
    assert r["effect_vs_placebo"] == r["estimate"] - r["placebo_mean"] and r["p_value"] < 0.05
    neg = menu.ablation_effect(NEGATIVE_GROUP, "sparse_G1", replicate=0)
    assert neg["placebo_replicates"] == PLACEBO_REPLICATES - 1  # judged against the OTHER noise replicates
    pc = menu.placebo_calibration()
    assert "rejections_adjusted_leave_one_out" in pc


def test_per_test_plan_items_normalize_and_validate():
    from cvr.agent.session import normalize, plan_schema, validate

    item = {"id": 1, "statement": "같은 상품 이력이 있으면 전환율이 높다", "claim_type": "associational",
            "evidence": ["[E3]"], "test": "association", "segment": "all", "feature": "same_item_in_history",
            "direction": "higher_cvr_with_feature"}
    h = normalize(item)
    assert h["tool"] == "association" and h["expected"] == "higher_cvr_with_feature" and h["evidence"] == ["E3"]
    assert validate(h, {"E3"}) == []
    abl = normalize({"id": 2, "statement": "s", "claim_type": "associational", "evidence": ["E2"],
                     "test": "ablation_effect", "segment": "sparse_G1", "group": "history_match"})
    assert abl["expected"] == "group_helps" and any("not allowed" in i for i in validate(abl, {"E2"}))
    assert len(plan_schema(["history_match"])["properties"]["hypotheses"]["items"]["anyOf"]) == 4


def test_rms_placebo_null_cannot_flip_a_zero_effect(tmp_path):
    # run 2 artefact: placebo models trained longer, their mean (-2.5e-5) was subtracted, and an uninformative
    # ablation (~0) became a "significant opposite" effect. The fixed-rounds menu uses an RMS null at zero.
    folder = _menu(tmp_path / "menu")
    df = pl.read_parquet(folder / "confirmation_predictions.parquet")
    shifted = {f"p_full_noise_{r}": (df["p_full"] * 0.999).cast(pl.Float32) for r in range(PLACEBO_REPLICATES)}
    df.with_columns(**shifted).write_parquet(folder / "confirmation_predictions.parquet")
    meta = json.loads((folder / "precompute.json").read_text())
    for mode, can_flip in (("mean_shift", True), ("rms", False)):
        (folder / "precompute.json").write_text(json.dumps(meta | {"placebo_null": mode}))
        r = Menu(folder).ablation_effect("visual_similarity", "all")  # truly uninformative here: estimate 0
        assert r["estimate"] == 0.0
        if not can_flip:
            assert r["effect_vs_placebo"] == 0.0 and r["p_value"] == 1.0
