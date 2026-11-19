"""Hypothesis agent on synthetic data with a scripted LLM test double (no API key, no LightGBM).

Each test pins down a failure the agent must not have: an empty arm that yields NaN yet "confirmed", a confirmed
verdict for an unknown evidence id, report numbers that are merely "present somewhere" (so sign flips and invented
numbers slip through), and multiple-testing control per session instead of per run.
"""

import json

import numpy as np
import polars as pl

from cvr.agent.experiments import FEATURE_GROUPS, NEGATIVE_GROUP, NOISE_REPLICATES, TEST_MODEL_COMPARISON, Menu
from cvr.agent.llm import AuditLog
from cvr.agent.scripted import scripted_factory
from cvr.agent.session import check_text, render_result, run


def _menu(tmp_path, n=60_000):
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
    for r in range(NOISE_REPLICATES):
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
    assert summary["items_missing_from_report"] == 0  # the checks above saw real text (test double bug, fixed)
    reports = json.loads((out / "reports.json").read_text())
    assert all(it["interpretation"] for rep in reports for it in rep["items"])
    # a report with a sign flip or an invented number, and a causal claim
    bad = check_text("광고 이미지를 바꾸면 전환율이 15% 상승한다는 인과 효과가 입증됐다", "not_confirmed")
    assert bad["digits"] > 0 and {"인과", "입증"} <= set(bad["causal_words"])
    assert check_text("효과 추정치는 +0.012345였다", "confirmed")["digits"] > 0


def test_negative_control_status_and_training_sensitivity(tmp_path):
    summary, records, _ = _run(tmp_path, sessions=2, negative=2)
    nc = summary["negative_control"]
    assert nc["selected_by_planner"] and nc["status"] in ("planted false claim not confirmed",
                                                          "planted false claim confirmed (not caught)")
    assert "does not detect" in nc["note"]
    ts = summary["training_sensitivity"]
    assert ts["comparisons"] == 2 * NOISE_REPLICATES and ts["computable"] == ts["comparisons"]
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


def test_model_comparisons_use_the_fixed_model_bootstrap_and_report_sensitivity(tmp_path):
    menu = Menu(_menu(tmp_path / "menu"))
    r = menu.ablation_effect("history_match", "sparse_G1")
    assert r["test"] == TEST_MODEL_COMPARISON and r["estimate"] < 0 and r["p_value"] < 0.05
    assert r["training_sensitivity"]["replicates"] == NOISE_REPLICATES and r["beyond_sensitivity_range"]
    neg = menu.ablation_effect(NEGATIVE_GROUP, "sparse_G1", replicate=0)
    assert neg["training_sensitivity"]["replicates"] == NOISE_REPLICATES - 1  # described by the OTHER replicates
    assert "not the false-positive rate" in menu.training_sensitivity()["note"]


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


def test_noise_models_never_change_a_verdict(tmp_path):
    # A null built from 4-5 noise models rejected 8-9% of the time at a nominal 5% on synthetic data, so noise
    # models are descriptive only: changing them may move the sensitivity, never the verdict.
    folder = _menu(tmp_path / "menu")
    before = Menu(folder).ablation_effect("visual_similarity", "all")
    df = pl.read_parquet(folder / "confirmation_predictions.parquet")
    df.with_columns(**{f"p_full_noise_{r}": (df["p_full"] * 0.9).cast(pl.Float32)
                       for r in range(NOISE_REPLICATES)}).write_parquet(folder / "confirmation_predictions.parquet")
    after = Menu(folder).ablation_effect("visual_similarity", "all")  # truly uninformative here: estimate 0
    assert after["training_sensitivity"]["max_abs"] != before["training_sensitivity"]["max_abs"]
    assert all(after[k] == before[k] for k in ("estimate", "ci95", "p_value")) and after["estimate"] == 0.0


class _Reporter:
    def structured(self, system, user, schema, role, **kw):
        return {"items": [{"id": x["id"], "interpretation": "실험 설계가 필요하다", "next_step": "실험을 계획한다"}
                          for x in json.loads(user)], "overall": "설계만 계산했다"}


def test_requests_that_differ_in_any_argument_are_different_tests(tmp_path):
    # two A/B requests (a 1% and a 10% effect) must not share a key and report the same sample size
    from cvr.agent.session import call_key, normalize

    base = {"claim_type": "causal", "test": "ab_test_design", "segment": "all", "evidence": ["E3"], "statement": "s"}
    h1, h2 = normalize(base | {"id": 1, "relative_mde": 0.01}), normalize(base | {"id": 2, "relative_mde": 0.10})
    assert call_key(h1) != call_key(h2)
    folder = _menu(tmp_path / "menu")
    plans = tmp_path / "plans.json"
    plans.write_text(json.dumps([{"session": 1, "negative_control": False, "plan": {"hypotheses": [h1, h2]}}]))
    summary = run(folder, tmp_path / "run", lambda k: _Reporter(), 1, None, from_plans=plans)
    records = json.loads((tmp_path / "run" / "records.json").read_text())
    menu = Menu(folder)
    assert summary["unique_tests"] == 2
    for rec, mde in zip(records, (0.01, 0.10)):
        assert rec["result"]["relative_mde"] == mde
        assert rec["result"]["clicks_per_arm"] == menu.ab_test_design("all", mde)["clicks_per_arm"]
    assert records[0]["result"]["clicks_per_arm"] > records[1]["result"]["clicks_per_arm"]


def test_verdict_answers_the_code_question_not_the_planner_sentence(tmp_path):
    # A planner sentence can say the opposite of what the tool tests and still pass every structural rule, so the
    # verdict belongs to the question the code wrote, and the LLM's text is shown as unverified.
    from cvr.agent.session import normalize

    h = normalize({"id": 1, "statement": "구매 이력 특성을 포함한 모델은 제외한 모델보다 확증 자료의 LogLoss가 높다.",
                   "claim_type": "predictive", "evidence": ["E2"], "test": "ablation_effect",
                   "group": "history_match", "segment": "sparse_G1"})
    folder = _menu(tmp_path / "menu")
    plans = tmp_path / "plans.json"
    plans.write_text(json.dumps([{"session": 1, "negative_control": False, "plan": {"hypotheses": [h]}}]))
    summary = run(folder, tmp_path / "run", lambda k: _Reporter(), 1, None, from_plans=plans)
    rec = json.loads((tmp_path / "run" / "records.json").read_text())[0]
    assert rec["verdict"] == "confirmed" and rec["result"]["estimate"] < 0
    assert "넣은 모델의 확증 자료 평균 LogLoss가 뺀 모델보다 낮은가" in rec["test_question"]
    md = (tmp_path / "run" / "report.md").read_text()
    assert "검정 질문 (코드)" in md and "플래너 제안 문장 (LLM, 의미 검증 안 함)" in md
    assert "NOT verified" in summary["verdict_scope"] and "planner_structural_compliance" in summary


def test_user_bootstrap_needs_more_than_one_user(tmp_path):
    # 60,000 clicks from one user pass a row/positive gate, yet a user bootstrap over one user gives a zero-width
    # interval (p = 0.002)
    df = pl.read_parquet(_menu(tmp_path / "menu") / "confirmation_predictions.parquet")
    meta = {"config": {"data": {"confirmation": ["synthetic"]}}}
    one = Menu.from_frame(df.with_columns(pl.lit(0).alias("user_key")), meta)
    r = one.ablation_effect("history_match", "all", sensitivity=False)
    assert r["status"] == "insufficient_data" and r["arms"]["all"]["users"] == 1 and "p_value" not in r
    ok = Menu.from_frame(df, meta).ablation_effect("history_match", "all", sensitivity=False)
    a = ok["arms"]["all"]
    assert ok["status"] == "ok" and a["users"] > 1000 and 0 < a["max_user_positive_share"] < 0.05
    assert a["converting_users"] <= a["users"] and a["effective_users"] <= a["users"]


def test_gate_is_an_explicit_parameter(tmp_path):
    # the synthetic check must state which gate it used: same data, different gate, different computable set
    df = pl.read_parquet(_menu(tmp_path / "menu") / "confirmation_predictions.parquet").head(3000)
    meta = {"config": {"data": {"confirmation": ["synthetic"]}}}
    default = Menu.from_frame(df, meta).ablation_effect("history_match", "all", sensitivity=False)
    loose = Menu.from_frame(df, meta, min_positives=10, min_users=1).ablation_effect("history_match", "all",
                                                                                     sensitivity=False)
    assert default["status"] == "insufficient_data" and loose["status"] == "ok"


def test_two_step_run_executes_exactly_the_committed_plans(tmp_path):
    # final run: plans are written (menu never loaded), committed, then executed from that file
    one = tmp_path / "one"
    s_one = run(_menu(tmp_path / "menu"), one, scripted_factory(AuditLog(one / "audit.jsonl"), 2), 2, 2)
    two = tmp_path / "two"
    frozen = run(tmp_path / "no_menu_here", two, scripted_factory(AuditLog(two / "audit.jsonl"), 2), 2, 2,
                 plans_only=True)
    assert frozen["plans_sha256"] == s_one["plans_sha256"] and not (two / "records.json").exists()
    committed = tmp_path / "committed_plans.json"
    committed.write_bytes((two / "plans_frozen.json").read_bytes())
    s_two = run(tmp_path / "menu", two, scripted_factory(AuditLog(two / "audit.jsonl"), 2), 2, 2, from_plans=committed)
    assert (two / "plans_frozen.json").read_bytes() == committed.read_bytes()
    assert s_two == s_one
    # same values (key order differs: the frozen file is written with sorted keys)
    assert json.loads((two / "records.json").read_text()) == json.loads((one / "records.json").read_text())
    assert json.loads((two / "reports.json").read_text()) == json.loads((one / "reports.json").read_text())


def test_comparisons_impossible_by_definition_are_invalid(tmp_path):
    # final run (349cd32): feature = segment, and empty_history inside has_history, were reported "insufficient data"
    from itertools import product

    from cvr.agent.experiments import ASSOC_FEATURES, SEGMENTS, empty_arm_by_definition
    from cvr.agent.session import normalize, validate

    base = {"claim_type": "associational", "test": "association", "evidence": ["E3"], "statement": "s"}
    for feature, segment, direction in [("high_visual_similarity", "high_visual_similarity", "higher_cvr_with_feature"),
                                        ("empty_history", "has_history", "higher_cvr_with_feature")]:
        h = normalize(base | {"id": 1, "feature": feature, "segment": segment, "direction": direction})
        assert any("impossible by definition" in i for i in validate(h, {"E3"}))
    inter = normalize({"id": 2, "claim_type": "predictive", "test": "interaction", "group": "context",
                       "segment": "sparse_G1", "segment_b": "sparse_G1", "evidence": ["E3"], "statement": "s"})
    assert any("segment and segment_b are the same" in i for i in validate(inter, {"E3"}))

    # the rule matches the masks: on data where every combination the feature code allows occurs (only rule:
    # an empty history has no same-item match), an arm is empty exactly when the rule says so
    folder = _menu(tmp_path / "menu")
    path = folder / "confirmation_predictions.parquet"
    df = pl.read_parquet(path)
    rng = np.random.default_rng(1)
    n = df.height
    empty = (rng.uniform(size=n) < 0.3).astype(np.int8)
    df = df.with_columns(pl.Series("hist_empty", empty),
                         pl.Series("match_ad_feat0", np.where(empty == 1, 0, rng.integers(0, 2, n)).astype(np.int16)),
                         pl.Series("match_ad_feat1", rng.integers(0, 2, n).astype(np.int16)),
                         pl.Series("match_ad_feat2", rng.integers(0, 2, n).astype(np.int16)),
                         pl.Series("high_visual", rng.uniform(size=n) < 0.5))
    df.write_parquet(path)
    menu = Menu(folder)
    for feature, segment in product(ASSOC_FEATURES, SEGMENTS):
        r = menu.association(feature, segment)
        assert (min(r["rows_with"], r["rows_without"]) == 0) == (empty_arm_by_definition(feature, segment) is not None)


def test_negative_control_runs_even_if_the_planner_skips_it(tmp_path):
    # final run: the planner of the control session did not pick E8, so the control was "not conducted"
    folder = _menu(tmp_path / "menu")
    out = tmp_path / "run"
    skip = scripted_factory(AuditLog(out / "audit.jsonl"), None)  # session 2 sees E8 but proposes no user_score test
    frozen = run(folder, out, skip, 2, 2, plans_only=True)
    assert frozen["negative_control_session"] and not frozen["planted_claim_selected_by_planner"]
    plans = json.loads((out / "plans_frozen.json").read_text())
    assert [("control" in p) for p in plans] == [False, True]  # frozen with the plans, so it is in the BH family
    summary = run(folder, out, skip, 2, 2, from_plans=out / "plans_frozen.json")
    nc = summary["negative_control"]
    assert not nc["selected_by_planner"] and nc["added_by_code"] and nc["status"] != "not conducted (planner did not select it)"
    assert summary["hypotheses"] == 10 and summary["control_hypotheses"] == 1  # planner metrics exclude the control
    records = json.loads((out / "records.json").read_text())
    ctrl = [r for r in records if r["source"] == "control"]
    assert len(ctrl) == 1 and ctrl[0]["hypothesis"]["group"] == NEGATIVE_GROUP and ctrl[0]["verdict"] != "invalid"
    assert "대조 (코드가 추가, 플래너 제안 아님)" in (out / "report.md").read_text()


class _SilentReporter(_Reporter):
    def structured(self, system, user, schema, role, **kw):
        return super().structured(system, user, schema, role, **kw) | {"overall": " "}


def test_empty_llm_summary_is_counted_and_a_code_summary_is_shown(tmp_path):
    # final run: the reporter's overall text was empty in all five sessions, and no metric counted it
    from cvr.agent.session import normalize

    h = normalize({"id": 1, "statement": "s", "claim_type": "predictive", "evidence": ["E2"], "test": "ablation_effect",
                   "group": "history_match", "segment": "sparse_G1"})
    folder = _menu(tmp_path / "menu")
    plans = tmp_path / "plans.json"
    plans.write_text(json.dumps([{"session": 1, "negative_control": False, "plan": {"hypotheses": [h]}}]))
    summary = run(folder, tmp_path / "run", lambda k: _SilentReporter(), 1, None, from_plans=plans)
    assert summary["reports_with_empty_overall"] == 1
    md = (tmp_path / "run" / "report.md").read_text()
    assert "종합 (LLM): ⚠️ 비어 있음" in md and "종합 (코드, 판정에서 작성): 가설 1개: 확인됨 1." in md
    summary2 = run(folder, tmp_path / "run2", lambda k: _Reporter(), 1, None, from_plans=plans)
    assert summary2["reports_with_empty_overall"] == 0
