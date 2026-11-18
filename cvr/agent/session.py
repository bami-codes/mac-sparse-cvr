"""Hypothesis agent run: several planner sessions, frozen before any confirmation result, then one
deduplicated execution with a single multiple-testing family, then reports and checks.

    uv run python -m cvr.agent.session runs/agent_menu runs/agent_runs/<name> --sessions 5 --negative-control 5
    # or in two steps, so the plans can be committed before any execution:
    uv run python -m cvr.agent.session runs/agent_menu runs/agent_runs/<name> --sessions 5 --negative-control 5 --plans-only
    uv run python -m cvr.agent.session runs/agent_menu runs/agent_runs/<name> --from-plans <committed plans_frozen.json>
    uv run python -m cvr.agent.session runs/agent_menu runs/agent_runs/<name> --scripted     # no API key

What the code decides (never the LLM):
- validity: evidence ids must exist (else the hypothesis is invalid and not run); the tool must fit the
  claim type; causal claims are never tested on observational data (routed to ab_test_design,
  verdict "needs_experiment");
- statistics: tools return "insufficient_data" when an arm is too small or non-finite; such results never
  enter the multiple-testing step; a test is a tool call (tool name + every argument it runs with), each runs
  once, and Benjamini-Hochberg (q = 0.05) covers ALL unique tested calls of the run (the family is fixed by
  the frozen plans). Model comparisons are judged by the paired user bootstrap with the fitted models held
  fixed; noise-model sensitivity is shown next to them and never changes a verdict;
- every number in the report: result sentences are rendered by code from the tool output;
- what a verdict is about: a test question written by code from the tool call (`test_question`). The
  planner's sentence and the reporter's text are shown as LLM text whose meaning is NOT verified.
What the LLM writes: an interpretation and a next step per hypothesis, with NO digits (checked). A keyword
scan flags causal wording on non-experimental results; it is a heuristic guard, not a verification of meaning.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np

from .evidence import EVIDENCE, FALSE_EVIDENCE, evidence_text
from .experiments import ASSOC_FEATURES, FEATURE_GROUPS, NEGATIVE_GROUP, SEGMENTS, Menu
from .llm import AuditLog, ClovaClient

VERDICTS = ["confirmed", "opposite_direction", "not_confirmed", "insufficient_data", "needs_experiment", "invalid"]
VERDICT_KO = {"confirmed": "확인됨", "opposite_direction": "반대 방향으로 유의", "not_confirmed": "확인되지 않음",
              "insufficient_data": "자료 부족으로 검증 불가", "needs_experiment": "관측 데이터로 검증 불가 (A/B 필요)",
              "invalid": "무효"}
EXPECTED = ["group_helps", "group_helps_more_in_segment", "higher_cvr_with_feature", "lower_cvr_with_feature",
            "needs_experiment"]
ALLOWED_TOOLS = {"predictive": {"ablation_effect", "interaction"}, "associational": {"association"},
                 "causal": {"ab_test_design"}}
# the expected direction must mean something for the chosen tool (a mismatch silently flipped verdicts in run 1)
COMPATIBLE_EXPECTED = {"ablation_effect": {"group_helps"}, "interaction": {"group_helps_more_in_segment"},
                       "association": {"higher_cvr_with_feature", "lower_cvr_with_feature"}}
REQUIRED_ARGS = {"ablation_effect": ["group"], "interaction": ["group", "segment_b"], "association": ["feature"]}
CAUSAL_WORDS = ["인과", "입증", "증명", "때문에", "덕분에", "원인", "증가시", "늘린다", "늘려", "높인다", "높여", "끌어올"]


def plan_schema(groups: list[str]) -> dict:
    """One object shape per test (anyOf), so a test can only carry the fields and directions that mean
    something for it. In run 1 a flat schema let the planner pair any tool with any 'expected' value, and
    20 of 21 hypotheses had a direction that did not fit their tool."""
    common = {"id": {"type": "integer"}, "statement": {"type": "string"},
              "claim_type": {"type": "string", "enum": ["predictive", "associational", "causal"]},
              "evidence": {"type": "array", "minItems": 1, "items": {"type": "string"}},
              "segment": {"type": "string", "enum": SEGMENTS}}
    req = ["id", "statement", "claim_type", "evidence", "test", "segment"]
    variants = [
        {"type": "object", "required": req + ["group"], "properties": common | {
            "test": {"type": "string", "enum": ["ablation_effect"]}, "group": {"type": "string", "enum": groups}}},
        {"type": "object", "required": req + ["group", "segment_b"], "properties": common | {
            "test": {"type": "string", "enum": ["interaction"]}, "group": {"type": "string", "enum": groups},
            "segment_b": {"type": "string", "enum": SEGMENTS}}},
        {"type": "object", "required": req + ["feature", "direction"], "properties": common | {
            "test": {"type": "string", "enum": ["association"]}, "feature": {"type": "string", "enum": list(ASSOC_FEATURES)},
            "direction": {"type": "string", "enum": ["higher_cvr_with_feature", "lower_cvr_with_feature"]}}},
        {"type": "object", "required": req + ["relative_mde"], "properties": common | {
            "test": {"type": "string", "enum": ["ab_test_design"]},
            "relative_mde": {"type": "number", "minimum": 0.01, "maximum": 0.5}}},
    ]
    return {"type": "object", "required": ["hypotheses"], "properties": {"hypotheses": {
        "type": "array", "minItems": 3, "maxItems": 8, "items": {"anyOf": variants}}}}


IMPLIED_EXPECTED = {"ablation_effect": "group_helps", "interaction": "group_helps_more_in_segment",
                    "ab_test_design": "needs_experiment"}


def normalize(h: dict) -> dict:
    """Plan item -> internal hypothesis. Accepts the per-test format (test + direction) and the run-1 flat
    format (tool + expected) unchanged. Evidence ids like "[E3]" are normalized to "E3"."""
    h = dict(h)
    if "test" in h and "tool" not in h:
        h["tool"] = h.pop("test")
        h["expected"] = h.pop("direction", None) or IMPLIED_EXPECTED.get(h["tool"], "none")
    for k in ("group", "feature", "segment_b"):
        h.setdefault(k, "none")
    h["evidence"] = [str(e).strip().strip("[]").strip() for e in h.get("evidence", [])]
    return h


REPORT_SCHEMA = {"type": "object", "required": ["items", "overall"], "properties": {
    "items": {"type": "array", "items": {"type": "object", "required": ["id", "interpretation", "next_step"],
                                         "properties": {"id": {"type": "integer"}, "interpretation": {"type": "string"},
                                                        "next_step": {"type": "string"}}}},
    "overall": {"type": "string"}}}

PLANNER_SYSTEM = """너는 광고 전환 분석가다. 주어진 근거 자료만 보고, 클릭한 사용자를 전환으로 이끄는 단서에 대한 검증 가능한 가설을 3~8개 제안한다.
검증 방법(test)은 네 가지이고, 각각 검증하는 것이 다르다:
- ablation_effect (claim_type=predictive): 같은 학습 조건(같은 데이터·시드·반복 횟수)에서 특성 그룹(group)을 빼고 학습한 모델이, 확증 자료의 그 세그먼트에서 전체 모델보다 LogLoss가 나쁜가. 이번에 학습한 두 모델의 비교이며, 그 특성이 일반적으로 전환 정보를 가진다는 증명은 아니다.
- interaction (claim_type=predictive): 위 두 모델의 차이가 segment에서 segment_b보다 더 큰가.
- association (claim_type=associational): 특성(feature)이 있는 클릭과 없는 클릭의 전환율이 다른가. direction에 전환율이 높을지(higher) 낮을지(lower)를 쓴다. 인과가 아니다.
- ab_test_design (claim_type=causal): 무언가를 바꾸면 전환이 늘어나는지는 관측 데이터로 확인할 수 없으므로, 필요한 실험 규모만 계산한다.
규칙:
- evidence에는 근거 id만 쓴다 (예: "E3"). 근거에 없는 사실을 지어내지 않는다.
- 문장(statement)은 고른 검증 방법이 실제로 확인하는 내용과 일치해야 한다.
- 숫자 결과를 예상해서 쓰지 않는다. 결과는 코드가 계산한다."""

REPORTER_SYSTEM = """너는 분석 보고서 작성자다. 가설별로 코드가 계산한 판정과 결과가 주어진다.
규칙:
- 숫자(아라비아 숫자)를 전혀 쓰지 않는다. 결과 수치와 판정 문장은 코드가 따로 붙인다.
- 판정을 바꾸거나 더 강하게 표현하지 않는다. '확인되지 않음', '자료 부족', 'A/B 필요'인 가설을 사실처럼 쓰지 않는다.
- associational 결과는 인과로 해석하지 않는다. 예측에 도움이 된다는 결과도 '바꾸면 전환이 늘어난다'는 뜻이 아니다.
- predictive 결과는 '같은 학습 조건으로 한 번 학습한 두 모델의 확증 자료상 성능 차이'다. 다른 학습 실행이나 다른 날짜에서도 같다거나, 특성 자체가 전환 정보를 가진다는 뜻으로 넓히지 않는다. 결과 문장의 '학습 민감도'는 정보 없는 잡음 특성을 넣었을 때의 차이로, 판정에는 쓰이지 않은 참고 정보다.
- 판정은 test_question(코드가 쓴 검정 질문)에 대한 것이다. planner_statement는 참고용 원래 제안이며 검정 질문과 다를 수 있다. 다르면 planner_statement를 사실처럼 쓰지 말고 검정 질문의 결과만 해석한다.
- interpretation에는 이 결과가 무엇을 뜻하고 무엇을 뜻하지 않는지, next_step에는 다음에 할 실험이나 확인을 한 문장씩 쓴다."""


# ------------------------------------------------------------------ helpers
def tool_call(h: dict) -> tuple[str, dict]:
    """The exact tool call a valid hypothesis runs: tool name and every argument. Execution and the test key
    both come from here, so two requests that differ in any argument can never share a result (in review,
    A/B requests for a 1% and a 10% effect shared one key and both got the 1% sample size)."""
    if h["claim_type"] == "causal":  # always an experiment design, whatever observational tool was proposed
        return "ab_test_design", {"segment": h["segment"], "relative_mde": float(h.get("relative_mde") or 0.05)}
    if h["tool"] == "ablation_effect":
        return "ablation_effect", {"group": h["group"], "segment": h["segment"]}
    if h["tool"] == "interaction":
        return "interaction", {"group": h["group"], "segment": h["segment"], "segment_b": h["segment_b"]}
    return "association", {"feature": h["feature"], "segment": h["segment"]}


def test_question(h: dict) -> str:
    """The question a verdict answers, written by code from the tool call and the expected direction.

    Verdicts apply to THIS question, never to the planner's own sentence: in review, a statement saying the
    opposite of what the tool tested ("the model WITH history has HIGHER LogLoss") was reported "confirmed"
    with 100% rule compliance, because only the structured fields are validated, not the sentence's meaning."""
    name, a = tool_call(h)
    if name == "ablation_effect":
        return (f"[{a['segment']}] 같은 학습 조건으로 한 번 학습한 두 모델 중, '{a['group']}' 그룹을 넣은 모델의 "
                f"확증 자료 평균 LogLoss가 뺀 모델보다 낮은가?")
    if name == "interaction":
        return (f"'{a['group']}' 그룹을 넣은 모델이 뺀 모델보다 평균 LogLoss가 낮은 정도가 [{a['segment']}]에서 "
                f"[{a['segment_b']}]보다 큰가? (같은 두 모델)")
    if name == "association":
        more = "높은가" if h["expected"] == "higher_cvr_with_feature" else "낮은가"
        return (f"[{a['segment']}] 클릭에서 '{a['feature']}'가 있는 클릭의 전환율이 없는 클릭보다 {more}? "
                f"(연관 비교, 인과 아님)")
    return (f"[{a['segment']}]에서 상대 {a['relative_mde']:.0%} 변화를 검출하려면 사용자 무작위 실험에서 그룹당 클릭이 "
            f"몇 건 필요한가? (관측 데이터로 인과를 판정하지 않음)")


def call_key(h: dict) -> str:
    """Identity of a test (execution, deduplication, one BH entry): the tool call."""
    return json.dumps(tool_call(h), sort_keys=True)


def hypothesis_key(h: dict) -> str:
    """Identity of a hypothesis for the stability metric: the test plus its claim type and expected direction."""
    return json.dumps([h["claim_type"], h["expected"], tool_call(h)], sort_keys=True)


def validate(h: dict, known_ev: set[str]) -> list[str]:
    issues = []
    bad_ev = [e for e in h.get("evidence", []) if e not in known_ev]
    if not h.get("evidence"):
        issues.append("no evidence cited")
    if bad_ev:
        issues.append(f"evidence ids not in the package: {bad_ev}")
    if h["tool"] == "ab_test_design" and h["claim_type"] != "causal":
        issues.append(f"ab_test_design is for causal claims, got claim_type={h['claim_type']}")
    if h["claim_type"] != "causal" and h["tool"] not in ALLOWED_TOOLS[h["claim_type"]]:
        issues.append(f"tool {h['tool']} not allowed for a {h['claim_type']} claim")
    if h["claim_type"] != "causal":
        if h["tool"] in COMPATIBLE_EXPECTED and h["expected"] not in COMPATIBLE_EXPECTED[h["tool"]]:
            issues.append(f"expected={h['expected']} does not fit tool {h['tool']} "
                          f"(allowed: {sorted(COMPATIBLE_EXPECTED[h['tool']])})")
        for a in REQUIRED_ARGS.get(h["tool"], []):
            if h.get(a, "none") == "none":
                issues.append(f"tool {h['tool']} needs argument {a}")
    return issues


def run_tool(h: dict, menu: Menu) -> dict:
    name, args = tool_call(h)
    return getattr(menu, name)(**args)


def bh(p: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg: (significant mask, adjusted p-values)."""
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj <= q, adj


def arm_note(r: dict) -> str:
    """Size and concentration of the compared arms (users are what the bootstrap resamples)."""
    parts = [f"{name} 사용자 {a['users']:,}명·전환 사용자 {a['converting_users']:,}명·한 사용자 최대 전환 비중 "
             f"{(a['max_user_positive_share'] or 0):.1%}" for name, a in (r.get("arms") or {}).items()]
    return f" [비교 집단: {'; '.join(parts)}]" if parts else ""


def render_result(rec: dict) -> str:
    """The only place numbers enter the report."""
    v, r = rec["verdict"], rec.get("result") or {}
    head = f"판정: {VERDICT_KO[v]}"
    if v == "invalid":
        return f"{head} ({'; '.join(rec['issues'])})"
    if r.get("status") == "insufficient_data":
        return f"{head} ({r['reason']})"
    if r.get("tool") == "ab_test_design":
        days = f", 50:50 배분 시 약 {r['days_at_50_50_split']:.1f}일" if r.get("days_at_50_50_split") else ""
        return (f"{head}. 기준 전환율 {r['baseline_cvr']:.4f}, 상대 최소 검출 효과 {r['relative_mde']:.0%}, "
                f"그룹당 필요 클릭 {r['clicks_per_arm']:,} (설계 효과 {r['design_effect']:.2f} 반영, 독립 가정 시 "
                f"{r['clicks_per_arm_iid']:,}){days}{arm_note(r)}")
    unit = {"ablation_effect": "LogLoss 차이", "interaction": "제거 효과의 차이", "association": "전환율 차이"}[r["tool"]]
    body = (f"{head}. {unit} {r['estimate']:+.6f} [95% CI {r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}], "
            f"BH 조정 p = {rec['bh_adjusted_p']:.3f}")
    if r["tool"] == "association":
        return body + f", 전환율 {r['cvr_with']:.4f} vs {r['cvr_without']:.4f}" + arm_note(r)
    sens = r.get("training_sensitivity")
    scope = " (고정 학습 조건으로 학습한 두 모델의 비교, 확증 자료 사용자 부트스트랩)"
    if not sens:
        return body + scope + arm_note(r)
    where = "밖" if r["beyond_sensitivity_range"] else "안"
    return (body + scope + f". 학습 민감도 (판정에 쓰지 않음): 잡음 특성 하나만 추가한 모델 {sens['replicates']}개의 "
            f"같은 비교 최대 절댓값 {sens['max_abs']:.6f}, 이번 추정치는 그 범위 {where}" + arm_note(r))


IDENTIFIER = re.compile(r"[A-Za-z_]+\d+[A-Za-z_\d]*")  # ad_feat1, sparse_G1, G1, E3: names, not numbers


def check_text(text: str, verdict: str) -> dict:
    digits = re.findall(r"\d", IDENTIFIER.sub("", text))
    hits = [w for w in CAUSAL_WORDS if w in text]
    if verdict == "needs_experiment":  # describing a future experiment may use causal words; "proven" may not
        hits = [w for w in hits if w in ("입증", "증명")]
    return {"digits": len(digits), "causal_words": hits}


# ------------------------------------------------------------------ run
def run(menu_dir: Path, out: Path, make_llm, sessions: int, negative_control_session: int | None,
        from_plans: Path | None = None, plans_only: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    # 1) all plans first; frozen (hashed) before any tool sees confirmation data. The menu is not even loaded here.
    #    plans_only stops after this step, so the frozen file can be committed before execution; a later run
    #    with from_plans executes exactly that file (also used for re-analysis without new planner calls).
    plans = json.loads(from_plans.read_text()) if from_plans else []
    for k in ([] if from_plans else range(1, sessions + 1)):
        neg = k == negative_control_session
        groups = list(FEATURE_GROUPS) + ([NEGATIVE_GROUP] if neg else [])
        lists = {"group": groups, "feature": list(ASSOC_FEATURES), "segment": SEGMENTS}
        llm = make_llm(k)
        plan = llm.structured(PLANNER_SYSTEM, f"근거 자료:\n{evidence_text(neg)}\n\n사용 가능한 인자:\n"
                              f"{json.dumps(lists, ensure_ascii=False)}", plan_schema(groups), role=f"planner_s{k}")
        plans.append({"session": k, "negative_control": neg, "plan": plan})
    frozen = json.dumps(plans, ensure_ascii=False, sort_keys=True)
    (out / "plans_frozen.json").write_text(frozen)
    plans_sha = hashlib.sha256(frozen.encode()).hexdigest()
    if plans_only:
        return {"sessions": len(plans), "plans_sha256": plans_sha, "plans_file": str(out / "plans_frozen.json")}
    menu = Menu(menu_dir)

    # 2) validate, deduplicate, execute each unique test once, one BH family for the whole run
    records, unique = [], {}
    for p in plans:
        known = {e["id"] for e in EVIDENCE} | ({FALSE_EVIDENCE["id"]} if p["negative_control"] else set())
        for raw in p["plan"]["hypotheses"]:
            h = normalize(raw)
            issues = validate(h, known)
            rec = {"session": p["session"], "hypothesis": h, "issues": issues, "hypothesis_key": hypothesis_key(h)}
            if issues:
                rec["verdict"] = "invalid"
            else:
                key = rec["key"] = call_key(h)
                rec["test_question"] = test_question(h)
                if key not in unique:
                    unique[key] = run_tool(h, menu)
                rec["result"] = unique[key]
                if h["claim_type"] == "causal":
                    if h["tool"] != "ab_test_design":
                        rec["issues"].append("causal claim rerouted to ab_test_design")
                    rec["verdict"] = "needs_experiment" if rec["result"]["status"] == "ok" else "insufficient_data"
                elif rec["result"]["status"] != "ok":
                    rec["verdict"] = "insufficient_data"
                else:
                    rec["verdict"] = None
            records.append(rec)
    tested_keys = sorted({r["key"] for r in records if r["verdict"] is None})
    if tested_keys:
        sig, adj = bh(np.array([unique[k]["p_value"] for k in tested_keys]))
        bh_by_key = {k: (bool(s), float(a)) for k, s, a in zip(tested_keys, sig, adj)}
        for r in records:
            if r["verdict"] is None:
                s, a = bh_by_key[r["key"]]
                r["bh_significant"], r["bh_adjusted_p"] = s, a
                want_negative = r["hypothesis"]["expected"] in ("group_helps", "group_helps_more_in_segment",
                                                                "lower_cvr_with_feature")
                if not s:
                    r["verdict"] = "not_confirmed"
                else:
                    r["verdict"] = "confirmed" if (r["result"]["estimate"] < 0) == want_negative else "opposite_direction"

    # 3) reports: LLM interpretation without digits + code-rendered results; checks
    reports = []
    for p in plans:
        recs = [r for r in records if r["session"] == p["session"]]
        payload = [{"id": r["hypothesis"]["id"], "test_question": r.get("test_question", "(검정하지 않음: 무효)"),
                    "claim_type": r["hypothesis"]["claim_type"], "verdict": VERDICT_KO[r["verdict"]],
                    "result_sentence": render_result(r), "planner_statement": r["hypothesis"]["statement"]}
                   for r in recs]
        rep = make_llm(p["session"]).structured(REPORTER_SYSTEM, json.dumps(payload, ensure_ascii=False), REPORT_SCHEMA,
                                                role=f"reporter_s{p['session']}")
        by_id = {it["id"]: it for it in rep.get("items", [])}
        items = []
        for r in recs:
            it = by_id.get(r["hypothesis"]["id"], {"interpretation": "", "next_step": ""})
            chk = check_text(it["interpretation"] + " " + it["next_step"], r["verdict"])
            items.append({"id": r["hypothesis"]["id"], "test_question": r.get("test_question"),
                          "statement": r["hypothesis"]["statement"],
                          "claim_type": r["hypothesis"]["claim_type"], "verdict": r["verdict"],
                          "result_sentence": render_result(r), "interpretation": it["interpretation"],
                          "next_step": it["next_step"], "text_check": chk,
                          "missing_from_report": r["hypothesis"]["id"] not in by_id})
        overall_chk = check_text(rep.get("overall", ""), "not_confirmed")
        reports.append({"session": p["session"], "items": items, "overall": rep.get("overall", ""),
                        "overall_check": overall_chk})

    summary = summarize(plans, records, reports, unique, menu) | {"plans_sha256": plans_sha}
    (out / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, default=float))
    (out / "reports.json").write_text(json.dumps(reports, ensure_ascii=False, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
    (out / "report.md").write_text(render_markdown(reports, summary))
    return summary


def summarize(plans, records, reports, unique, menu) -> dict:
    normal = [p["session"] for p in plans if not p["negative_control"]]
    keysets = [{r["hypothesis_key"] for r in records if r["session"] == s} for s in normal]
    jacc = [len(a & b) / len(a | b) for i, a in enumerate(keysets) for b in keysets[i + 1:] if a | b]
    neg = [r for r in records if r["hypothesis"].get("group") == NEGATIVE_GROUP]
    items = [it for rep in reports for it in rep["items"]]
    return {
        "sessions": len(plans), "hypotheses": len(records), "unique_tests": len(unique),
        "tested_in_bh_family": len({r["key"] for r in records if "bh_adjusted_p" in r}),
        "verdicts": {v: sum(r["verdict"] == v for r in records) for v in VERDICTS},
        "executable_share": float(np.mean([r["verdict"] != "invalid" for r in records])),
        "planner_structural_compliance": float(np.mean([not r["issues"] for r in records])),
        "verdict_scope": "each verdict answers the code-generated test question; the planner's sentence and the "
                         "reporter's text are shown separately and their meaning is NOT verified",
        "llm_text_with_digits": sum(it["text_check"]["digits"] > 0 for it in items)
        + sum(rep["overall_check"]["digits"] > 0 for rep in reports),
        "llm_text_causal_word_flags": sum(bool(it["text_check"]["causal_words"]) for it in items),
        "items_missing_from_report": sum(it["missing_from_report"] for it in items),
        "stability_mean_jaccard": float(np.mean(jacc)) if jacc else None,
        "negative_control": {"session_run": any(p["negative_control"] for p in plans),
                             "selected_by_planner": bool(neg),
                             "verdicts": [r["verdict"] for r in neg],
                             "status": ("not run" if not any(p["negative_control"] for p in plans) else
                                        "not conducted (planner did not select it)" if not neg else
                                        "planted false claim confirmed (not caught)"
                                        if any(r["verdict"] == "confirmed" for r in neg)
                                        else "planted false claim not confirmed"),
                             "note": "the label comes from the experiment design (E8 was planted on purpose); "
                                     "the agent does not detect false evidence"},
        "arm_gate": menu.gate,
        "training_sensitivity": menu.training_sensitivity(),
    }


def render_markdown(reports: list[dict], summary: dict) -> str:
    ts = summary["training_sensitivity"]
    lines = ["# Hypothesis agent run", "",
             f"- sessions {summary['sessions']}, hypotheses {summary['hypotheses']}, unique tests {summary['unique_tests']}, "
             f"BH family {summary['tested_in_bh_family']} (q = 0.05, one family for the whole run, plans frozen "
             f"`{summary['plans_sha256'][:12]}` before execution)",
             f"- verdicts {summary['verdicts']}",
             f"- verdict scope: {summary['verdict_scope']}",
             f"- executable {summary['executable_share']:.0%}, planner structural-rule compliance "
             f"{summary['planner_structural_compliance']:.0%} (fields only, not meaning), "
             f"LLM text with digits {summary['llm_text_with_digits']}, causal-word flags {summary['llm_text_causal_word_flags']}",
             f"- stability (mean Jaccard of tests across sessions) {summary['stability_mean_jaccard']}",
             f"- negative control: {summary['negative_control']['status']} ({summary['negative_control']['note']})",
             f"- arm gate (per compared arm): {summary['arm_gate']}",
             f"- training sensitivity (full + one noise feature vs full, same training conditions, alpha {ts['alpha']}, "
             f"descriptive): {ts['different_at_alpha']}/{ts['computable']} comparisons differ (noise model better in "
             f"{ts['noise_model_better_at_alpha']}), largest |difference| "
             f"{'-' if ts['max_abs_estimate'] is None else format(ts['max_abs_estimate'], '.6f')}",
             ""]
    for rep in reports:
        lines += [f"## Session {rep['session']}", ""]
        for it in rep["items"]:
            flags = []
            if it["text_check"]["digits"]:
                flags.append("LLM 문장에 숫자 포함")
            if it["text_check"]["causal_words"]:
                flags.append(f"인과 표현 의심: {', '.join(it['text_check']['causal_words'])}")
            lines += [f"**H{it['id']}** ({it['claim_type']}) 검정 질문 (코드): {it['test_question'] or '없음 (무효)'}", "",
                      f"- 판정·결과 (코드, 위 질문에 대한 것): {it['result_sentence']}",
                      f"- 플래너 제안 문장 (LLM, 의미 검증 안 함): {it['statement']}",
                      f"- 해석 (LLM, 의미 검증 안 함): {it['interpretation']}",
                      f"- 다음 단계 (LLM, 의미 검증 안 함): {it['next_step']}"]
            if flags:
                lines.append(f"- ⚠️ {'; '.join(flags)}")
            lines.append("")
        lines += [f"종합 (LLM, 의미 검증 안 함): {rep['overall']}", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("menu")
    ap.add_argument("out")
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--negative-control", type=int, default=None, help="session number that sees the false E8")
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--from-plans", type=Path, default=None, help="execute a frozen plans_frozen.json (no planner calls)")
    ap.add_argument("--plans-only", action="store_true", help="write and hash the plans, then stop (menu not loaded)")
    a = ap.parse_args()
    audit = AuditLog(Path(a.out) / "audit.jsonl")
    if a.scripted:
        from .scripted import scripted_factory

        factory = scripted_factory(audit, a.negative_control)
    else:
        def factory(k: int):
            return ClovaClient(audit, seed=k)
    print(json.dumps(run(Path(a.menu), Path(a.out), factory, a.sessions, a.negative_control, a.from_plans,
                         a.plans_only),
                     ensure_ascii=False, indent=2, default=float))
