"""Hypothesis agent run: several planner sessions, frozen before any confirmation result, then one
deduplicated execution with a single multiple-testing family, then reports and checks.

    uv run python -m cvr.agent.session runs/agent_menu runs/agent_runs/<name> --sessions 5 --negative-control 5
    uv run python -m cvr.agent.session runs/agent_menu runs/agent_runs/<name> --scripted     # no API key

What the code decides (never the LLM):
- validity: evidence ids must exist (else the hypothesis is invalid and not run); the tool must fit the
  claim type; causal claims are never tested on observational data (routed to ab_test_design,
  verdict "needs_experiment");
- statistics: tools return "insufficient_data" when an arm is too small or non-finite; such results never
  enter the multiple-testing step; Benjamini-Hochberg (q = 0.05) over ALL unique tested hypotheses of the
  run (the family is fixed by the frozen plans);
- every number in the report: result sentences are rendered by code from the tool output.
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
    return {"type": "object", "required": ["hypotheses"], "properties": {"hypotheses": {
        "type": "array", "minItems": 3, "maxItems": 8, "items": {
            "type": "object",
            "required": ["id", "statement", "claim_type", "evidence", "tool", "segment", "expected"],
            "properties": {
                "id": {"type": "integer"}, "statement": {"type": "string"},
                "claim_type": {"type": "string", "enum": ["predictive", "associational", "causal"]},
                "evidence": {"type": "array", "items": {"type": "string"}},
                "tool": {"type": "string", "enum": ["ablation_effect", "association", "interaction", "ab_test_design"]},
                "group": {"type": "string", "enum": groups + ["none"]},
                "feature": {"type": "string", "enum": list(ASSOC_FEATURES) + ["none"]},
                "segment": {"type": "string", "enum": SEGMENTS},
                "segment_b": {"type": "string", "enum": SEGMENTS + ["none"]},
                "relative_mde": {"type": "number", "minimum": 0.01, "maximum": 0.5},
                "expected": {"type": "string", "enum": EXPECTED}}}}}}


REPORT_SCHEMA = {"type": "object", "required": ["items", "overall"], "properties": {
    "items": {"type": "array", "items": {"type": "object", "required": ["id", "interpretation", "next_step"],
                                         "properties": {"id": {"type": "integer"}, "interpretation": {"type": "string"},
                                                        "next_step": {"type": "string"}}}},
    "overall": {"type": "string"}}}

PLANNER_SYSTEM = """너는 광고 전환 분석가다. 주어진 근거 자료만 보고, 클릭한 사용자를 전환으로 이끄는 단서에 대한 검증 가능한 가설을 3~8개 제안한다.
규칙:
- 각 가설은 근거 자료의 id(E1 등)를 evidence에 인용한다. 근거에 없는 사실을 지어내지 않는다.
- claim_type: 예측에 도움이 되는지(predictive), 함께 나타나는지(associational), 바꾸면 전환이 늘어나는지(causal)를 정확히 구분한다.
- predictive는 ablation_effect 또는 interaction, associational은 association, causal은 ab_test_design 도구를 쓴다.
- 도구 인자는 주어진 목록(group, feature, segment)에서만 고른다. 쓰지 않는 인자는 "none".
- 숫자 결과를 예상해서 쓰지 않는다. 결과는 코드가 계산한다."""

REPORTER_SYSTEM = """너는 분석 보고서 작성자다. 가설별로 코드가 계산한 판정과 결과가 주어진다.
규칙:
- 숫자(아라비아 숫자)를 전혀 쓰지 않는다. 결과 수치와 판정 문장은 코드가 따로 붙인다.
- 판정을 바꾸거나 더 강하게 표현하지 않는다. '확인되지 않음', '자료 부족', 'A/B 필요'인 가설을 사실처럼 쓰지 않는다.
- associational 결과는 인과로 해석하지 않는다. 예측에 도움이 된다는 결과도 '바꾸면 전환이 늘어난다'는 뜻이 아니다.
- interpretation에는 이 결과가 무엇을 뜻하고 무엇을 뜻하지 않는지, next_step에는 다음에 할 실험이나 확인을 한 문장씩 쓴다."""


# ------------------------------------------------------------------ helpers
def hypothesis_key(h: dict) -> tuple:
    """Identity of a test: two hypotheses with the same key are the same comparison."""
    return (h["claim_type"], h["tool"], h.get("group", "none"), h.get("feature", "none"), h["segment"],
            h.get("segment_b", "none") if h["tool"] == "interaction" else "none", h["expected"])


def validate(h: dict, known_ev: set[str]) -> list[str]:
    issues = []
    bad_ev = [e for e in h.get("evidence", []) if e not in known_ev]
    if not h.get("evidence"):
        issues.append("no evidence cited")
    if bad_ev:
        issues.append(f"evidence ids not in the package: {bad_ev}")
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
    t = h["tool"]
    if h["claim_type"] == "causal":
        return menu.ab_test_design(h["segment"], float(h.get("relative_mde") or 0.05))
    if t == "ablation_effect":
        return menu.ablation_effect(h["group"], h["segment"])
    if t == "interaction":
        return menu.interaction(h["group"], h["segment"], h["segment_b"])
    return menu.association(h["feature"], h["segment"])


def bh(p: np.ndarray, q: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini-Hochberg: (significant mask, adjusted p-values)."""
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj <= q, adj


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
                f"{r['clicks_per_arm_iid']:,}){days}")
    unit = {"ablation_effect": "LogLoss 차이", "interaction": "제거 효과의 차이", "association": "전환율 차이"}[r["tool"]]
    if "effect_vs_placebo" in r:  # retrained-model comparison: judged against the placebo (noise) null
        lo, hi = r["ci95_vs_placebo"]
        return (f"{head}. {unit} {r['estimate']:+.6f} (위약 평균 {r['placebo_mean']:+.6f}, 위약 {r['placebo_replicates']}개), "
                f"위약 대비 효과 {r['effect_vs_placebo']:+.6f} [95% 근사 구간 {lo:+.6f}, {hi:+.6f}], "
                f"BH 조정 p = {rec['bh_adjusted_p']:.3f}")
    extra = (f", 전환율 {r['cvr_with']:.4f} vs {r['cvr_without']:.4f}" if r["tool"] == "association" else "")
    return (f"{head}. {unit} {r['estimate']:+.6f} [95% CI {r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}], "
            f"BH 조정 p = {rec['bh_adjusted_p']:.3f}{extra}")


IDENTIFIER = re.compile(r"[A-Za-z_]+\d+[A-Za-z_\d]*")  # ad_feat1, sparse_G1, G1, E3: names, not numbers


def check_text(text: str, verdict: str) -> dict:
    digits = re.findall(r"\d", IDENTIFIER.sub("", text))
    hits = [w for w in CAUSAL_WORDS if w in text]
    if verdict == "needs_experiment":  # describing a future experiment may use causal words; "proven" may not
        hits = [w for w in hits if w in ("입증", "증명")]
    return {"digits": len(digits), "causal_words": hits}


# ------------------------------------------------------------------ run
def run(menu_dir: Path, out: Path, make_llm, sessions: int, negative_control_session: int | None,
        from_plans: Path | None = None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    menu = Menu(menu_dir)
    # 1) all plans first; frozen (hashed) before any tool sees confirmation data.
    #    With from_plans, an earlier frozen set is reused as is (re-analysis without new planner calls).
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

    # 2) validate, deduplicate, execute each unique test once, one BH family for the whole run
    records, unique = [], {}
    for p in plans:
        known = {e["id"] for e in EVIDENCE} | ({FALSE_EVIDENCE["id"]} if p["negative_control"] else set())
        for h in p["plan"]["hypotheses"]:
            issues = validate(h, known)
            rec = {"session": p["session"], "hypothesis": h, "issues": issues, "key": list(hypothesis_key(h))}
            if issues:
                rec["verdict"] = "invalid"
            else:
                key = hypothesis_key(h)
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
    tested_keys = sorted({tuple(r["key"]) for r in records if r["verdict"] is None})
    if tested_keys:
        sig, adj = bh(np.array([unique[k]["p_value"] for k in tested_keys]))
        bh_by_key = {k: (bool(s), float(a)) for k, s, a in zip(tested_keys, sig, adj)}
        for r in records:
            if r["verdict"] is None:
                s, a = bh_by_key[tuple(r["key"])]
                r["bh_significant"], r["bh_adjusted_p"] = s, a
                want_negative = r["hypothesis"]["expected"] in ("group_helps", "group_helps_more_in_segment",
                                                                "lower_cvr_with_feature")
                eff = r["result"].get("effect_vs_placebo", r["result"]["estimate"])
                if not s:
                    r["verdict"] = "not_confirmed"
                else:
                    r["verdict"] = "confirmed" if (eff < 0) == want_negative else "opposite_direction"

    # 3) reports: LLM interpretation without digits + code-rendered results; checks
    reports = []
    for p in plans:
        recs = [r for r in records if r["session"] == p["session"]]
        payload = [{"id": r["hypothesis"]["id"], "statement": r["hypothesis"]["statement"],
                    "claim_type": r["hypothesis"]["claim_type"], "verdict": VERDICT_KO[r["verdict"]],
                    "result_sentence": render_result(r)} for r in recs]
        rep = make_llm(p["session"]).structured(REPORTER_SYSTEM, json.dumps(payload, ensure_ascii=False), REPORT_SCHEMA,
                                                role=f"reporter_s{p['session']}")
        by_id = {it["id"]: it for it in rep.get("items", [])}
        items = []
        for r in recs:
            it = by_id.get(r["hypothesis"]["id"], {"interpretation": "", "next_step": ""})
            chk = check_text(it["interpretation"] + " " + it["next_step"], r["verdict"])
            items.append({"id": r["hypothesis"]["id"], "statement": r["hypothesis"]["statement"],
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
    keysets = [{tuple(r["key"]) for r in records if r["session"] == s} for s in normal]
    jacc = [len(a & b) / len(a | b) for i, a in enumerate(keysets) for b in keysets[i + 1:] if a | b]
    neg = [r for r in records if r["hypothesis"].get("group") == NEGATIVE_GROUP]
    items = [it for rep in reports for it in rep["items"]]
    return {
        "sessions": len(plans), "hypotheses": len(records), "unique_tests": len(unique),
        "tested_in_bh_family": len({tuple(r["key"]) for r in records if "bh_adjusted_p" in r}),
        "verdicts": {v: sum(r["verdict"] == v for r in records) for v in VERDICTS},
        "executable_share": float(np.mean([r["verdict"] != "invalid" for r in records])),
        "planner_rule_compliance": float(np.mean([not r["issues"] for r in records])),
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
                                        "wrongly confirmed" if any(r["verdict"] == "confirmed" for r in neg) else
                                        "not confirmed")},
        "placebo_calibration": menu.placebo_calibration(),
    }


def render_markdown(reports: list[dict], summary: dict) -> str:
    pc = summary["placebo_calibration"]
    lines = ["# Hypothesis agent run", "",
             f"- sessions {summary['sessions']}, hypotheses {summary['hypotheses']}, unique tests {summary['unique_tests']}, "
             f"BH family {summary['tested_in_bh_family']} (q = 0.05, one family for the whole run, plans frozen "
             f"`{summary['plans_sha256'][:12]}` before execution)",
             f"- verdicts {summary['verdicts']}",
             f"- executable {summary['executable_share']:.0%}, planner rule compliance {summary['planner_rule_compliance']:.0%}, "
             f"LLM text with digits {summary['llm_text_with_digits']}, causal-word flags {summary['llm_text_causal_word_flags']}",
             f"- stability (mean Jaccard of tests across sessions) {summary['stability_mean_jaccard']}",
             f"- negative control: {summary['negative_control']['status']}",
             f"- placebo calibration (pure-noise features, alpha {pc['alpha']}): evaluation-only test "
             f"{pc['rejections']}/{pc['computable']} rejections (noise looked helpful in {pc['rejections_noise_helps']}); "
             f"training-variance adjusted test (leave-one-out null) {pc['rejections_adjusted_leave_one_out']}/{pc['computable']}",
             ""]
    for rep in reports:
        lines += [f"## Session {rep['session']}", ""]
        for it in rep["items"]:
            flags = []
            if it["text_check"]["digits"]:
                flags.append("LLM 문장에 숫자 포함")
            if it["text_check"]["causal_words"]:
                flags.append(f"인과 표현 의심: {', '.join(it['text_check']['causal_words'])}")
            lines += [f"**H{it['id']}** ({it['claim_type']}) {it['statement']}", "",
                      f"- 결과 (코드): {it['result_sentence']}",
                      f"- 해석 (LLM): {it['interpretation']}", f"- 다음 단계 (LLM): {it['next_step']}"]
            if flags:
                lines.append(f"- ⚠️ {'; '.join(flags)}")
            lines.append("")
        lines += [f"종합 (LLM): {rep['overall']}", ""]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("menu")
    ap.add_argument("out")
    ap.add_argument("--sessions", type=int, default=5)
    ap.add_argument("--negative-control", type=int, default=None, help="session number that sees the false E8")
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--from-plans", type=Path, default=None, help="reuse a plans_frozen.json (re-analysis)")
    a = ap.parse_args()
    audit = AuditLog(Path(a.out) / "audit.jsonl")
    if a.scripted:
        from .scripted import scripted_factory

        factory = scripted_factory(audit)
    else:
        def factory(k: int):
            return ClovaClient(audit, seed=k)
    print(json.dumps(run(Path(a.menu), Path(a.out), factory, a.sessions, a.negative_control, a.from_plans),
                     ensure_ascii=False, indent=2, default=float))
