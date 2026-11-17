"""One hypothesis session: plan (LLM) -> execute (code) -> multiplicity correction (code) -> report (LLM)
-> verify (code). Verdicts are decided by code, never by the LLM.

    uv run python -m cvr.agent.session runs/agent_menu runs/agent_sessions/<name> --seed 1
    uv run python -m cvr.agent.session runs/agent_menu runs/agent_sessions/<name> --scripted   # no API key

Rules enforced in code:
- a causal hypothesis is never "tested" on observational data: it is routed to ab_test_design and its
  verdict is "needs_experiment";
- predictive hypotheses must use ablation_effect / interaction, associational ones association;
- evidence ids must exist in the evidence package;
- Benjamini-Hochberg (q = 0.05) over all tested hypotheses of the session;
- the report's verdicts must equal the code verdicts, and every number in the report must appear in the
  tool results or the evidence package (rounding-tolerant), otherwise it is flagged as ungrounded.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np

from .evidence import EVIDENCE, FALSE_EVIDENCE, evidence_text
from .experiments import ASSOC_FEATURES, FEATURE_GROUPS, NEGATIVE_GROUP, SEGMENTS, Menu
from .llm import AuditLog, ClovaClient, ScriptedLLM

VERDICTS = ["confirmed", "opposite_direction", "not_confirmed", "needs_experiment", "invalid"]
EXPECTED = ["group_helps", "group_helps_more_in_segment", "higher_cvr_with_feature", "lower_cvr_with_feature",
            "needs_experiment"]

def plan_schema(groups: list[str]) -> dict:
    s = json.loads(json.dumps(PLAN_SCHEMA))
    s["properties"]["hypotheses"]["items"]["properties"]["group"]["enum"] = groups + ["none"]
    return s


PLAN_SCHEMA = {
    "type": "object",
    "properties": {"hypotheses": {"type": "array", "minItems": 3, "maxItems": 8, "items": {
        "type": "object",
        "properties": {
            "id": {"type": "integer"},
            "statement": {"type": "string"},
            "claim_type": {"type": "string", "enum": ["predictive", "associational", "causal"]},
            "evidence": {"type": "array", "items": {"type": "string"}},
            "tool": {"type": "string", "enum": ["ablation_effect", "association", "interaction", "ab_test_design"]},
            "group": {"type": "string", "enum": list(FEATURE_GROUPS) + ["none"]},
            "feature": {"type": "string", "enum": list(ASSOC_FEATURES) + ["none"]},
            "segment": {"type": "string", "enum": SEGMENTS},
            "segment_b": {"type": "string", "enum": SEGMENTS + ["none"]},
            "relative_mde": {"type": "number", "minimum": 0.01, "maximum": 0.5},
            "expected": {"type": "string", "enum": EXPECTED},
        },
        "required": ["id", "statement", "claim_type", "evidence", "tool", "segment", "expected"]}}},
    "required": ["hypotheses"],
}
REPORT_SCHEMA = {
    "type": "object",
    "properties": {
        "items": {"type": "array", "items": {"type": "object", "properties": {
            "id": {"type": "integer"}, "verdict": {"type": "string", "enum": VERDICTS},
            "summary": {"type": "string"}, "next_step": {"type": "string"}},
            "required": ["id", "verdict", "summary", "next_step"]}},
        "overall": {"type": "string"}},
    "required": ["items", "overall"],
}

PLANNER_SYSTEM = """너는 광고 전환 분석가다. 주어진 근거 자료만 보고, 클릭한 사용자를 전환으로 이끄는 단서에 대한 검증 가능한 가설을 3~8개 제안한다.
규칙:
- 각 가설은 근거 자료의 id(E1 등)를 evidence에 인용한다. 근거에 없는 사실을 지어내지 않는다.
- claim_type: 예측에 도움이 되는지(predictive), 함께 나타나는지(associational), 바꾸면 전환이 늘어나는지(causal)를 정확히 구분한다.
- predictive는 ablation_effect 또는 interaction, associational은 association, causal은 ab_test_design 도구를 쓴다.
- 도구 인자는 주어진 목록(group, feature, segment)에서만 고른다. 쓰지 않는 인자는 "none".
- 숫자 결과를 예상해서 쓰지 않는다. 결과는 코드가 계산한다."""

REPORTER_SYSTEM = """너는 분석 보고서 작성자다. 가설별로 코드가 계산한 결과와 판정(verdict)이 주어진다.
규칙:
- verdict는 주어진 값을 그대로 쓴다. 바꾸지 않는다.
- 숫자는 주어진 결과에 있는 값만, 소수점 4자리 이하로 반올림해 쓴다. 새 숫자를 만들지 않는다.
- associational 결과는 인과로 해석하지 않는다. needs_experiment는 'A/B 테스트가 필요하다'고 쓰고 표본 수를 함께 적는다.
- next_step에는 다음에 할 실험이나 확인을 한 문장으로 쓴다."""

ALLOWED_TOOLS = {"predictive": {"ablation_effect", "interaction"}, "associational": {"association"},
                 "causal": {"ab_test_design"}}


def execute(plan: dict, menu: Menu, negative_control: bool = False) -> list[dict]:
    known_ev = {e["id"] for e in EVIDENCE} | ({FALSE_EVIDENCE["id"]} if negative_control else set())
    out = []
    for h in plan["hypotheses"]:
        rec = {"hypothesis": h, "issues": []}
        bad_ev = [e for e in h.get("evidence", []) if e not in known_ev]
        if bad_ev or not h.get("evidence"):
            rec["issues"].append(f"evidence ids not in the package: {bad_ev or 'none given'}")
        ctype, tool = h["claim_type"], h["tool"]
        if ctype == "causal":
            if tool != "ab_test_design":
                rec["issues"].append("causal claim rerouted to ab_test_design (observational data cannot confirm it)")
            rec["result"] = menu.ab_test_design(h["segment"], float(h.get("relative_mde") or 0.05))
            rec["verdict"] = "needs_experiment"
        elif tool not in ALLOWED_TOOLS[ctype]:
            rec["issues"].append(f"tool {tool} not allowed for a {ctype} claim")
            rec["verdict"] = "invalid"
        else:
            try:
                if tool == "ablation_effect":
                    rec["result"] = menu.ablation_effect(h["group"], h["segment"])
                elif tool == "interaction":
                    rec["result"] = menu.interaction(h["group"], h["segment"], h["segment_b"])
                else:
                    rec["result"] = menu.association(h["feature"], h["segment"])
                rec["verdict"] = None  # decided after the multiplicity correction
            except (ValueError, KeyError) as e:
                rec["issues"].append(f"tool call failed: {e}")
                rec["verdict"] = "invalid"
        out.append(rec)
    # Benjamini-Hochberg over the tested hypotheses of this session
    tested = [r for r in out if r["verdict"] is None]
    if tested:
        p = np.array([r["result"]["p_value"] for r in tested])
        order = np.argsort(p)
        m = len(p)
        thresh = 0.05 * (np.arange(1, m + 1) / m)
        passed = p[order] <= thresh
        k = np.max(np.nonzero(passed)[0]) + 1 if passed.any() else 0
        significant = set(order[:k].tolist())
        for i, r in enumerate(tested):
            r["bh_significant"] = i in significant
            est, exp = r["result"]["estimate"], r["hypothesis"]["expected"]
            want_negative = exp in ("group_helps", "group_helps_more_in_segment", "lower_cvr_with_feature")
            if exp == "needs_experiment":
                r["verdict"] = "invalid"
                r["issues"].append("expected=needs_experiment on a non-causal claim")
            elif not r["bh_significant"]:
                r["verdict"] = "not_confirmed"
            else:
                r["verdict"] = "confirmed" if (est < 0) == want_negative else "opposite_direction"
    return out


def _numbers(text: str) -> list[float]:
    return [float(x.replace(",", "")) for x in re.findall(r"-?\d[\d,]*\.?\d*", text)]


def _allowed_numbers(records: list[dict]) -> list[float]:
    vals: list[float] = []

    def walk(v):
        if isinstance(v, bool):
            return
        if isinstance(v, (int, float)):
            vals.extend([float(v), float(v) * 100])
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)
    for r in records:
        walk(r.get("result", {}))
        walk({"id": r["hypothesis"]["id"]})
    for e in EVIDENCE:
        vals.extend(e.get("numbers", []))
    return vals


def grounded(x: float, allowed: list[float]) -> bool:
    """A reported number is grounded if some allowed value rounds to it (up to 6 decimals)."""
    if float(x).is_integer() and abs(x) <= 20:  # identifiers and small counts (hypothesis ids, E3, G1, ...)
        return True
    for v in allowed:
        for d in range(0, 7):
            if round(v, d) == x or round(abs(v), d) == abs(x):
                return True
    return False


def verify(records: list[dict], report: dict) -> dict:
    code = {r["hypothesis"]["id"]: r["verdict"] for r in records}
    allowed = _allowed_numbers(records) + [95.0, 0.05, 0.8, 50.0]
    mismatched, ungrounded = [], []
    for it in report["items"]:
        if code.get(it["id"]) != it["verdict"]:
            mismatched.append({"id": it["id"], "report": it["verdict"], "code": code.get(it["id"])})
        for x in _numbers(it["summary"] + " " + it["next_step"]):
            if not grounded(x, allowed):
                ungrounded.append({"id": it["id"], "number": x})
    for x in _numbers(report["overall"]):
        if not grounded(x, allowed):
            ungrounded.append({"id": "overall", "number": x})
    n_numbers = sum(len(_numbers(it["summary"] + " " + it["next_step"])) for it in report["items"]) + len(
        _numbers(report["overall"]))
    missing = sorted(set(code) - {it["id"] for it in report["items"]})
    return {"verdict_mismatches": mismatched, "ungrounded_numbers": ungrounded, "numbers_in_report": n_numbers,
            "hypotheses_missing_from_report": missing,
            "passed": not mismatched and not ungrounded and not missing}


def run_session(menu_dir: Path, out: Path, llm, negative_control: bool = False) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    menu = Menu(menu_dir)
    groups = list(FEATURE_GROUPS) + ([NEGATIVE_GROUP] if negative_control else [])
    lists = {"group": groups, "feature": list(ASSOC_FEATURES), "segment": SEGMENTS}
    plan = llm.structured(PLANNER_SYSTEM, f"근거 자료:\n{evidence_text(negative_control)}\n\n사용 가능한 인자:\n"
                          f"{json.dumps(lists, ensure_ascii=False)}", plan_schema(groups), role="planner")
    records = execute(plan, menu, negative_control)
    payload = [{"id": r["hypothesis"]["id"], "statement": r["hypothesis"]["statement"],
                "claim_type": r["hypothesis"]["claim_type"], "verdict": r["verdict"],
                "result": {k: v for k, v in r.get("result", {}).items() if k != "meaning"},
                "result_meaning": r.get("result", {}).get("meaning"), "issues": r["issues"]} for r in records]
    report = llm.structured(REPORTER_SYSTEM, json.dumps(payload, ensure_ascii=False, default=float),
                            REPORT_SCHEMA, role="reporter")
    check = verify(records, report)
    summary = {
        "hypotheses": len(records),
        "by_claim_type": {t: sum(r["hypothesis"]["claim_type"] == t for r in records)
                          for t in ("predictive", "associational", "causal")},
        "verdicts": {v: sum(r["verdict"] == v for r in records) for v in VERDICTS},
        "executable_share": float(np.mean([r["verdict"] != "invalid" for r in records])),
        "hypotheses_with_issues": sum(bool(r["issues"]) for r in records),
        "verification": check,
        "negative_control": ({"hypotheses_on_control": sum(r["hypothesis"].get("group") == NEGATIVE_GROUP for r in records),
                              "wrongly_confirmed": sum(r["hypothesis"].get("group") == NEGATIVE_GROUP
                                                       and r["verdict"] == "confirmed" for r in records)}
                             if negative_control else None),
    }
    (out / "plan.json").write_text(json.dumps(plan, ensure_ascii=False, indent=2))
    (out / "records.json").write_text(json.dumps(records, ensure_ascii=False, indent=2, default=float))
    (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("menu")
    ap.add_argument("out")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--negative-control", action="store_true")
    a = ap.parse_args()
    audit = AuditLog(Path(a.out) / "audit.jsonl")
    if a.scripted:
        from .scripted import scripted_llm

        llm = scripted_llm(audit)
    else:
        llm = ClovaClient(audit, seed=a.seed)
    print(json.dumps(run_session(Path(a.menu), Path(a.out), llm, a.negative_control), ensure_ascii=False, indent=2))
