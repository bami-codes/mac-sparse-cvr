"""Agent v2: does a planner that reads yesterday's verdicts pick better tests for today? (docs/DESIGN.md §19)

    uv run python -m cvr.agent.loop evidence        # evidence from the menu's training days only
    uv run python -m cvr.agent.loop exhaustive      # every menu test on each round's day (the scoring sets)
    uv run python -m cvr.agent.loop run             # three HCX strategies (add --scripted to run without the API)
    uv run python -m cvr.agent.loop score           # scores, tokens and latency -> runs/agent_v2/summary.json

Each round is scored on a later day than the last: round 1 on train-07, round 2 on train-10, round 3 on train-11.
Every strategy fills exactly five slots per round. A slot scores when its test passes BH over the whole menu on
that day, in the predicted direction. The planner never sees those sets; what it sees between rounds is computed
from the tests it asked for, judged in order with LOND.
"""

import argparse
import itertools
import json
import math
import sys
import time
import tomllib
from pathlib import Path

import numpy as np
import polars as pl

from .evidence import FALSE_EVIDENCE
from .experiments import (
    ASSOC_FEATURES,
    FEATURE_GROUPS,
    NEGATIVE_GROUP,
    SEGMENTS,
    Menu,
    empty_arm_by_definition,
)
from .llm import AuditLog, ClovaClient, LLMError, ScriptedLLM
from .session import (
    PLANNER_SYSTEM,
    VERDICT_KO,
    bh,
    control_hypothesis,
    normalize,
    plan_schema,
    test_question,
    tool_call,
    validate,
)

OUT = Path("runs/agent_v2")
MENUS = {1: Path("runs/agent_menu_v2"), 2: Path("runs/agent_menu_v2_r2"), 3: Path("runs/agent_menu_v2_r3")}
DAYS = {1: "train-07", 2: "train-10", 3: "train-11"}
CONFIG = Path("configs/agent_menu_v2.toml")
SLOTS, SESSIONS, ROUNDS, Q, ALPHA = 5, 5, 3, 0.05, 0.05
ARMS = ("H-mem", "H-fb", "H-fb-think")
NEGATIVE_SESSION = 5
EXPECTED_SIGN = {"group_helps": -1, "group_helps_more_in_segment": -1, "higher_cvr_with_feature": 1,
                 "lower_cvr_with_feature": -1}

# ------------------------------------------------------------------------------------------------------ evidence


def evidence(config: Path = CONFIG) -> list[dict]:
    """What an analyst could know before round 1: simple statistics of the menu's own training days (train-05, 06).

    No model results: the menu models were fitted on these rows, so any score measured here is in-sample."""
    from .. import groups as G
    from .. import vocab as V
    from ..features import featurize, read_raw
    from ..schema import VOCAB_COLS
    cfg = tomllib.loads(config.read_text())
    root, files = Path(cfg["data"]["root"]), cfg["data"]["discovery"]
    counts = None
    for f in files:
        counts = V.count_values([read_raw(root / f, columns=VOCAB_COLS).select(VOCAB_COLS)], counts=counts)
    vocab = V.build_vocab(counts)
    days = {f: featurize(read_raw(root / f), vocab) for f in files}
    disc = pl.concat(list(days.values()))
    gc = cfg["groups"]
    disc = pl.concat([disc, G.classify(disc, G.train_stats(disc), gc["n_min"], gc["c_min"], gc["c_dense"])],
                     how="horizontal")

    def rate(mask) -> tuple[float, float]:
        sub = disc.filter(mask)
        return 100 * sub.height / disc.height, 100 * float(sub["y"].mean())

    hist = {}
    for feat, col in (("같은 상품", "match_ad_feat0"), ("같은 ad_feat1 값", "match_ad_feat1"), ("같은 ad_feat2 값", "match_ad_feat2")):
        share, with_ = rate(pl.col(col) > 0)
        _, without = rate(pl.col(col) == 0)
        hist[feat] = (share, with_, without)
    cuts = [float(disc["mm_feat0_max"].quantile(x)) for x in (0.2, 0.4, 0.6, 0.8)]
    quint = (disc.with_columns(pl.col("mm_feat0_max").cut(cuts, labels=[str(i) for i in range(5)]).alias("qq"))
             .group_by("qq").agg(pl.col("y").mean()).sort("qq")["y"].to_list())
    empty_share, empty_cvr = rate(pl.col("hist_empty") == 1)
    _, has_cvr = rate(pl.col("hist_empty") == 0)
    g1_share, g1_cvr = rate(pl.col("group_class") == "G1")
    dense_share, dense_cvr = rate(pl.col("group_class") == "dense")
    daily = {f.replace(".parquet", ""): 100 * float(d["y"].mean()) for f, d in days.items()}
    r = lambda x: round(x, 2)
    items = [
        {"id": "E1", "title": "데이터 사전 (타오바오 광고 클릭, MAC 공개 벤치마크)",
         "text": "한 행은 광고 클릭 1건, 정답은 마지막 클릭 귀속 전환 여부다. 특성 그룹: history_match(과거 구매 시퀀스에 타깃과 "
                 "같은 상품·ad_feat1·ad_feat2 값이 나온 횟수, 구매 이력 없음 여부), visual_similarity(과거 구매 상품과 타깃 상품의 "
                 "시각 유사도 최댓값·평균), user_profile(익명 사용자 특성 6개), context(익명 맥락 특성 3개), item_attributes(익명 "
                 f"상품 특성 3개). ad_feat1은 카테고리로 추정되는 익명 특성이다. sparse_G1은 이 기간에 클릭이 {gc['n_min']}건 이상인데 "
                 f"전환이 {gc['c_min']}건 미만인 ad_feat1 그룹이고, dense는 전환이 {gc['c_dense']}건 이상인 그룹이다.", "numbers": []},
        {"id": "E2", "title": "구매 이력 일치와 전환율 (train-05·06, 단순 연관)",
         "text": " ".join(f"과거 구매에 {k}이 있는 클릭 {r(s)}%, 그때 전환율 {r(a)}% vs 없을 때 {r(b)}%."
                          for k, (s, a, b) in hist.items()) + " 단순 연관이며 인과가 아니다.",
         "numbers": [r(x) for v in hist.values() for x in v]},
        {"id": "E3", "title": "시각 유사도와 전환율 (train-05·06, 단순 연관)",
         "text": "과거 구매 상품과의 시각 유사도 최댓값 5분위별 전환율: " + ", ".join(f"{r(100 * x)}%" for x in quint)
                 + " (낮은 분위부터). 단순 연관이다.", "numbers": [r(100 * x) for x in quint]},
        {"id": "E4", "title": "구매 이력 없음 (train-05·06)",
         "text": f"구매 이력이 비어 있는 클릭은 {r(empty_share)}%이고 전환율은 {r(empty_cvr)}%다(이력 있음 {r(has_cvr)}%). "
                 "이들에게는 history_match와 visual_similarity 정보가 없다.",
         "numbers": [r(empty_share), r(empty_cvr), r(has_cvr)]},
        {"id": "E5", "title": "희소 그룹과 밀집 그룹 (train-05·06)",
         "text": f"sparse_G1 클릭은 전체의 {r(g1_share)}%, 전환율 {r(g1_cvr)}%. dense 클릭은 {r(dense_share)}%, 전환율 "
                 f"{r(dense_cvr)}%.", "numbers": [r(g1_share), r(g1_cvr), r(dense_share), r(dense_cvr)]},
        {"id": "E6", "title": "날짜별 전환율",
         "text": "양성 비율 " + ", ".join(f"{k} {r(v)}%" for k, v in daily.items()) + ".", "numbers": [r(v) for v in daily.values()]},
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    rec = {"items": items, "sources": files, "rows": disc.height,
           "association_effects": {k: r(a - b) for k, (_, a, b) in hist.items()},
           "visual_top_minus_bottom": r(100 * (quint[-1] - quint[0])), "empty_minus_has": r(empty_cvr - has_cvr)}
    (OUT / "evidence.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2))
    return items


def evidence_text(items: list[dict], negative_control: bool) -> str:
    return "\n".join(f"[{e['id']}] {e['title']}: {e['text']}" for e in items + ([FALSE_EVIDENCE] if negative_control else []))


# ----------------------------------------------------------------------------------------- the menu and scoring


def hyp(tool: str, segment: str, group: str = "none", feature: str = "none", segment_b: str = "none",
        expected: str | None = None, evidence_ids: tuple[str, ...] = ("E1",), statement: str = "") -> dict:
    claim = "associational" if tool == "association" else "predictive"
    exp = expected or {"ablation_effect": "group_helps", "interaction": "group_helps_more_in_segment"}.get(tool)
    return {"statement": statement, "claim_type": claim, "evidence": list(evidence_ids), "tool": tool,
            "segment": segment, "group": group, "feature": feature, "segment_b": segment_b, "expected": exp}


def menu_tests() -> list[dict]:
    """The 263 tests of the menu: interactions as unordered segment pairs (the reverse order is the same test)."""
    out = [hyp("ablation_effect", s, group=g) for g in FEATURE_GROUPS for s in SEGMENTS]
    out += [hyp("association", s, feature=f, expected="higher_cvr_with_feature") for f in ASSOC_FEATURES for s in SEGMENTS
            if empty_arm_by_definition(f, s) is None]
    out += [hyp("interaction", a, group=g, segment_b=b) for g in FEATURE_GROUPS for a, b in itertools.combinations(SEGMENTS, 2)]
    return out


def canon(h: dict) -> tuple[str, int]:
    """Test identity independent of the order of an interaction's two segments, and the sign flip of that order."""
    name, a = tool_call(h)
    if name == "interaction" and SEGMENTS.index(a["segment"]) > SEGMENTS.index(a["segment_b"]):
        return json.dumps([name, {"group": a["group"], "segment": a["segment_b"], "segment_b": a["segment"]}], sort_keys=True), -1
    return json.dumps([name, a], sort_keys=True), 1


def exhaustive(day: int) -> dict:
    """Every menu test on one day. Saved only; no strategy reads it."""
    menu = Menu(MENUS[day])
    res = {}
    t0 = time.time()
    for h in menu_tests():
        name, a = tool_call(h)
        r = menu.ablation_effect(**a, sensitivity=False) if name == "ablation_effect" else getattr(menu, name)(**a)
        res[canon(h)[0]] = {k: r.get(k) for k in ("status", "estimate", "ci95", "p_wald", "p_value", "reason")}
    OUT.mkdir(parents=True, exist_ok=True)
    out = {"day": DAYS[day], "menu": str(MENUS[day]), "tests": len(res), "seconds": round(time.time() - t0, 1), "results": res}
    (OUT / f"exhaustive_r{day}.json").write_text(json.dumps(out, indent=1))
    return out


def passing_set(results: dict, q: float = Q) -> dict[str, int]:
    """Tests that pass BH(q) over every menu test with a result, and the sign of each estimate."""
    keys = [k for k, r in results.items() if r["status"] == "ok"]
    if not keys:
        return {}
    sig, _ = bh(np.array([results[k]["p_wald"] for k in keys]), q)
    return {k: int(np.sign(results[k]["estimate"])) for k, s in zip(keys, sig) if s}


def is_hit(h: dict, passing: dict[str, int]) -> bool:
    key, flip = canon(h)
    return key in passing and passing[key] * flip == EXPECTED_SIGN.get(h["expected"], 0)


class Lond:
    """LOND online FDR (Javanmard & Montanari 2018): the i-th test rejects when p <= beta_i (discoveries so far + 1),
    with beta_i = alpha * 6 / (pi^2 i^2), so the betas sum to alpha."""

    def __init__(self, alpha: float = ALPHA):
        self.alpha, self.i, self.d = alpha, 0, 0

    def test(self, p: float) -> bool:
        self.i += 1
        reject = p <= self.alpha * 6 / (math.pi ** 2 * self.i ** 2) * (self.d + 1)
        self.d += int(reject)
        return reject


# ------------------------------------------------------------------------------------------- one round of a session


def run_canonical(key: str, day: int, cache: dict) -> dict:
    """One test on one day, always in canonical form, computed once and shared by every strategy."""
    if (day, key) not in cache:
        if ("menu", day) not in cache:
            cache[("menu", day)] = Menu(MENUS[day])
        menu = cache[("menu", day)]
        name, a = json.loads(key)
        r = menu.ablation_effect(**a, sensitivity=False) if name == "ablation_effect" else getattr(menu, name)(**a)
        cache[(day, key)] = {k: r.get(k) for k in ("status", "estimate", "ci95", "p_wald", "reason")}
    return cache[(day, key)]


def oriented(r: dict, flip: int) -> dict:
    """A canonical result as seen from the hypothesis' own segment order."""
    if flip == 1 or r["status"] != "ok":
        return dict(r)
    return r | {"estimate": -r["estimate"], "ci95": [-r["ci95"][1], -r["ci95"][0]]}


def execute(hyps: list[dict], day: int, cache: dict, tested: set[str], lond: Lond, known: set[str]) -> list[dict]:
    """Validate, drop repeats, run each new test on that day, judge it with the session's LOND sequence."""
    rows = []
    for raw in hyps:
        h = normalize(raw)
        rec = {"hypothesis": h, "issues": validate(h, known), "day": DAYS[day]}
        if not rec["issues"] and h["claim_type"] == "causal":
            rec["issues"] = ["causal claims are not scored in this study"]
        if rec["issues"]:
            rec["verdict"] = "invalid"
            rows.append(rec)
            continue
        key, flip = canon(h)
        rec["key"], rec["test_question"] = key, test_question(h)
        if key in tested:
            rec["verdict"] = "repeat"
            rows.append(rec)
            continue
        tested.add(key)
        r = rec["result"] = oriented(run_canonical(key, day, cache), flip)
        if r["status"] != "ok":
            rec["verdict"] = "insufficient_data"
        else:
            reject = lond.test(r["p_wald"])
            right = np.sign(r["estimate"]) == EXPECTED_SIGN[h["expected"]]
            rec["verdict"] = ("confirmed" if right else "opposite_direction") if reject else "not_confirmed"
        rows.append(rec)
    return rows


def effect_text(r: dict) -> str:
    res = r.get("result") or {}
    if res.get("status") != "ok":
        return ""
    e, (lo, hi) = res["estimate"], res["ci95"]
    if r["hypothesis"]["tool"] == "association":
        return f" 전환율 차이 {100 * e:+.2f}%p [95% {100 * lo:+.2f}, {100 * hi:+.2f}]"
    return f" LogLoss 차이 {e:+.1e} [95% {lo:+.1e}, {hi:+.1e}] (음수 = 그룹을 넣은 모델이 나음)"


def history_ids(rounds: list[list[dict]]) -> set[str]:
    return {f"R{k}-{j}" for k, rows in enumerate(rounds, start=1) for j in range(1, len(rows) + 1)}


def history_text(rounds: list[list[dict]], with_results: bool, ids: bool = False) -> str:
    """What the planner sees about earlier rounds, written by code. With ids (v2.1) every line can be cited."""
    lines = []
    for k, rows in enumerate(rounds, start=1):
        for j, r in enumerate(rows, start=1):
            tag = f"[R{k}-{j}]" if ids else f"R{k}"
            if r["verdict"] == "invalid":
                if with_results:
                    lines.append(f"- {tag} 무효: {'; '.join(r['issues'])}")
                continue
            if r["verdict"] == "repeat" or not with_results:
                lines.append(f"- {tag} {r['test_question']}")
                continue
            effect = effect_text(r)
            lines.append(f"- {tag} [{VERDICT_KO.get(r['verdict'], r['verdict'])}] {r['test_question']}{effect}")
    head = ("이전 라운드 결과 (코드가 그날 자료로 계산, 판정은 순서대로 온라인 FDR 적용):" if with_results
            else "이전 라운드에 이미 검정한 질문 (결과는 알려 주지 않음):")
    return head + "\n" + "\n".join(lines) + "\n같은 질문을 다시 제안하면 그 자리는 점수가 없다."


_AB = "- ab_test_design (claim_type=causal): 무언가를 바꾸면 전환이 늘어나는지는 관측 데이터로 확인할 수 없으므로, 필요한 실험 규모만 계산한다.\n"
assert _AB in PLANNER_SYSTEM and "검증 방법(test)은 네 가지" in PLANNER_SYSTEM
PLANNER_V2 = (PLANNER_SYSTEM.replace("3~8개", "정확히 5개").replace(_AB, "")
              .replace("검증 방법(test)은 네 가지", "검증 방법(test)은 세 가지") + """
- 인과 주장(무언가를 바꾸면 전환이 늘어난다)은 관측 자료로 확인할 수 없으므로 이번에는 제안하지 않는다.
- 이번 라운드의 가설은 다음 날 자료로 검정한다. 이미 검정한 질문을 다시 제안하지 않는다.""")
PLANNER_V21 = PLANNER_V2 + """
- 이전 라운드 항목을 근거로 쓸 때는 evidence에 그 항목의 id(예: "R1-2")를 쓴다."""
REPAIR = """방금 낸 계획 중 일부 자리는 코드 검사에서 실행할 수 없다고 판정됐다. 사유를 읽고, 그 자리만 새 가설로 바꾼다.
새 가설도 같은 규칙을 따른다. 특히 association에서 feature와 segment가 같은 조건을 가리키면 한쪽 집단이 정의상 비므로
쓸 수 없다. 이미 검정한 질문과 같은 라운드 안의 다른 가설도 다시 내지 않는다."""
THINK_EXTRA = """
이번 답은 JSON이 아니라 글로 쓴다. 가설 5개마다 statement, claim_type, test, group 또는 feature, segment, segment_b,
direction, evidence를 빠짐없이 적는다."""
TRANSCRIBE = """너는 받은 계획 글을 주어진 JSON 스키마로 옮긴다. 가설의 내용, 개수, 순서를 바꾸지 않는다.
글에 없는 값은 지어내지 않는다. 글이 가설을 5개보다 적게 담고 있으면 있는 것만 옮긴다."""


def slot_problems(hyps: list[dict], tested: set[str], known: set[str]) -> dict[int, str]:
    """Slots that would score nothing, found before anything runs: invalid, already tested, or a duplicate."""
    out, seen = {}, set()
    for i, raw in enumerate(hyps):
        h = normalize(raw)
        issues = validate(h, known) + (["인과 주장은 이번에 점수가 없다"] if h["claim_type"] == "causal" else [])
        if issues:
            out[i] = "; ".join(issues)
            continue
        key = canon(h)[0]
        if key in tested:
            out[i] = f"이미 검정한 질문이다: {test_question(h)}"
        elif key in seen:
            out[i] = f"같은 라운드 안의 다른 가설과 같은 질문이다: {test_question(h)}"
        seen.add(key)
    return out


def repair(llm, user: str, hyps: list[dict], problems: dict[int, str], groups: list[str], role: str) -> list[dict]:
    """One validator round trip: return the code's reasons, take replacements for those slots only."""
    lines = "\n".join(f"- 자리 {i + 1}: {why}" for i, why in sorted(problems.items()))
    msg = (f"{user}\n\n방금 낸 계획:\n{json.dumps({'hypotheses': hyps}, ensure_ascii=False)}\n\n실행할 수 없는 자리:\n{lines}"
           f"\n\n이 {len(problems)}자리만 새 가설로 바꿔서 정확히 {len(problems)}개를 낸다.")
    s = schema_five(groups)
    s["properties"]["hypotheses"] |= {"minItems": len(problems), "maxItems": len(problems)}
    try:
        new = llm.structured(PLANNER_V21 + "\n" + REPAIR, msg, s, role=role).get("hypotheses", [])
    except LLMError:
        return hyps
    out = list(hyps)
    for i, h in zip(sorted(problems), new):
        out[i] = h
    return out


def schema_five(groups: list[str]) -> dict:
    """The final run's per-test schema with exactly five items and no experiment-design variant."""
    s = plan_schema(groups)
    h = s["properties"]["hypotheses"]
    h |= {"minItems": SLOTS, "maxItems": SLOTS}
    h["items"]["anyOf"] = [v for v in h["items"]["anyOf"] if v["properties"]["test"]["enum"] != ["ab_test_design"]]
    return s


def round_context(items: list[dict], session: int, history: str | None) -> tuple[str, list[str], dict]:
    neg = session == NEGATIVE_SESSION
    groups = list(FEATURE_GROUPS) + ([NEGATIVE_GROUP] if neg else [])
    lists = {"group": groups, "feature": list(ASSOC_FEATURES), "segment": SEGMENTS}
    user = f"근거 자료:\n{evidence_text(items, neg)}\n\n사용 가능한 인자:\n{json.dumps(lists, ensure_ascii=False)}"
    return (user + (f"\n\n{history}" if history else "")), groups, lists


def plan_round(llm, items: list[dict], session: int, history: str | None, think: bool, role: str,
               attempts: int = 3, system: str = PLANNER_V2) -> list[dict]:
    """One planner turn, retried on API or JSON errors (every attempt stays in the audit log)."""
    for i in range(attempts):
        try:
            return _plan_round(llm, items, session, history, think, role, system)
        except LLMError:
            if i == attempts - 1:
                raise
            time.sleep(10 * (i + 1))
    return []


def _plan_round(llm, items: list[dict], session: int, history: str | None, think: bool, role: str,
                system: str) -> list[dict]:
    user, groups, lists = round_context(items, session, history)
    if not think:
        plan = llm.structured(system, user, schema_five(groups), role=role)
    else:
        draft = llm.text(system + THINK_EXTRA, user, role=f"{role}_think", max_tokens=8192, thinking="medium")
        plan = llm.structured(TRANSCRIBE, f"{draft}\n\n사용 가능한 인자:\n{json.dumps(lists, ensure_ascii=False)}",
                              schema_five(groups), role=f"{role}_transcribe")
    return list(plan.get("hypotheses", []))[:SLOTS]


def run(scripted: bool = False, out: Path = OUT, v21: bool = False) -> dict:
    """v2 (§19) or, with v21, the same run with the repair step and citable feedback ids (§19.9)."""
    items = json.loads((out / "evidence.json").read_text())["items"]
    known = {e["id"] for e in items}
    cache: dict = {}
    record = {"version": "v2.1" if v21 else "v2", "arms": {a: {} for a in ARMS}, "common_round_1": {}, "repairs": []}
    tag = "_v21" if v21 else ""
    dest = out / (f"run{tag}_scripted.json" if scripted else f"run{tag}.json")
    audit = AuditLog(out / f"audit{tag}.jsonl")
    system = PLANNER_V21 if v21 else PLANNER_V2

    def turn(llm, k, rounds, arm, tested, known_now, role, history, think):
        hyps = plan_round(llm, items, k, history, think=think, role=role, system=system)
        if v21:
            probs = slot_problems(hyps, tested, known_now)
            if probs:
                user, groups, _ = round_context(items, k, history)
                hyps = repair(llm, user, hyps, probs, groups, role=f"{role}_repair")
                after = slot_problems(hyps, tested, known_now)
                record["repairs"].append({"role": role, "before": len(probs), "after": len(after)})
        return hyps

    def client(k: int):
        return ScriptedLLM(audit, scripted_plans(k), [scripted_text(k)] * 24) if scripted else ClovaClient(audit, seed=k)

    for k in range(1, SESSIONS + 1):
        llm = client(k)
        known_1 = known | ({FALSE_EVIDENCE["id"]} if k == NEGATIVE_SESSION else set())
        r1 = turn(llm, k, [], None, set(), known_1, f"s{k}_r1", None, False)
        record["common_round_1"][k] = r1
        for arm in ARMS:
            neg = k == NEGATIVE_SESSION
            known_k = known | ({FALSE_EVIDENCE["id"]} if neg else set())
            tested, lond, rounds = set(), Lond(), []
            control = control_hypothesis({"hypotheses": [{"id": i} for i in range(len(r1))]}) if neg else None
            for day in range(1, ROUNDS + 1):
                known_now = known_k | (history_ids(rounds) if v21 else set())
                hyps = r1 if day == 1 else turn(llm, k, rounds, arm, tested, known_now, f"s{k}_{arm}_r{day}",
                                                history_text(rounds, arm != "H-mem", ids=v21), arm == "H-fb-think")
                rows = execute(hyps, day, cache, tested, lond, known_now)
                if control is not None and day == 1:  # outside the five slots and the score
                    rows.append(execute([control], day, cache, tested, Lond(), known_k)[0] | {"control": True})
                rounds.append(rows)
            record["arms"][arm][k] = rounds
        dest.write_text(json.dumps(record, ensure_ascii=False, indent=1, default=str))  # after every session
    return record


# ----------------------------------------------------------------------------------------------- rule baselines


def rule_order(ev: dict) -> list[dict]:
    """Evidence rule: the tests the evidence items name directly, associations ordered by the effect the evidence
    states (largest first), then a fixed list of ablations, then the rest of the menu in its own order."""
    names = {"같은 상품": "same_item_in_history", "같은 ad_feat1 값": "same_category_in_history",
             "같은 ad_feat2 값": "same_shop_in_history"}
    assoc = [(abs(v), names[k], v) for k, v in ev["association_effects"].items()]
    assoc.append((abs(ev["visual_top_minus_bottom"]), "high_visual_similarity", ev["visual_top_minus_bottom"]))
    assoc.append((abs(ev["empty_minus_has"]), "empty_history", ev["empty_minus_has"]))
    first = [hyp("association", "all", feature=f, expected="higher_cvr_with_feature" if v > 0 else "lower_cvr_with_feature")
             for _, f, v in sorted(assoc, reverse=True)]
    first += [hyp("ablation_effect", "all", group="history_match"), hyp("ablation_effect", "all", group="visual_similarity"),
              hyp("ablation_effect", "sparse_G1", group="history_match"), hyp("ablation_effect", "sparse_G1", group="visual_similarity"),
              hyp("ablation_effect", "has_history", group="history_match"),
              hyp("interaction", "sparse_G1", group="history_match", segment_b="dense"),
              hyp("ablation_effect", "all", group="user_profile"), hyp("ablation_effect", "all", group="context"),
              hyp("ablation_effect", "all", group="item_attributes")]
    seen = {canon(h)[0] for h in first}
    return first + [h for h in menu_tests() if canon(h)[0] not in seen]


def neighbours(h: dict) -> list[dict]:
    """The same feature or group in the segments not yet looked at."""
    if h["tool"] == "association":
        return [hyp("association", s, feature=h["feature"], expected=h["expected"]) for s in SEGMENTS
                if s != h["segment"] and empty_arm_by_definition(h["feature"], s) is None]
    if h["tool"] == "ablation_effect":
        return [hyp("ablation_effect", s, group=h["group"]) for s in SEGMENTS if s != h["segment"]]
    return [hyp("interaction", h["segment"], group=h["group"], segment_b=s) for s in SEGMENTS
            if s not in (h["segment"], h["segment_b"])]


def run_rules(out: Path = OUT) -> dict:
    ev = json.loads((out / "evidence.json").read_text())
    order = rule_order(ev)
    known = {e["id"] for e in ev["items"]}
    cache: dict = {}
    res = {}
    for name in ("evidence_rule", "adaptive_rule"):
        tested, lond, rounds, queue = set(), Lond(), [], list(order)
        for day in range(1, ROUNDS + 1):
            pick = []
            if name == "adaptive_rule" and rounds:
                for r in [r for rr in rounds for r in rr if r["verdict"] == "confirmed"]:
                    pick += [n for n in neighbours(r["hypothesis"]) if canon(n)[0] not in tested]
            pick = list({canon(h)[0]: h for h in pick if canon(h)[0] not in tested}.values())[:SLOTS]
            for h in queue:
                if len(pick) == SLOTS:
                    break
                if canon(h)[0] not in tested and canon(h)[0] not in {canon(p)[0] for p in pick}:
                    pick.append(h)
            rounds.append(execute(pick, day, cache, tested, lond, known))
        res[name] = rounds
    (out / "rules.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str))
    return res


# ---------------------------------------------------------------------------------------------------- scoring


def score(out: Path = OUT, run_file: str = "run.json", audit_file: str = "audit.jsonl",
          summary_file: str = "summary.json") -> dict:
    passing = {d: passing_set(json.loads((out / f"exhaustive_r{d}.json").read_text())["results"]) for d in MENUS}
    run_rec = json.loads((out / run_file).read_text())
    rules = json.loads((out / "rules.json").read_text())

    def round_score(rows: list[dict], day: int) -> dict:
        slots = [r for r in rows if not r.get("control")][:SLOTS]
        hits = sum(r["verdict"] not in ("invalid", "repeat") and is_hit(r["hypothesis"], passing[day]) for r in slots)
        return {"hits": int(hits), "invalid": sum(r["verdict"] == "invalid" for r in slots),
                "repeat": sum(r["verdict"] == "repeat" for r in slots), "empty": SLOTS - len(slots),
                "lond_confirmed": sum(r["verdict"] == "confirmed" for r in slots)}

    def cumulative(rounds: list[list[dict]]) -> int:
        keys = set()
        for day, rows in enumerate(rounds, start=1):
            keys |= {r["key"] for r in rows if not r.get("control") and r.get("key") and is_hit(r["hypothesis"], passing[day])}
        return len(keys)

    arms = {}
    for arm, sessions in run_rec["arms"].items():
        arms[arm] = {k: {"rounds": [round_score(rows, d) for d, rows in enumerate(rounds, start=1)],
                         "cumulative_unique_hits": cumulative(rounds)} for k, rounds in sessions.items()}
    base = {name: {"rounds": [round_score(rows, d) for d, rows in enumerate(rounds, start=1)],
                   "cumulative_unique_hits": cumulative(rounds)} for name, rounds in rules.items()}
    tests = menu_tests()
    base["random"] = {"expected_hits_per_round": [SLOTS * random_hit_rate(tests, passing[d]) for d in MENUS]}
    pairs = {f"R{d}": {k: arms["H-fb"][k]["rounds"][d - 1]["hits"] - arms["H-mem"][k]["rounds"][d - 1]["hits"]
                       for k in arms["H-fb"]} for d in (2, 3)}
    think = {f"R{d}": {k: arms["H-fb-think"][k]["rounds"][d - 1]["hits"] - arms["H-fb"][k]["rounds"][d - 1]["hits"]
                       for k in arms["H-fb"]} for d in (2, 3)}
    controls = {arm: [r["verdict"] for rounds in s.values() for rows in rounds for r in rows if r.get("control")]
                for arm, s in run_rec["arms"].items()}
    usage = usage_summary(out / audit_file)
    summary = {"passing_set_sizes": {DAYS[d]: len(p) for d, p in passing.items()}, "menu_tests": len(tests),
               "arms": arms, "baselines": base, "feedback_pairs_fb_minus_mem": pairs, "think_pairs_minus_fb": think,
               "feedback_helped_rounds": [r for r, v in pairs.items() if all(x > 0 for x in v.values())],
               "negative_control": controls, "usage": usage}
    summary["repairs"] = run_rec.get("repairs", [])
    (out / summary_file).write_text(json.dumps(summary, ensure_ascii=False, indent=1))
    return summary


def random_hit_rate(tests: list[dict], passing: dict[str, int]) -> float:
    """Chance that a uniformly drawn menu test, with a direction drawn at random where the tool has one, is a hit.
    Model comparisons carry their fixed direction ("helps"); associations pick higher or lower with equal odds."""
    p = 0.0
    for h in tests:
        key = canon(h)[0]
        if key in passing:
            p += 0.5 if h["tool"] == "association" else float(passing[key] == -1)
    return p / len(tests)


def usage_summary(audit: Path) -> dict:
    if not audit.exists():
        return {}
    rows = [json.loads(line) for line in audit.read_text().splitlines() if line.strip()]
    out: dict = {}
    for r in rows:
        kind = next((s for s in ("think", "transcribe", "repair") if r["role"].endswith(f"_{s}")), "plan")
        u = r.get("usage") or {}
        o = out.setdefault(kind, {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "thinking_tokens": 0,
                                  "seconds": 0.0})
        o["calls"] += 1
        o["prompt_tokens"] += int(u.get("promptTokens") or 0)
        o["completion_tokens"] += int(u.get("completionTokens") or 0)  # includes the thinking tokens
        o["thinking_tokens"] += int((u.get("completionTokensDetails") or {}).get("thinkingTokens") or 0)
        o["seconds"] += float(r.get("seconds") or 0)
    return out


# ---------------------------------------------------------------------------------------- scripted stand-ins


def scripted_plans(k: int) -> list[dict]:
    """Fixed plans for tests and dry runs (enough for every structured call of a session)."""
    base = [{"id": i, "statement": "", "claim_type": "associational", "evidence": ["E2"], "test": "association",
             "feature": f, "segment": s, "direction": "higher_cvr_with_feature"}
            for i, (f, s) in enumerate(itertools.product(ASSOC_FEATURES, SEGMENTS[: 3]), start=1)]
    plans = []
    for j in range(24):
        chunk = base[(j * SLOTS + k) % len(base):][:SLOTS]
        plans.append({"hypotheses": (chunk + base)[:SLOTS]})
    return plans


def scripted_text(k: int) -> str:
    return f"scripted draft for session {k}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["evidence", "exhaustive", "run", "rules", "score"])
    ap.add_argument("--scripted", action="store_true")
    ap.add_argument("--v21", action="store_true", help="the repair step and citable feedback ids (§19.9)")
    a = ap.parse_args()
    if a.cmd == "evidence":
        print(json.dumps(evidence(), ensure_ascii=False, indent=1))
    elif a.cmd == "exhaustive":
        for d in MENUS:
            r = exhaustive(d)
            print(DAYS[d], r["tests"], "tests", r["seconds"], "s", flush=True)
    elif a.cmd == "run":
        run(scripted=a.scripted, v21=a.v21)
    elif a.cmd == "rules":
        run_rules()
    elif a.v21:
        print(json.dumps(score(run_file="run_v21.json", audit_file="audit_v21.jsonl", summary_file="summary_v21.json"), indent=1))
    else:
        print(json.dumps(score(), indent=1))
        sys.exit(0)
