"""Scripted planner / reporter for running a session without an API key (tests, dry runs).

The plan deliberately mixes valid hypotheses with ones the code must catch: a causal claim given an
observational tool (must be rerouted to ab_test_design), a tool that does not fit the claim type, and an
unknown evidence id. The reporter copies the code verdicts and numbers, so a correct pipeline passes.
"""

import json

from .llm import ScriptedLLM

PLAN = {"hypotheses": [
    {"id": 1, "statement": "구매 이력 일치 정보는 희소 그룹에서 전환 예측에 도움이 된다", "claim_type": "predictive",
     "evidence": ["E2", "E3"], "tool": "ablation_effect", "group": "history_match", "feature": "none",
     "segment": "sparse_G1", "segment_b": "none", "relative_mde": 0.05, "expected": "group_helps"},
    {"id": 2, "statement": "구매 이력이 없는 클릭에서는 시각 유사도의 도움이 이력이 있는 클릭보다 작다", "claim_type": "predictive",
     "evidence": ["E7"], "tool": "interaction", "group": "visual_similarity", "feature": "none",
     "segment": "has_history", "segment_b": "no_history", "relative_mde": 0.05, "expected": "group_helps_more_in_segment"},
    {"id": 3, "statement": "같은 카테고리 구매 이력이 있으면 전환율이 높다", "claim_type": "associational",
     "evidence": ["E3"], "tool": "association", "group": "none", "feature": "same_category_in_history",
     "segment": "all", "segment_b": "none", "relative_mde": 0.05, "expected": "higher_cvr_with_feature"},
    {"id": 4, "statement": "광고 이미지를 과거 구매와 비슷하게 바꾸면 전환이 늘어난다", "claim_type": "causal",
     "evidence": ["E3"], "tool": "association", "group": "none", "feature": "high_visual_similarity",
     "segment": "all", "segment_b": "none", "relative_mde": 0.05, "expected": "needs_experiment"},
    {"id": 5, "statement": "맥락 특성은 희소 그룹에서 도움이 된다", "claim_type": "associational",
     "evidence": ["E99"], "tool": "ablation_effect", "group": "context", "feature": "none",
     "segment": "sparse_G1", "segment_b": "none", "relative_mde": 0.05, "expected": "group_helps"},
]}


def _echo_report(system: str, user: str) -> dict:
    payload = json.loads(user)
    items = []
    for h in payload:
        r = h["result"]
        nums = " ".join(f"{k}={round(v, 4) if isinstance(v, float) else v}" for k, v in r.items()
                        if isinstance(v, (int, float)) and not isinstance(v, bool))
        items.append({"id": h["id"], "verdict": h["verdict"], "summary": f"{h['statement']}: {nums}",
                      "next_step": "다음 실험을 설계한다"})
    return {"items": items, "overall": "코드가 판정한 결과를 그대로 요약했다"}


class _Scripted(ScriptedLLM):
    def structured(self, system, user, schema, role, max_tokens=4096):
        out = self.s.pop(0)
        if callable(out):
            out = out(system, user)
        self.audit.write({"role": role, "model": "scripted", "request": {"system": system, "user": user[:2000]},
                          "content": out})
        return out


def scripted_llm(audit) -> ScriptedLLM:
    return _Scripted(audit, [PLAN, _echo_report], [])
