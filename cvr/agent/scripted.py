"""Scripted planner / reporter for running the agent without an API key (tests, dry runs).

This is a test double with fixed answers, not a validation with a real LLM. The plans mix valid
hypotheses with ones the code must catch: a causal claim given an observational tool (rerouted to
ab_test_design), a tool that does not fit the claim type, an unknown evidence id (E999), and in the
negative-control session the planted user_score claim. The reporter writes digit-free text.
"""

from .llm import ScriptedLLM

BASE = [
    {"id": 1, "statement": "구매 이력 일치 정보는 희소 그룹에서 전환 예측에 도움이 된다", "claim_type": "predictive",
     "evidence": ["E2", "E3"], "tool": "ablation_effect", "group": "history_match", "feature": "none",
     "segment": "sparse_G1", "segment_b": "none", "relative_mde": 0.05, "expected": "group_helps"},
    {"id": 2, "statement": "같은 카테고리 구매 이력이 있으면 전환율이 높다", "claim_type": "associational",
     "evidence": ["E3"], "tool": "association", "group": "none", "feature": "same_category_in_history",
     "segment": "all", "segment_b": "none", "relative_mde": 0.05, "expected": "higher_cvr_with_feature"},
    {"id": 3, "statement": "광고 이미지를 과거 구매와 비슷하게 바꾸면 전환이 늘어난다", "claim_type": "causal",
     "evidence": ["E3"], "tool": "association", "group": "none", "feature": "high_visual_similarity",
     "segment": "all", "segment_b": "none", "relative_mde": 0.05, "expected": "needs_experiment"},
    {"id": 4, "statement": "맥락 특성은 희소 그룹에서 도움이 된다", "claim_type": "predictive",
     "evidence": ["E999"], "tool": "ablation_effect", "group": "context", "feature": "none",
     "segment": "sparse_G1", "segment_b": "none", "relative_mde": 0.05, "expected": "group_helps"},
    # empty "with" arm because of the data (the synthetic menu has no same-item match), not by definition
    {"id": 5, "statement": "구매 이력이 있으면 같은 상품이 이력에 있을 때 전환율이 높다", "claim_type": "associational",
     "evidence": ["E7"], "tool": "association", "group": "none", "feature": "same_item_in_history",
     "segment": "has_history", "segment_b": "none", "relative_mde": 0.05, "expected": "higher_cvr_with_feature"},
]
NEGATIVE = {"id": 6, "statement": "사용자 점수는 희소 그룹 전환 예측에 도움이 된다", "claim_type": "predictive",
            "evidence": ["E8"], "tool": "ablation_effect", "group": "user_score", "feature": "none",
            "segment": "sparse_G1", "segment_b": "none", "relative_mde": 0.05, "expected": "group_helps"}


def _report(system: str, user: str) -> dict:
    import json

    items = [{"id": h["id"], "interpretation": "코드 판정의 범위 안에서만 해석한다",
              "next_step": "다른 기간의 자료로 같은 비교를 다시 확인한다"} for h in json.loads(user)]
    return {"items": items, "overall": "판정은 코드가 내렸고 해석은 그 범위를 넘지 않는다"}


class _Scripted(ScriptedLLM):
    """Answers by call role, not by call order: a planner call gets the plan, a reporter call gets _report."""

    def __init__(self, audit, plan: dict):
        super().__init__(audit, [], [])
        self.plan = plan

    def structured(self, system, user, schema, role, max_tokens=4096):
        out = self.plan if role.startswith("planner") else _report(system, user)
        self.audit.write({"role": role, "model": "scripted", "content": out})
        return out


def scripted_factory(audit, negative_session: int | None = None):
    """make_llm(k): session k gets the base plan (plus the planted control in the negative session).

    Answers depend on the role of the call. An earlier queue (plan first, then report) broke twice: a fresh
    instance per call handed the PLAN to the reporter, so every report item was missing and the LLM-text
    checks passed without checking any text; and an execution from a frozen plan file (no planner call)
    did the same."""

    def make(k: int):
        return _Scripted(audit, {"hypotheses": BASE + ([NEGATIVE] if k == negative_session else [])})
    return make
