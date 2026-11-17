"""One small HyperCLOVA X call per request type, to check the API integration before the agent runs.

    PYTHONPATH=. uv run python scripts/clova_smoke.py runs/clova_smoke

Prints status, finish reason and token usage only (never the key). Requests and responses go to the
audit log in the given folder.
"""

import json
import sys
from pathlib import Path

from cvr.agent.llm import AuditLog, ClovaClient

out = Path(sys.argv[1])
llm = ClovaClient(AuditLog(out / "audit.jsonl"), seed=1)
schema = {"type": "object", "required": ["hypotheses"], "properties": {"hypotheses": {
    "type": "array", "minItems": 1, "maxItems": 2, "items": {"type": "object", "required": ["statement", "claim_type"],
                                                             "properties": {"statement": {"type": "string"},
                                                                            "claim_type": {"type": "string",
                                                                                           "enum": ["predictive", "causal"]}}}}}}
res = llm.structured("너는 분석가다. JSON으로만 답한다.", "구매 이력과 전환에 대한 가설을 하나만 써라.", schema,
                     role="smoke_structured", max_tokens=512)
print("structured OK:", json.dumps(res, ensure_ascii=False))
log = [json.loads(line) for line in (out / "audit.jsonl").read_text().splitlines()]
print("status / finish / usage:", log[-1].get("status"), log[-1].get("finish_reason"), log[-1].get("usage"))
