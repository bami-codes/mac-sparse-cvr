"""LLM clients. Every request and response is appended to an audit log (jsonl).

ClovaClient calls HyperCLOVA X through the CLOVA Studio Chat Completions v3 API:
    POST https://clovastudio.stream.ntruss.com/v3/chat-completions/HCX-007
    Authorization: Bearer $CLOVASTUDIO_API_KEY
Structured outputs (`responseFormat` with a JSON schema) are used for the planner; the JSON comes
back as a string in result.message.content. The key is read from the environment only and is
never written to disk or logs.

ScriptedLLM returns fixed outputs so the whole pipeline can be tested without an API key.
"""

import json
import os
import time
import urllib.request
import uuid
from pathlib import Path

CLOVA_URL = "https://clovastudio.stream.ntruss.com/v3/chat-completions/{model}"


class LLMError(RuntimeError):
    pass


class AuditLog:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: dict) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class ClovaClient:
    def __init__(self, audit: AuditLog, model: str = "HCX-007", temperature: float = 0.3, seed: int = 0,
                 timeout: int = 120):
        self.key = os.environ.get("CLOVASTUDIO_API_KEY")
        if not self.key:
            raise LLMError("set CLOVASTUDIO_API_KEY (CLOVA Studio test or service API key)")
        self.audit, self.model, self.temperature, self.seed, self.timeout = audit, model, temperature, seed, timeout

    def _post(self, body: dict, role: str) -> str:
        req_id = uuid.uuid4().hex
        req = urllib.request.Request(
            CLOVA_URL.format(model=self.model), data=json.dumps(body).encode(), method="POST",
            headers={"Authorization": f"Bearer {self.key}", "X-NCP-CLOVASTUDIO-REQUEST-ID": req_id,
                     "Content-Type": "application/json"})
        t = time.time()
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            out = json.loads(r.read().decode())
        status = out.get("status", {}).get("code")
        content = (out.get("result") or {}).get("message", {}).get("content")
        self.audit.write({"role": role, "model": self.model, "request_id": req_id, "seconds": round(time.time() - t, 2),
                          "request": body, "status": status, "usage": (out.get("result") or {}).get("usage"),
                          "finish_reason": (out.get("result") or {}).get("finishReason"), "content": content})
        if status != "20000" or content is None:
            raise LLMError(f"CLOVA Studio returned status {status}")
        return content

    def structured(self, system: str, user: str, schema: dict, role: str, max_tokens: int = 4096) -> dict:
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "responseFormat": {"type": "json", "schema": schema}, "maxCompletionTokens": max_tokens,
                "temperature": self.temperature, "seed": self.seed}
        text = self._post(body, role)
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"structured output is not valid JSON: {e}") from e

    def text(self, system: str, user: str, role: str, max_tokens: int = 4096) -> str:
        body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "maxCompletionTokens": max_tokens, "temperature": self.temperature, "seed": self.seed,
                "thinking": {"effort": "none"}}
        return self._post(body, role)


class ScriptedLLM:
    """Deterministic stand-in: returns the given outputs in order (for tests and dry runs)."""

    def __init__(self, audit: AuditLog, structured_outputs: list[dict], text_outputs: list[str]):
        self.audit, self.s, self.t = audit, list(structured_outputs), list(text_outputs)

    def structured(self, system: str, user: str, schema: dict, role: str, max_tokens: int = 4096) -> dict:
        out = self.s.pop(0)
        self.audit.write({"role": role, "model": "scripted", "request": {"system": system, "user": user}, "content": out})
        return out

    def text(self, system: str, user: str, role: str, max_tokens: int = 4096) -> str:
        out = self.t.pop(0)
        self.audit.write({"role": role, "model": "scripted", "request": {"system": system, "user": user}, "content": out})
        return out
