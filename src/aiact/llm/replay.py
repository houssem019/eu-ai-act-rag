"""Deterministic LLM doubles for tests and CI: no API key, no network, no cost.

* ScriptedLLM - returns fixed responses in order (unit tests).
* ReplayLLM   - a "cassette": in record mode it wraps a real LLM and saves each response keyed by a hash of the
  full request (system prompt + messages, which include the retrieved sources); in replay mode it serves them
  back. Changing the prompt or the sources changes the hash, so CI fails loudly ("cassette miss") instead of
  silently grading stale behaviour. Re-record with a real model then.
"""

import hashlib
import json
from pathlib import Path

from aiact.llm.base import LLMClient, LLMResponse, Usage


class ScriptedLLM:
    def __init__(self, replies: list[str], model: str = "scripted"):
        self.model = model
        self.replies = list(replies)
        self.calls: list[dict] = []

    def complete(self, system: str, messages: list[dict]) -> LLMResponse:
        self.calls.append({"system": system, "messages": [dict(m) for m in messages]})
        return LLMResponse(self.replies.pop(0), "end_turn", self.model, Usage(1000, 100))


class CassetteMiss(RuntimeError):
    pass


def request_key(system: str, messages: list[dict]) -> str:
    payload = json.dumps({"system": system, "messages": messages}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


class ReplayLLM:
    def __init__(self, cassette: Path, inner: LLMClient | None = None):
        """inner=None -> replay only; inner given -> record (calls inner on misses and saves)."""
        self.cassette = cassette
        self.inner = inner
        self.entries: dict = json.loads(cassette.read_text(encoding="utf-8")) if cassette.exists() else {}
        self.model = inner.model if inner else next((e["model"] for e in self.entries.values()), "replay")

    def complete(self, system: str, messages: list[dict]) -> LLMResponse:
        key = request_key(system, messages)
        if key in self.entries:
            e = self.entries[key]
            return LLMResponse(e["text"], e["stop_reason"], e["model"], Usage(**e["usage"]), e["latency_ms"])
        if self.inner is None:
            raise CassetteMiss(
                f"No recorded response for request {key} in {self.cassette.name}. The prompt or the retrieved "
                "sources changed: re-record with a real model (README: Answer evaluation)."
            )
        resp = self.inner.complete(system, messages)
        self.entries[key] = {
            "text": resp.text,
            "stop_reason": resp.stop_reason,
            "model": resp.model,
            "usage": vars(resp.usage),
            "latency_ms": resp.latency_ms,
        }
        self.cassette.parent.mkdir(parents=True, exist_ok=True)
        self.cassette.write_text(json.dumps(self.entries, indent=1, ensure_ascii=False), encoding="utf-8")
        return resp
