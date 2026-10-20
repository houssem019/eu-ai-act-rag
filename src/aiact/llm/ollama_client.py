"""Local open-weight models via Ollama (free, runs on CPU)."""

import time

import httpx

from aiact.llm.base import LLMResponse, Usage


class OllamaLLM:
    def __init__(self, model: str, host: str, timeout_s: float = 900.0, num_ctx: int = 8192):
        self.model = model
        # Ollama's default context (2-4k tokens) silently truncates the prompt from the front, which would
        # drop the instructions once five sources are attached.
        self.num_ctx = num_ctx
        self.http = httpx.Client(base_url=host, timeout=timeout_s)

    def complete(self, system: str, messages: list[dict], json_schema: dict | None = None) -> LLMResponse:
        start = time.perf_counter()
        body = {
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_ctx": self.num_ctx},
            "messages": [{"role": "system", "content": system}, *messages],
        }
        if json_schema:
            body["format"] = json_schema  # Ollama structured outputs: decoding constrained to the schema
        r = self.http.post("/api/chat", json=body)
        r.raise_for_status()
        body = r.json()
        return LLMResponse(
            text=body["message"].get("content", ""),
            stop_reason="max_tokens" if body.get("done_reason") == "length" else "end_turn",
            model=self.model,
            usage=Usage(
                input_tokens=body.get("prompt_eval_count", 0), output_tokens=body.get("eval_count", 0)
            ),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )
