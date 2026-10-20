"""Claude via the official Anthropic SDK."""

import time
from collections.abc import Iterator

import anthropic

from aiact.llm.base import LLMResponse, Usage

# Models that accept server-side refusal fallbacks (a declined request is re-run on another model).
_FALLBACK_MODELS = ("claude-opus-5", "claude-sonnet-5-5", "claude-fable-5")


class AnthropicLLM:
    def __init__(self, model: str, effort: str = "medium", max_tokens: int = 4000):
        self.model = model
        self.effort = effort
        self.max_tokens = max_tokens  # answers are a few sentences; thinking shares this budget
        # credentials from ANTHROPIC_API_KEY or an `ant auth login` profile
        self.client = anthropic.Anthropic()

    def _kwargs(self, system: str, messages: list[dict]) -> dict:
        kwargs: dict = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            # The instructions are identical on every call; the per-question sources go in the user turn.
            "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
            "messages": messages,
            "output_config": {"effort": self.effort},
        }
        if self.model.startswith(_FALLBACK_MODELS):
            kwargs |= {"betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        return kwargs

    def _response(self, msg, start: float) -> LLMResponse:
        u = msg.usage
        return LLMResponse(
            text="".join(b.text for b in msg.content if b.type == "text"),
            stop_reason=msg.stop_reason or "end_turn",
            model=msg.model,
            usage=Usage(
                input_tokens=u.input_tokens,
                output_tokens=u.output_tokens,
                cache_read_tokens=u.cache_read_input_tokens or 0,
                cache_write_tokens=u.cache_creation_input_tokens or 0,
            ),
            latency_ms=round((time.perf_counter() - start) * 1000, 1),
        )

    def complete(self, system: str, messages: list[dict], json_schema: dict | None = None) -> LLMResponse:
        # json_schema is not sent: the judge prompts already embed the schema, and the caller validates the reply
        # with pydantic and retries once, which strong models don't need in practice.
        start = time.perf_counter()
        return self._response(self.client.beta.messages.create(**self._kwargs(system, messages)), start)

    def stream(self, system: str, messages: list[dict]) -> Iterator[str | LLMResponse]:
        start = time.perf_counter()
        with self.client.beta.messages.stream(**self._kwargs(system, messages)) as s:
            yield from s.text_stream
            msg = s.get_final_message()
        yield self._response(msg, start)  # stop_reason "refusal" is checked by the caller, as with complete()
