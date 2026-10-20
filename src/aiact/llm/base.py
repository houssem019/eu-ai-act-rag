"""Provider-neutral LLM interface. Answering is a single grounded completion (plus at most one repair turn),
so messages are plain {"role", "content": str} dicts and there are no tools. The eval judge also asks for JSON."""

from dataclasses import dataclass, field
from typing import Protocol

# USD per million tokens: (input, output, cache read). Keep in sync with provider pricing pages.
PRICING = {
    "claude-opus-5-5": (4.00, 20.00, 0.20),
    "claude-sonnet-5-5": (2.00, 10.00, 0.20),
    "claude-haiku-5-5": (0.10, 0.50, 0.01),
}


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0

    def cost_usd(self, model: str) -> float:
        inp, out, cached = PRICING.get(model, (0.0, 0.0, 0.0))  # local models are free
        return (
            self.input_tokens * inp
            + self.cache_write_tokens * inp * 1.25
            + self.cache_read_tokens * cached
            + self.output_tokens * out
        ) / 1e6

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(*(a + b for a, b in zip(vars(self).values(), vars(other).values(), strict=True)))


@dataclass
class LLMResponse:
    text: str
    stop_reason: str  # end_turn | max_tokens | refusal
    model: str
    usage: Usage = field(default_factory=Usage)
    latency_ms: float = 0.0


class LLMClient(Protocol):
    model: str

    def complete(self, system: str, messages: list[dict], json_schema: dict | None = None) -> LLMResponse:
        """json_schema: ask for JSON matching this schema (the eval judge); providers enforce it if they can."""
        ...
