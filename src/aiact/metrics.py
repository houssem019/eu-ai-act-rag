"""In-process service metrics for /metrics: counts, latency percentiles (split into retrieval vs LLM time), cache
hit rate, refusals by trigger, tokens and cost. Bounded memory (the last `window` requests); at scale these would
be Prometheus counters/histograms scraped from every replica."""

import statistics
import threading
from collections import Counter, deque


def _pct(values: list[float], q: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return round(values[0], 1)
    return round(statistics.quantiles(values, n=100, method="inclusive")[int(q) - 1], 1)


class Metrics:
    def __init__(self, window: int = 1000):
        self._lock = threading.Lock()
        self.requests = 0
        self.errors = 0
        self.outcomes: Counter[str] = Counter()  # answered | refused:<reason>
        self.cache: Counter[str] = Counter()  # exact | semantic | miss
        self.input_tokens = 0
        self.output_tokens = 0
        self.cost_usd = 0.0
        self.total_ms: deque[float] = deque(maxlen=window)
        self.retrieval_ms: deque[float] = deque(maxlen=window)
        self.llm_ms: deque[float] = deque(maxlen=window)
        self.cached_ms: deque[float] = deque(maxlen=window)

    def record(self, answer: dict, total_ms: float, cached: str | None) -> None:
        with self._lock:
            self.requests += 1
            self.cache[cached or "miss"] += 1
            self.outcomes["answered" if not answer["refused"] else f"refused:{answer['refusal_reason']}"] += 1
            if cached:
                self.cached_ms.append(total_ms)
                return
            self.total_ms.append(total_ms)
            self.retrieval_ms.append(answer["retrieval_ms"])
            if answer["llm_calls"]:
                self.llm_ms.append(max(0.0, answer["latency_ms"] - answer["retrieval_ms"]))
            self.input_tokens += answer["usage"]["input_tokens"]
            self.output_tokens += answer["usage"]["output_tokens"]
            self.cost_usd += answer["cost_usd"]

    def record_error(self) -> None:
        with self._lock:
            self.requests += 1
            self.errors += 1

    def snapshot(self) -> dict:
        with self._lock:
            hits = self.cache["exact"] + self.cache["semantic"]
            served = sum(self.cache.values())

            def lat(d):
                v = list(d)
                return {"p50": _pct(v, 50), "p95": _pct(v, 95), "n": len(v)}

            return {
                "requests": self.requests,
                "errors": self.errors,
                "outcomes": dict(self.outcomes),
                "cache": dict(self.cache) | {"hit_rate": round(hits / served, 3) if served else 0.0},
                "latency_ms": {
                    "total_uncached": lat(self.total_ms),
                    "retrieval": lat(self.retrieval_ms),
                    "llm": lat(self.llm_ms),
                    "cached": lat(self.cached_ms),
                },
                "tokens": {"input": self.input_tokens, "output": self.output_tokens},
                "cost_usd": round(self.cost_usd, 6),
            }
