"""Serving: answer streaming, the guarded semantic cache, metrics and the HTTP API. Offline (fakes)."""

import json
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient

from aiact.answer import REFUSAL_SENTINEL, Answerer
from aiact.api import create_app
from aiact.cache import AnswerCache, discriminators, normalize
from aiact.chunking import Chunk
from aiact.evals.cache_bench import evaluate
from aiact.llm.replay import ScriptedLLM
from aiact.metrics import Metrics
from aiact.retrieve import Hit, Result

HITS = [
    Hit(
        Chunk("a", "Article 5 - Prohibited AI practices\n(c) social scoring", ("Article 5(1)",), "Article 5"),
        6.0,
    ),
    Hit(Chunk("b", "Article 99 - Penalties\n35 000 000 EUR", ("Article 99(3)",), "Article 99"), 4.0),
]


class WordEncoder:
    """Bag-of-words embedding: questions sharing most words are 'similar'."""

    def embed_query(self, text):
        v = np.zeros(512, dtype=np.float32)
        for w in normalize(text).split():
            v[zlib.crc32(w.encode()) % 512] += 1
        return v


class FakeRetriever:
    dense = WordEncoder()

    def __init__(self, hits=HITS):
        self.hits, self.calls = hits, 0

    def search(self, query, mode="rerank", k=10):
        self.calls += 1
        return Result(self.hits[:k], 5.0)


def answerer(replies, threshold=2.0):
    return Answerer(FakeRetriever(), ScriptedLLM(replies), threshold=threshold)


# --------------------------------------------------------------------------- streaming
def test_stream_event_order_and_final_matches_ask():
    events = list(answerer(["Social scoring is banned [Article 5(1)(c)]."]).stream("q"))
    assert [e["type"] for e in events][0] == "retrieval" and events[-1]["type"] == "final"
    deltas = "".join(e["text"] for e in events if e["type"] == "delta")
    assert deltas == "Social scoring is banned [Article 5(1)(c)]."
    assert events[-1]["answer"].citations == ["Article 5(1)"]
    assert (
        answerer(["Social scoring is banned [Article 5(1)(c)]."]).ask("q").text == events[-1]["answer"].text
    )


def test_refusal_sentinel_is_never_streamed():
    events = list(answerer([REFUSAL_SENTINEL]).stream("q"))
    assert not [e for e in events if e["type"] == "delta"]
    assert events[-1]["answer"].refusal_reason == "model"


def test_repair_is_announced_and_the_second_draft_streams():
    events = list(answerer(["Banned.", "Banned [Article 5(1)]."]).stream("q"))
    types = [e["type"] for e in events]
    assert "repair" in types and types.index("repair") < len(types) - 1
    after = "".join(e["text"] for e in events[types.index("repair") :] if e["type"] == "delta")
    assert after == "Banned [Article 5(1)]."


def test_low_evidence_stream_is_retrieval_then_final():
    events = list(answerer([], threshold=99).stream("capital of France?"))
    assert [e["type"] for e in events] == ["retrieval", "final"]


# --------------------------------------------------------------------------- cache
@pytest.mark.parametrize(
    ("q", "expected"),
    [
        ("Who counts as a deployer?", {"deployer"}),
        ("What must providers of general-purpose AI models do?", {"provider", "general purpose ai model"}),
        (
            "Is real-time remote biometric identification allowed?",
            {"real time remote biometric identification system"},
        ),
        ("Is post-remote biometric identification allowed?", {"post remote biometric identification system"}),
        ("How does the AI Act define deep fakes?", {"deep fake"}),
        (
            "What does Article 5(1)(f) say about Annex III point 4?",
            {"article 5(1)(f)", "annex iii", "point 4"},
        ),
        ("Within 15 days?", {"15"}),
    ],
)
def test_discriminators(q, expected):
    assert discriminators(q) == expected


def make_cache(threshold=0.8, **kw):
    return AnswerCache(WordEncoder(), version="v1", threshold=threshold, **kw)


def test_exact_hit_ignores_case_and_punctuation():
    c = make_cache()
    c.put("Is social scoring banned?", {"text": "yes"})
    assert c.get("is social   scoring banned") == ({"text": "yes"}, "exact")


def test_semantic_hit_needs_similarity_and_same_terms():
    c = make_cache(threshold=0.7)
    c.put("What obligations do providers of high-risk AI systems have under the Act?", {"text": "providers"})
    hit = c.get("Under the Act, what obligations do providers of high-risk AI systems have exactly?")
    assert hit and hit[1] == "semantic"
    # near-identical wording, different role: blocked by the guard, counted
    assert c.get("What obligations do deployers of high-risk AI systems have under the Act?") is None
    assert c.stats.guard_rejections == 1 and c.stats.semantic_hits == 1


def test_cache_never_serves_across_versions():
    old = make_cache()
    old.put("Is social scoring banned?", {"text": "old prompt"})
    new = AnswerCache(
        WordEncoder(), version="v2", threshold=0.0
    )  # threshold 0: anything would match by cosine
    new.entries = old.entries  # same storage (think Redis), new prompt/model version
    assert new.get("Is social scoring banned?") is None


def test_cache_evicts_least_recently_used():
    c = make_cache(max_entries=2, threshold=1.01)  # exact matches only
    c.put("first question here", {"n": 1})
    c.put("second question here", {"n": 2})
    assert c.get("first question here")  # touch: "second" is now least recently used
    c.put("third question here", {"n": 3})
    assert c.get("second question here") is None and c.get("first question here")[0] == {"n": 1}


def test_cache_entries_expire():
    c = make_cache(ttl_s=0)
    c.put("Is social scoring banned?", {"text": "yes"})
    assert c.get("Is social scoring banned?") is None


def test_cache_bench_chooses_lowest_safe_threshold():
    pairs = [
        {"a": "Who is a deployer?", "b": "Who counts as a deployer?", "same": True},
        {"a": "Who is a deployer?", "b": "Who is a provider?", "same": False},  # guard blocks
        {"a": "Is X banned?", "b": "Is Y banned?", "same": False},  # guard can't see it: only the threshold
    ]
    r = evaluate(pairs, sims=[0.85, 0.97, 0.90])
    assert r["chosen"]["threshold"] == 0.92 and r["chosen"]["hit_rate"] == 0.0
    no_guard = next(x for x in r["rows"] if x["threshold"] == 0.80 and not x["guard"])
    assert no_guard["false_hits"] == 2


# --------------------------------------------------------------------------- metrics
def test_metrics_percentiles_and_split():
    m = Metrics()
    base = {
        "refused": False,
        "refusal_reason": None,
        "llm_calls": 1,
        "usage": {"input_tokens": 10, "output_tokens": 2},
    }
    for i in range(1, 11):
        m.record(
            base | {"retrieval_ms": 100.0, "latency_ms": 100.0 + i * 100, "cost_usd": 0.01}, i * 100.0, None
        )
    m.record(base | {"refused": True, "refusal_reason": "low_evidence"}, 1.0, "exact")
    s = m.snapshot()
    assert s["requests"] == 11 and s["cache"]["exact"] == 1 and s["cache"]["hit_rate"] == round(1 / 11, 3)
    assert s["latency_ms"]["llm"]["p50"] == 550.0 and s["latency_ms"]["retrieval"]["p95"] == 100.0
    assert s["outcomes"] == {"answered": 10, "refused:low_evidence": 1} and s["cost_usd"] == 0.1


# --------------------------------------------------------------------------- API
@pytest.fixture
def client():
    llm = ScriptedLLM(["Social scoring is banned [Article 5(1)(c)]."] * 5)
    retriever = FakeRetriever()
    app = create_app(Answerer(retriever, llm, threshold=2.0))
    with TestClient(app) as c:
        c.retriever, c.llm = retriever, llm
        yield c


def test_ask_then_exact_cache_hit(client):
    r1 = client.post("/ask", json={"question": "Is social scoring banned?"}).json()
    assert r1["cached"] is None and r1["citations"] == ["Article 5(1)"] and "cost_usd" in r1
    r2 = client.post("/ask", json={"question": "is social scoring banned"}).json()
    assert r2["cached"] == "exact" and r2["text"] == r1["text"]
    assert client.retriever.calls == 1 and len(client.llm.calls) == 1  # the hit cost nothing
    m = client.get("/metrics").json()
    assert m["requests"] == 2 and m["cache"]["hit_rate"] == 0.5 and m["outcomes"]["answered"] == 2


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "?"}).status_code == 422


def test_stream_sse(client):
    with client.stream("POST", "/ask/stream", json={"question": "Is social scoring banned?"}) as r:
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [json.loads(line[5:]) for line in r.iter_lines() if line.startswith("data:")]
    assert events[0]["type"] == "retrieval" and events[-1]["type"] == "final"
    assert events[-1]["answer"]["citations"] == ["Article 5(1)"] and events[-1]["answer"]["cached"] is None
    assert any(e["type"] == "delta" for e in events)


def test_health(client):
    h = client.get("/health").json()
    assert h["status"] == "ok" and h["cache_entries"] == 0 and h["model"] == "scripted"
