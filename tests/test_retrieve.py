"""Retrieval wiring against a real in-memory Qdrant, with deterministic offline fake encoders."""

import re
import zlib

import numpy as np
import pytest
from qdrant_client import models

from aiact.bench import choose_threshold, first_hit_rank, summarize
from aiact.chunking import Chunk
from aiact.index import build_index
from aiact.retrieve import Hit, Retriever

DIM = 256


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


class FakeDense:
    """Hashed bag of words: cosine similarity ~ word overlap."""

    name = "fake"

    def _vec(self, text):
        v = np.zeros(DIM, dtype=np.float32)
        for t in _tokens(text):
            v[zlib.crc32(t.encode()) % DIM] += 1
        return v / (np.linalg.norm(v) or 1)

    def embed_docs(self, texts):
        return np.stack([self._vec(t) for t in texts])

    def embed_query(self, text):
        return self._vec(text)


class FakeSparse:
    def _vec(self, text):
        counts = {}
        for t in _tokens(text):
            i = zlib.crc32(t.encode()) % 100_000
            counts[i] = counts.get(i, 0) + 1.0
        return models.SparseVector(indices=list(counts), values=list(counts.values()))

    def embed_docs(self, texts):
        return [self._vec(t) for t in texts]

    def embed_query(self, text):
        return self._vec(text)


class FakeReranker:
    """Scores by the fraction of query words present; deliberately prefers 'penalties'."""

    def score(self, query, docs):
        q = set(_tokens(query))
        return [len(q & set(_tokens(d))) / len(q) + (0.5 * ("penalties" in d.lower())) for d in docs]


CHUNKS = [
    Chunk(
        "a#0",
        "Article 5 - Prohibited AI practices\nsocial scoring by public authorities",
        ("Article 5(1)",),
        "Article 5",
    ),
    Chunk(
        "b#0",
        "Article 50 - Transparency\ndeep fake content shall be disclosed",
        ("Article 50(4)",),
        "Article 50",
    ),
    Chunk("c#0", "Article 99 - Penalties\nfines up to 35 000 000 EUR", ("Article 99(3)",), "Article 99"),
    Chunk("d#0", "Recital 1 - the purpose of this Regulation", ("Recital 1",), "Recital 1"),
]


@pytest.fixture(scope="module")
def retriever():
    index = build_index(CHUNKS, FakeDense(), FakeSparse(), "test")
    return Retriever(index, FakeDense(), FakeSparse(), FakeReranker(), candidates=4)


@pytest.mark.parametrize("mode", ["dense", "hybrid", "rerank"])
def test_modes_find_the_matching_chunk(retriever, mode):
    res = retriever.search("must deep fake content be disclosed", mode, k=2)
    assert res.hits[0].chunk.id == "b#0"
    assert res.latency_ms > 0


def test_rerank_reorders_and_returns_reranker_scores(retriever):
    res = retriever.search("fines for social scoring", "rerank", k=4)
    assert [h.chunk.id for h in res.hits][:1] == ["c#0"]  # FakeReranker's 'penalties' bonus wins
    assert res.hits[0].score == pytest.approx(1 / 4 + 0.5)


def test_bad_mode(retriever):
    with pytest.raises(ValueError):
        retriever.search("x", "bm25")


def test_rebuilding_an_index_replaces_the_collection(retriever):
    idx = build_index(CHUNKS[:2], FakeDense(), FakeSparse(), "test", client=retriever.index.client)
    assert idx.client.count("test").count == 2


def test_metrics():
    hits = [Hit(c, 0.0) for c in CHUNKS]
    assert first_hit_rank(hits, {"Article 99(3)", "Recital 1"}) == 3
    assert first_hit_rank(hits, {"Article 6(1)"}) is None
    m = summarize([1, 3, None, 12])
    assert m["recall@1"] == 0.25 and m["recall@5"] == 0.5 and m["recall@10"] == 0.5
    assert m["mrr@10"] == pytest.approx((1 + 1 / 3) / 4)


def test_choose_threshold_separates_and_prefers_answering():
    t = choose_threshold(pos=[5.0, 6.0, 1.0], neg=[-3.0, -1.0])
    assert t["threshold"] == 1.0 and t["refusal_rate"] == 1.0 and t["answer_rate"] == 1.0


def test_gate_flags_drops_and_missing_configs():
    from aiact.bench import check_gate

    result = {
        "rows": [{"chunking": "paragraph", "embedder": "bge-small", "mode": "rerank", "recall@5": 0.85}],
        "refusal": [],
    }
    fails = check_gate(
        result,
        {
            "_comment": "ignored",
            "paragraph/bge-small/rerank": {"recall@5": 0.87},
            "refusal/paragraph/bge-small": {"refusal_rate": 0.75},
        },
    )
    assert len(fails) == 2 and "0.850 < baseline 0.87" in fails[0] and "not in this run" in fails[1]
    assert check_gate(result, {"paragraph/bge-small/rerank": {"recall@5": 0.80}}) == []
