"""Three retrieval modes over the same index:

- dense:  cosine kNN on the dense vector.
- hybrid: dense + BM25 candidate lists fused with Reciprocal Rank Fusion inside Qdrant.
- rerank: hybrid candidates re-scored by a cross-encoder. Its top score doubles as the evidence-strength signal
          used for refusal. Optionally ranks recitals below operative text (recital_penalty): recitals explain
          the law in plain words and outrank the binding article, but an answer must cite the article.
"""

import json
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from qdrant_client import models

from aiact.chunking import Chunk
from aiact.index import DenseEncoder, Index, Reranker, SparseEncoder

MODES = ("dense", "hybrid", "rerank")


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float  # cosine for dense, RRF score for hybrid, cross-encoder logit for rerank


@dataclass
class Result:
    hits: list[Hit]
    latency_ms: float

    @property
    def top_score(self) -> float:
        """Best raw score, independent of any rank prior: the refusal signal stays calibrated."""
        return max((h.score for h in self.hits), default=float("-inf"))


class Retriever:
    def __init__(
        self,
        index: Index,
        dense: DenseEncoder,
        sparse: SparseEncoder,
        reranker: Reranker | None = None,
        candidates: int = 30,
        recital_penalty: float = 0.0,
    ):
        self.index, self.dense, self.sparse, self.reranker = index, dense, sparse, reranker
        self.candidates, self.recital_penalty = candidates, recital_penalty
        # Qdrant local mode and the ONNX sessions are shared in-process: serialize searches from the API's
        # threadpool. LLM calls (the slow part) still run concurrently. At scale: a Qdrant server + replicas.
        self._lock = threading.Lock()

    def _query(self, query: str, mode: str, limit: int) -> list[Hit]:
        dq = self.dense.embed_query(query).tolist()
        if mode == "dense":
            res = self.index.client.query_points(self.index.collection, query=dq, using="dense", limit=limit)
        else:
            res = self.index.client.query_points(
                self.index.collection,
                prefetch=[
                    models.Prefetch(query=dq, using="dense", limit=self.candidates),
                    models.Prefetch(
                        query=self.sparse.embed_query(query), using="bm25", limit=self.candidates
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=limit,
            )
        return [Hit(self.index.chunks[p.id], p.score) for p in res.points]

    def search(self, query: str, mode: str = "rerank", k: int = 10) -> Result:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")
        with self._lock:
            return self._search(query, mode, k)

    def _search(self, query: str, mode: str, k: int) -> Result:
        t0 = time.perf_counter()
        if mode == "rerank":
            if self.reranker is None:
                raise ValueError("rerank mode needs a reranker")
            cands = self._query(query, "hybrid", self.candidates)
            scores = self.reranker.score(query, [h.chunk.text for h in cands])
            hits = sorted((Hit(h.chunk, s) for h, s in zip(cands, scores, strict=True)), key=self._rank_key)
            hits = hits[:k]
        else:
            hits = self._query(query, mode, k)
        return Result(hits, (time.perf_counter() - t0) * 1000)

    def _rank_key(self, h: Hit) -> float:
        return -(h.score - (self.recital_penalty if h.chunk.provision.startswith("Recital") else 0.0))


class SnapshotRetriever:
    """Record/replay retrieval results, like the LLM cassette.

    The answer eval replays a snapshot so CI grades generation without downloading models, and so tiny ONNX
    float differences between machines can't reorder sources, change the prompt and miss the LLM cassette.
    Retrieval itself is graded separately by the bench gate. inner=None -> replay only; inner given -> record.
    """

    def __init__(self, path: Path, inner: Retriever | None = None):
        self.path, self.inner = path, inner
        self.entries: dict = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    def search(self, query: str, mode: str = "rerank", k: int = 10) -> Result:
        key = f"{mode}|{k}|{query}"
        if key not in self.entries:
            if self.inner is None:
                raise KeyError(f"no retrieval snapshot for {key!r} in {self.path.name}: re-record")
            res = self.inner.search(query, mode, k)
            self.entries[key] = {
                "latency_ms": res.latency_ms,
                "hits": [{**asdict(h.chunk), "score": h.score} for h in res.hits],
            }
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.entries, indent=1, ensure_ascii=False), encoding="utf-8")
        e = self.entries[key]
        hits = [
            Hit(Chunk(h["id"], h["text"], tuple(h["citations"]), h["provision"]), h["score"])
            for h in e["hits"]
        ]
        return Result(hits, e["latency_ms"])
