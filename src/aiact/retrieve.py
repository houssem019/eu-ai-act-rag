"""Three retrieval modes over the same index:

- dense:  cosine kNN on the dense vector.
- hybrid: dense + BM25 candidate lists fused with Reciprocal Rank Fusion inside Qdrant.
- rerank: hybrid candidates re-scored by a cross-encoder. Its top score doubles as the evidence-strength signal
          used for refusal.
"""

import time
from dataclasses import dataclass

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


class Retriever:
    def __init__(
        self,
        index: Index,
        dense: DenseEncoder,
        sparse: SparseEncoder,
        reranker: Reranker | None = None,
        candidates: int = 30,
    ):
        self.index, self.dense, self.sparse, self.reranker = index, dense, sparse, reranker
        self.candidates = candidates

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
        t0 = time.perf_counter()
        if mode == "rerank":
            if self.reranker is None:
                raise ValueError("rerank mode needs a reranker")
            cands = self._query(query, "hybrid", self.candidates)
            scores = self.reranker.score(query, [h.chunk.text for h in cands])
            hits = sorted(
                (Hit(h.chunk, s) for h, s in zip(cands, scores, strict=True)), key=lambda h: -h.score
            )
            hits = hits[:k]
        else:
            hits = self._query(query, mode, k)
        return Result(hits, (time.perf_counter() - t0) * 1000)
