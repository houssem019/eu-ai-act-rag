"""Encoders and Qdrant index: one collection per (chunking, embedder) with a named dense vector and a BM25 sparse
vector (IDF applied server-side by Qdrant), so dense, hybrid and reranked retrieval all run off the same points.

Encoders are small protocols so tests can swap in deterministic fakes and run offline.
"""

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import Protocol

import numpy as np
from qdrant_client import QdrantClient, models

from aiact.chunking import Chunk
from aiact.config import settings

DENSE_MODELS = {"bge-small": "BAAI/bge-small-en-v1.5", "bge-base": "BAAI/bge-base-en-v1.5"}
# BGE v1.5 is trained with this instruction on short queries; fastembed does not add it for us.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class DenseEncoder(Protocol):
    name: str

    def embed_docs(self, texts: Sequence[str]) -> np.ndarray: ...
    def embed_query(self, text: str) -> np.ndarray: ...


class SparseEncoder(Protocol):
    def embed_docs(self, texts: Sequence[str]) -> list[models.SparseVector]: ...
    def embed_query(self, text: str) -> models.SparseVector: ...


class Reranker(Protocol):
    def score(self, query: str, docs: Sequence[str]) -> list[float]: ...


def _models_dir() -> str:
    return str(settings.cache_dir / "models")


class FastEmbedDense:
    def __init__(self, name: str = "bge-small"):
        self.name = name

    @cached_property
    def _model(self):
        from fastembed import TextEmbedding

        return TextEmbedding(DENSE_MODELS[self.name], cache_dir=_models_dir())

    def embed_docs(self, texts: Sequence[str]) -> np.ndarray:
        """Embed documents, cached on disk by model + content hash (re-embedding bge-base on CPU takes minutes)."""
        digest = hashlib.sha1("\x00".join(texts).encode()).hexdigest()[:16]
        path = settings.cache_dir / "embeddings" / f"{self.name}_{digest}.npy"
        if path.exists():
            return np.load(path)
        vecs = np.array(list(self._model.embed(list(texts), batch_size=32)), dtype=np.float32)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
        return vecs

    def embed_query(self, text: str) -> np.ndarray:
        return next(iter(self._model.embed([BGE_QUERY_PREFIX + text])))


class FastEmbedBM25:
    @cached_property
    def _model(self):
        from fastembed import SparseTextEmbedding

        return SparseTextEmbedding("Qdrant/bm25", cache_dir=_models_dir())

    @staticmethod
    def _to_qdrant(e) -> models.SparseVector:
        return models.SparseVector(indices=e.indices.tolist(), values=e.values.tolist())

    def embed_docs(self, texts: Sequence[str]) -> list[models.SparseVector]:
        return [self._to_qdrant(e) for e in self._model.embed(list(texts))]

    def embed_query(self, text: str) -> models.SparseVector:
        # query_embed gives each query term weight 1; document-side BM25 weights x IDF do the scoring.
        return self._to_qdrant(next(iter(self._model.query_embed(text))))


class FastEmbedReranker:
    def __init__(self, name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        self.name = name

    @cached_property
    def _model(self):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        return TextCrossEncoder(self.name, cache_dir=_models_dir())

    def score(self, query: str, docs: Sequence[str]) -> list[float]:
        return list(self._model.rerank(query, list(docs)))


@dataclass
class Index:
    client: QdrantClient
    collection: str
    chunks: list[Chunk]


def build_index(
    chunks: list[Chunk],
    dense: DenseEncoder,
    sparse: SparseEncoder,
    collection: str,
    client: QdrantClient | None = None,
) -> Index:
    client = client or QdrantClient(":memory:")
    texts = [c.text for c in chunks]
    dvecs = dense.embed_docs(texts)
    svecs = sparse.embed_docs(texts)
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config={"dense": models.VectorParams(size=dvecs.shape[1], distance=models.Distance.COSINE)},
        sparse_vectors_config={"bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)},
    )
    client.upload_points(
        collection,
        [
            models.PointStruct(
                id=i, vector={"dense": dvecs[i].tolist(), "bm25": svecs[i]}, payload={"chunk_id": c.id}
            )
            for i, c in enumerate(chunks)
        ],
    )
    return Index(client, collection, chunks)
