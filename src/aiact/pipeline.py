"""Wire the production pipeline from settings: corpus -> chunks -> Qdrant index -> retriever -> answerer."""

from aiact.answer import Answerer
from aiact.chunking import chunk
from aiact.config import Settings, settings
from aiact.index import FastEmbedBM25, FastEmbedDense, FastEmbedReranker, build_index
from aiact.llm import LLMClient, get_llm
from aiact.retrieve import Retriever


def build_retriever(cfg: Settings = settings) -> Retriever:
    dense, sparse = FastEmbedDense(cfg.embedder), FastEmbedBM25()
    index = build_index(chunk(cfg.chunking), dense, sparse, f"{cfg.chunking}_{cfg.embedder}")
    return Retriever(index, dense, sparse, FastEmbedReranker(), recital_penalty=cfg.recital_penalty)


def build_answerer(cfg: Settings = settings, llm: LLMClient | None = None, retriever=None) -> Answerer:
    return Answerer(
        retriever or build_retriever(cfg),
        llm or get_llm(cfg),
        threshold=cfg.refusal_threshold,
        top_k=cfg.top_k,
    )
