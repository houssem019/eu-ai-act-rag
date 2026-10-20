"""Download the retrieval models and embed the corpus once (Docker build step, or first-run setup), then run one
query so every model is loaded. Afterwards the service starts without network access."""

import time

from aiact.config import settings
from aiact.pipeline import build_retriever


def main() -> None:
    t0 = time.perf_counter()
    retriever = build_retriever(settings)
    res = retriever.search("Which AI practices are prohibited?", "rerank", k=3)
    labels = [h.chunk.citations[0] for h in res.hits]
    print(
        f"Warm-up done in {time.perf_counter() - t0:.0f}s: {len(retriever.index.chunks)} chunks, top hits {labels}"
    )


if __name__ == "__main__":
    main()
