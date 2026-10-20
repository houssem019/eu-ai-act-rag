# EU AI Act Assistant — Production RAG Service

RAG service answering questions on EU AI Act (Regulation (EU) 2024/1689) obligations with article-level citations,
hybrid retrieval (BM25 + dense), cross-encoder reranking, refusal on insufficient evidence and Ragas-gated CI.

> 🚧 Work in progress. See `docs/BUILD_LOG.md`.

## Corpus
The source is the official English Official Journal XHTML of Regulation (EU) 2024/1689, committed in `data/raw/`. It's parsed into `data/corpus/ai_act.jsonl`: **306 provisions** (113 articles, 180 recitals, 13 annexes; 84.6k words) split into **786 citable units** (`Article 5(1)`, `Article 3(44)`, `Recital 27`, `Annex III`).

```bash
uv sync --python 3.11
uv run python -m aiact.corpus.parse     # rebuild the corpus from the committed XHTML
uv run pytest
```
