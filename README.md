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

## Retrieval benchmark
Three chunking strategies × two embedders × three retrieval modes, run on the curated eval set (`data/eval/questions.yaml`). The run itself is `uv run python -m aiact.bench` and writes `data/bench/retrieval.{json,md}`.
- **Chunking strategies:**
  - `article`: a provision packed to 400 tokens.
  - `paragraph`: one citable unit plus an "Article N - title (Chapter)" header.
  - `fixed`: 256-token windows, no headers.
- **Embedders:** `BAAI/bge-small-en-v1.5` and `bge-base-en-v1.5`, via fastembed/ONNX.
- **Modes:** `dense`, `hybrid` (dense + BM25 fused by RRF in Qdrant), and `rerank` (30 hybrid candidates re-scored by `ms-marco-MiniLM-L-6-v2`).

**Chosen:** `paragraph` + `bge-small` + `rerank`, with refusal when the top rerank score is < 2.36. The scope is small: 55 answerable and 8 unanswerable questions, single run, laptop CPU.

Answerable questions: 55. Hit = any gold citation in the top-k chunks. ctx@5 = mean tokens in the top-5 chunks. Latency is per query on CPU (laptop), including query embedding.

| Chunking | Embedder | Mode | R@1 | R@5 | R@10 | MRR@10 | ctx@5 tokens | p50 ms |
|---|---|---|---|---|---|---|---|---|
| article | bge-small | dense | 0.31 | 0.73 | 0.78 | 0.478 | 1342 | 78 |
| article | bge-small | hybrid | 0.29 | 0.78 | 0.87 | 0.509 | 1397 | 141 |
| article | bge-small | rerank | 0.35 | 0.78 | 0.87 | 0.557 | 1353 | 5418 |
| paragraph | bge-small | dense | 0.51 | 0.84 | 0.91 | 0.647 | 875 | 78 |
| paragraph | bge-small | hybrid | 0.35 | 0.87 | 0.91 | 0.559 | 1004 | 167 |
| paragraph | bge-small | rerank | 0.56 | 0.89 | 0.93 | 0.712 | 899 | 5419 |
| fixed | bge-small | dense | 0.33 | 0.64 | 0.71 | 0.464 | 1280 | 77 |
| fixed | bge-small | hybrid | 0.35 | 0.78 | 0.84 | 0.531 | 1277 | 141 |
| fixed | bge-small | rerank | 0.45 | 0.78 | 0.89 | 0.604 | 1280 | 3878 |
| article | bge-base | dense | 0.29 | 0.73 | 0.82 | 0.491 | 1360 | 151 |
| article | bge-base | hybrid | 0.29 | 0.84 | 0.89 | 0.519 | 1417 | 228 |
| article | bge-base | rerank | 0.35 | 0.78 | 0.87 | 0.555 | 1367 | 5813 |
| paragraph | bge-base | dense | 0.53 | 0.91 | 0.95 | 0.689 | 855 | 85 |
| paragraph | bge-base | hybrid | 0.44 | 0.89 | 0.93 | 0.632 | 982 | 163 |
| paragraph | bge-base | rerank | 0.56 | 0.91 | 0.93 | 0.711 | 911 | 3356 |
| fixed | bge-base | dense | 0.35 | 0.69 | 0.76 | 0.481 | 1280 | 109 |
| fixed | bge-base | hybrid | 0.36 | 0.75 | 0.85 | 0.522 | 1277 | 119 |
| fixed | bge-base | rerank | 0.45 | 0.78 | 0.89 | 0.604 | 1280 | 2484 |

Refusal threshold on the top rerank score (55 answerable vs 8 unanswerable questions):

| Chunking | Embedder | Threshold | Answer rate (answerable) | Refusal rate (unanswerable) |
|---|---|---|---|---|
| article | bge-small | 1.78 | 0.93 | 0.88 |
| paragraph | bge-small | 2.36 | 0.96 | 0.88 |
| fixed | bge-small | 2.23 | 0.89 | 0.88 |
| article | bge-base | 1.78 | 0.93 | 0.88 |
| paragraph | bge-base | 2.36 | 0.96 | 0.88 |
| fixed | bge-base | 2.23 | 0.89 | 0.88 |
