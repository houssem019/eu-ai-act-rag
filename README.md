# EU AI Act Assistant — Production RAG Service

RAG service answering questions on EU AI Act (Regulation (EU) 2024/1689) obligations with article-level citations,
hybrid retrieval (BM25 + dense), cross-encoder reranking, refusal on insufficient evidence and Ragas-gated CI.

> 🚧 Work in progress: retrieval and grounded answering are done; LLM-judged evaluation (Ragas) and the HTTP service are next. See `docs/BUILD_LOG.md`.

## Quickstart
```bash
uv sync --python 3.11
uv run python -m aiact.warmup                       # one-off: ~160 MB of ONNX models + corpus embeddings
LLM_PROVIDER=ollama uv run python -m aiact.ask "Do chatbots have to disclose that they are AI?"
LLM_PROVIDER=anthropic uv run python -m aiact.ask --json "Who counts as a deployer?"   # needs ANTHROPIC_API_KEY
```
Docker: `docker build -t aiact .`, then `docker run --rm -e LLM_PROVIDER=ollama -e OLLAMA_HOST=http://host.docker.internal:11434 aiact "your question"`.

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

**Chosen:** `paragraph` + `bge-small` + `rerank`. The scope is small: 55 answerable and 8 unanswerable questions, single run, laptop CPU. The grid below was measured *before* the two fixes described next.

**Fixes after the grid**, both on the production config (`data/bench/production_p0.md` and `production_p2.md`):

| Production config | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|
| paragraph + bge-small + rerank (grid) | 0.56 | 0.89 | 0.93 | 0.712 |
| + chapter cross-references spelled out in chunk text | 0.58 | 0.93 | 0.96 | 0.739 |
| + recital rank prior (−2.0 on the rerank logit) | **0.73** | **0.95** | **0.98** | **0.834** |

Recitals explain the law in plain words and outranked the binding articles. Article 113 ("Chapters I and II shall apply from 2 February 2025") was never retrieved for "when do the prohibitions apply?" until the chapter titles were spelled out. Both knobs were tuned on the eval set, so they still need a held-out check.

The refusal threshold is a top rerank score below **1.37**, the midpoint of the gap between the classes. It answers 54/55 answerable questions and refuses 7/8 unanswerable ones at the retrieval stage. CI's `retrieval-gate` job re-runs this config with real models and fails below `data/bench/baseline.json`.

The full grid:

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

## Grounded answering
Each answer is built in five steps (`aiact.answer`):
1. **Retrieve:** the top 5 reranked chunks become `<source label="Article 5(1)">` blocks.
2. **Answer:** the LLM must cite a source label after every claim.
3. **Verify:** each citation is resolved against the retrieved labels in code. `Article 5(1)(f)` resolves to unit `Article 5(1)`; `Article 50` never matches `Article 5`.
4. **Repair:** an uncited answer, or one citing something outside the sources, gets one repair turn. Invalid citations are stripped before display.
5. **Refuse** when:
   - the top rerank score is < 1.37, checked before any LLM call;
   - the model says `INSUFFICIENT_EVIDENCE`;
   - the answer is still uncited after repair.

The LLM is pluggable via `LLM_PROVIDER`:
- `anthropic`: `claude-sonnet-5-5`, with prompt caching and server-side refusal fallbacks.
- `ollama`: local.
- `fake`: a recorded cassette, used for CI.

### Answer evaluation
63 questions (55 answerable + 8 unanswerable), qwen2.5:7b via Ollama on a laptop CPU, one recorded run (`data/eval/results/answers.md`):

| Answer rate | Refusal rate | Citation accuracy | Invalid citations in drafts | Cites an article/annex | p50 latency |
|---|---|---|---|---|---|
| 0.96 (53/55) | 1.00 (8/8) | 0.92 (49/53) | 0.02 | 0.96 | 75 s (CPU) |

- **Answer rate and refusal rate:** questions answered or refused correctly.
- **Citation accuracy:** answered questions that cite a gold citation unit.
- **Invalid citations in drafts:** first drafts that cited something not in the sources. These are repaired or stripped and never shown.

**Refusal triggers:**
- 8 low-evidence refusals, made with no LLM call: 7 unanswerable questions and 1 false refusal (t01).
- 2 model refusals: u04, a near-domain trap ("which models are on the systemic-risk list?") that passed the retrieval threshold, and h09.

**Remaining misses:**
- **Retrieval, gold unit not in the top 5:** h09, p03, dp03.
- **Threshold, gold retrieved but scored low:** t01.
- **Generation, gold in the sources but the 7B model cited a recital or a specific case instead:** e03, s04.

These are deterministic metrics. LLM-judged faithfulness and correctness come next.

```bash
uv run python -m aiact.evals.answers --provider fake --gate data/eval/answer_baseline.json   # replay (CI, free, ~5 s)
uv run python -m aiact.evals.answers --provider ollama --record --out data/eval/results      # re-record (~1 h CPU)
```
CI replays the recorded retrieval snapshot and LLM cassette. A prompt change without re-recording fails loudly as a cassette miss.
