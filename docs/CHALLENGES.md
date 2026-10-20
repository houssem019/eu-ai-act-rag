# Challenges & Lessons

Real problems hit while building, how they were solved, and known limitations.
Format: **Problem → Root cause → Fix → What I'd do at scale.**

## 1. EUR-Lex blocks scripted downloads
- **Problem:** `curl` to `eur-lex.europa.eu/legal-content/...HTML` returned **HTTP 202 with 0 bytes**, an anti-bot challenge. The ELI URL redirected to the same thing.
- **Fix:** the EU Publications Office "cellar" serves the same Official Journal XHTML by content negotiation on `publications.europa.eu/resource/celex/32024R1689`. The downloader checks that the response actually contains `id="art_1"` before saving.
- **Lesson:** for regulatory corpora, prefer the official machine-readable channel (cellar, ELI, Formex XML) over scraping the human-facing site.

## 2. Points split from their labels in unnumbered articles
- **Problem:** first-pass output for Article 3 (Definitions) and Article 113 had "(1)" on one line and "'AI system' means…" on the next. The renderer was walking a table's children (`col`, `tbody`) instead of the table itself.
- **Fix:** render a copy of the article with headings removed, using the same table-aware renderer as numbered paragraphs. Tests pin it: `(1) 'AI system' means` must be one line.
- **Why it matters:** a chunk that starts with an orphan "(1)" can't be cited or understood. Retrieval quality is decided at parse time.

## 3. Chunk budget in one tokenizer, truncation in another
- **Problem:** chunks are budgeted with tiktoken (`cl100k_base`). BGE truncates at **512 of its own WordPiece tokens**, and does it silently. Legal text with many numbers and references tokenizes differently in the two.
- **Fix:** measured it with BGE's own `tokenizer.json`. A 400-token cl100k budget becomes **at most 435 WordPiece tokens**, so no chunk in any strategy is truncated. The check is recorded here so the budget isn't raised blindly.
- **At scale:** budget with the embedder's tokenizer directly, and assert on it at index time.

## 4. fastembed doesn't add the BGE query instruction
- **Problem:** BGE v1.5 is trained with the query prefix `"Represent this sentence for searching relevant passages: "` for short queries. fastembed's `query_embed` for BGE embeds the raw text.
- **Fix:** `FastEmbedDense.embed_query` adds the prefix. Documents are embedded without it, which is the asymmetric setup the model card describes.

## 5. The cross-encoder dominates latency on CPU
- **Problem:** the first full benchmark run was on pace for over an hour. Profiling one query showed dense search at about 0.4 s (mostly query embedding) and hybrid RRF at about 0.02 s, but reranking 30 candidates of up to 400 tokens with MiniLM-L6 took **2–6 s**. ONNX `threads` and batch size made no improvement.
- **Fix:** the benchmark reranks each question once and reuses that pass for both recall and the refusal-threshold scores. That halves rerank calls.
- **At scale:** rerank fewer candidates (top 20), truncate candidates to the passage around the BM25 hit, run the reranker on a GPU or a quantized ONNX export, or cache scores for repeated (query, chunk) pairs together with the semantic cache.

## 6. Recitals outrank the operative articles
- **Problem:** four questions are missed in every configuration. The two timeline questions (gold `Article 113`) are always beaten by **Recital 179**. Recital 179 explains the application dates in plain words ("the prohibitions should apply already from 2 February 2025"), while Article 113 only says "Chapters I and II shall apply from 2 February 2025". The employee-emotion question surfaces Recital 44 and the Article 3(39) definition rather than the prohibition in Article 5(1)(f).
- **Root cause:** recitals are written to *explain*, so their wording is closer to how people ask. Articles are written by cross-reference ("Chapters I and II", "point (h) of paragraph 1"), which no embedder can resolve. The retrieval isn't wrong, since the recital is relevant. It's a **citation-policy** problem: recitals aren't binding, so an answer must cite the article.
- **Diagnosis:** a dump of all 30 reranked candidates per question showed **Article 113 was never among the candidates**. This is a vocabulary gap: no embedder or BM25 can link "prohibitions" to "Chapters I and II". The ordering problem is separate: recitals took 1.8 of the top-5 slots on average.
- **Fix, two measured steps:**
  1. **Cross-reference expansion at chunk time.** "Chapters I (GENERAL PROVISIONS) and II (PROHIBITED AI PRACTICES)" goes into the embedded text; citations are unchanged. This fixed both timeline questions: MRR 0.712 → 0.739, R@5 0.89 → 0.93.
  2. **Recital rank prior** (−2.0 on the rerank logit), chosen offline from the saved candidate scores: MRR → **0.834**, R@1 0.56 → **0.73**, R@5 → **0.95**. Refusal keeps using the raw top score, so the threshold isn't distorted.
- **Also in the prompt:** "cite the article or annex; cite a recital only in addition".
- **Still open:** p03 (employee emotions). Article 5(1)(f) says "infer emotions … in the areas of workplace", and the question says "recognise emotions of employees at work". The gold chunk is outside the top 30.
- **At scale:**
  - resolve *all* internal cross-references, not just chapters: "point (h) of paragraph 1", "referred to in Article 6(2)";
  - add a recital → article map from the Act's own references;
  - add query rewriting into the Act's vocabulary;
  - tune the prior on a held-out set. Mine was tuned in-sample.

## 7. Hybrid search can hurt top-1
- **Problem:** with paragraph chunks, adding BM25 through RRF dropped R@1 from 0.51 to 0.35 (bge-small) while R@5 went up.
- **Root cause:** RRF weights both lists equally by rank. When dense is already right at rank 1 and BM25's rank 1 is a keyword-heavy neighbour (another paragraph of the same article, or a recital), fusion can promote the BM25 pick.
- **Fix:** hybrid is used only for *candidate generation* (30 candidates). The cross-encoder decides the order, which gives the recall of hybrid and the precision of dense or better (R@1 0.56).
- **At scale:** weighted RRF or a learned fusion weight, tuned on a larger labelled set.

## 8. A refusal threshold sitting on one data point
- **Problem:** after the retrieval fixes, the auto-tuned threshold landed at 1.64. That's *exactly* the score of one answerable question (h09). A last-digit float change on another machine would have flipped that question to a refusal.
- **Fix:** max-margin cut. Take the best balanced-accuracy split, then put the threshold midway between its boundary scores. The highest refused unanswerable question scores 1.11 and the lowest answered answerable one scores 1.64, so the threshold is **1.37**.
- **Lesson:** with 8 negatives, a threshold is an operating point, not a measurement. Report it with its margin, and measure refusal end to end in the answer eval.

## 9. Record/replay must not depend on float-exact retrieval
- **Problem:** the LLM cassette is keyed on the full prompt, sources included. Replaying with live retrieval in CI would make the key depend on the exact ranking. A last-digit ONNX difference between Windows (recording) and Linux (CI) can swap two near-tied sources, change the prompt and cause a cassette miss.
- **Fix:** record a **retrieval snapshot** next to the LLM cassette (`SnapshotRetriever`), and replay both in the answer gate. Retrieval quality has its own gate on real models (`aiact.bench --gate`), so each gate tests one thing.
- **At scale:** the same split. Retrieval evals on a pinned index version; generation evals on frozen contexts. That's also what makes prompt A/B tests comparable.

## 10. fastembed calls the Hugging Face hub on every model load
- **Problem:** even with models cached, each load logs "Fetching…" and checks the hub. A container with no network would stall or fail, and a hub outage would take the service down at startup.
- **Fix:** bake the models into the image at build time (`aiact.warmup`), then set `HF_HUB_OFFLINE=1`. CI runs the image with `--network none` to prove it.

## 11. Small local models and Ollama's context window
- **Risk, prevented rather than hit:** Ollama's default context is 2–4k tokens and it truncates from the *front*. Five sources plus the instructions are about 1.5–2.5k tokens, so a longer question could silently lose the system prompt, including the refusal and citation rules.
- **Fix:** `num_ctx=8192` is sent on every request, and a unit test pins it.

## 12. Where answers still fail: attributing each miss to a stage
- **Method:** for every answerable question that was refused or cited no gold unit, check whether the gold unit was among the 5 sources the model saw. That separates retrieval failures from generation failures, and the two need different fixes.
- **Retrieval (3), gold not in the top 5:**
  - **h09:** the 2-person verification rule in Article 14(5). The model correctly said `INSUFFICIENT_EVIDENCE`.
  - **p03:** employee emotion recognition, Article 5(1)(f). The model answered correctly in substance from Recital 44.
  - **dp03:** the deployer FRIA scope, Article 27(1). It answered from Article 27(3).
  - The model behaved correctly given what it saw. The fix belongs in retrieval: query rewriting into the Act's vocabulary, and splitting the huge Article 5(1) unit by point so (f) gets its own embedding.
- **Threshold (1), t01, "do chatbots have to tell users…":** Article 50(1) *was* retrieved, but MiniLM scored "chatbots" against "AI systems intended to interact directly with natural persons" at −2.08, the lowest score of any answerable question. A general-domain cross-encoder doesn't know that a chatbot is such a system. Fix: a stronger or legal-tuned reranker (bge-reranker-base), or lowering the threshold, which would trade away refusal of u03.
- **Generation (2), gold in the sources:**
  - **e03:** cited Recital 168 although Article 99(6) was the top source. The 7B model ignored the "prefer articles" rule.
  - **s04:** cited the GPAI and high-risk special cases (Articles 54 and 22) instead of the general scope rule, Article 2(1)(a).
  - Fix: a stronger model (re-record with Sonnet). Or enforce it in code: when an answer cites only recitals while an article is in the sources, use the repair turn to ask for the article.
- **Lesson:** a single "accuracy" number hides which component to fix. Stage-attributed failure analysis is cheap, because the snapshot already stores what the model saw.
