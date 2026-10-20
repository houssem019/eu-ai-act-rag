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
- **Fix (planned for milestone 3):** keep recitals as supporting context. Then either map each retrieved recital to the articles it explains, using a small recital → article table from the Act's own references, or give operative provisions a small rank prior before the answer step. Measure both on the same benchmark.
- **At scale:** resolve cross-references during parsing. For example, expand "Chapters I and II" into the chapter titles inside the chunk text. Then the operative text carries the words people actually search with.

## 7. Hybrid search can hurt top-1
- **Problem:** with paragraph chunks, adding BM25 through RRF dropped R@1 from 0.51 to 0.35 (bge-small) while R@5 went up.
- **Root cause:** RRF weights both lists equally by rank. When dense is already right at rank 1 and BM25's rank 1 is a keyword-heavy neighbour (another paragraph of the same article, or a recital), fusion can promote the BM25 pick.
- **Fix:** hybrid is used only for *candidate generation* (30 candidates). The cross-encoder decides the order, which gives the recall of hybrid and the precision of dense or better (R@1 0.56).
- **At scale:** weighted RRF or a learned fusion weight, tuned on a larger labelled set.
