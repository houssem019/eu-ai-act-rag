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
