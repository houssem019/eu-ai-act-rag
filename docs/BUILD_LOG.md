# Build Log

Chronological record of what was built, **why**, and which alternatives were rejected.
Read this before an interview to retell the story of the project.

| Step | What | Why / alternatives considered |
|------|------|-------------------------------|
| 1 | **Corpus acquisition** (`aiact.corpus.download`): the official English Official Journal XHTML of Regulation (EU) 2024/1689, from the Publications Office CELEX endpoint. Committed to `data/raw/` (1.26 MB). | EUR-Lex's HTML pages answer scripted requests with **HTTP 202 and an empty body** (bot challenge). The Publications Office serves the identical XHTML via content negotiation. The text is immutable, versioned law that EU reuse rules allow, so committing it makes parsing, tests and evals reproducible offline. Rejected: the PDF, where layout extraction loses the article, paragraph and point structure that citations need. |
| 2 | **Structure-aware parser** (`aiact.corpus.parse`) produces `data/corpus/ai_act.jsonl`: **306 provisions** (113 articles, 180 recitals, 13 annexes; 84.6k words) and **786 citable units**. Each article carries its chapter/section, title and numbered paragraphs, and each unit has a ready-made citation (`Article 5(1)`, `Article 3(44)`, `Recital 27`, `Annex III`). | Citations are the product, so the legal structure is kept from the start rather than guessed from chunks. Point labels (a), (b), (i) stay inline with their text on separate lines, so a chunk is quotable and citable down to `Article 5(1)(a)`. Article 3's 68 definitions become individual units, because that's how they're cited. |
