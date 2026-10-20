"""Turn parsed provisions into retrieval chunks. Three strategies, benchmarked against each other in bench.py.

- article:   one provision per chunk (paragraphs packed up to max_tokens), with an "Article N - title" header.
- paragraph: one citable unit per chunk, with the same header so the embedding knows where it sits.
             Both structured strategies also spell out chapter cross-references (expand_chapter_refs).
- fixed:    naive baseline - fixed token windows over the concatenated text, no headers; citations are
             recovered afterwards from character-offset overlap with the citable units.

Every chunk carries the citation units it contains, which is what retrieval is scored on.
"""

import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import tiktoken

from aiact.config import settings

STRATEGIES = ("article", "paragraph", "fixed")


@dataclass(frozen=True)
class Chunk:
    id: str
    text: str  # what gets embedded and shown to the LLM
    citations: tuple[str, ...]  # citable units inside this chunk, in document order
    provision: str  # "Article 5", "Recital 12", "Annex III"; "" when a fixed window spans several


@cache
def _enc() -> tiktoken.Encoding:
    return tiktoken.get_encoding("cl100k_base")


def n_tokens(text: str) -> int:
    return len(_enc().encode(text))


def load_provisions(path: Path | None = None) -> list[dict]:
    path = path or settings.corpus_path
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def header(prov: dict) -> str:
    """'Article 5 - Prohibited AI practices (Chapter II: PROHIBITED AI PRACTICES)'."""
    h = prov["citation"] + (f" - {prov['title']}" if prov["title"] else "")
    ctx = [f"Chapter {prov['chapter']}: {prov['chapter_title']}"] if prov.get("chapter") else []
    if prov.get("section"):
        ctx.append(f"Section {prov['section']}: {prov['section_title']}")
    return h + (f" ({'; '.join(ctx)})" if ctx else "")


_CHAPTER_REF = re.compile(r"\bChapters? ([IVX]+(?:(?:,| and| or) [IVX]+)*)\b")


def expand_chapter_refs(text: str, titles: dict[str, str]) -> str:
    """'Chapters I and II shall apply' -> 'Chapters I (GENERAL PROVISIONS) and II (PROHIBITED AI PRACTICES)
    shall apply'. Operative articles refer to chapters by number, which no embedder or BM25 can connect to a
    question about "prohibitions"; the titles put the searchable words into the chunk. Citations are unchanged."""

    def numerals(m: re.Match) -> str:
        expanded = re.sub(
            r"\b([IVX]+)\b", lambda n: f"{n[1]} ({titles[n[1]]})" if n[1] in titles else n[1], m.group(1)
        )
        return m.group(0).replace(m.group(1), expanded)

    return _CHAPTER_REF.sub(numerals, text)


def _split_long(text: str, budget: int) -> list[str]:
    """Split text that exceeds the token budget on line boundaries; hard-split single over-long lines."""
    pieces, cur = [], []
    for line in text.split("\n"):
        if n_tokens(line) > budget:  # rare: one enormous line
            if cur:
                pieces.append("\n".join(cur))
                cur = []
            toks = _enc().encode(line)
            pieces += [_enc().decode(toks[i : i + budget]) for i in range(0, len(toks), budget)]
            continue
        if cur and n_tokens("\n".join([*cur, line])) > budget:
            pieces.append("\n".join(cur))
            cur = []
        cur.append(line)
    if cur:
        pieces.append("\n".join(cur))
    return pieces


def _structured(provisions: Iterable[dict], max_tokens: int, pack: bool) -> list[Chunk]:
    provisions = list(provisions)
    titles = {p["chapter"]: p["chapter_title"] for p in provisions if p.get("chapter")}
    chunks: list[Chunk] = []
    for prov in provisions:
        head = header(prov)
        budget = max_tokens - n_tokens(head) - 1
        # groups of (text, citations): one per unit, or packed consecutive units when pack=True
        groups: list[tuple[str, list[str]]] = []
        for para in prov["paragraphs"]:
            for piece in _split_long(expand_chapter_refs(para["text"], titles), budget):
                last = groups[-1] if groups else None
                if pack and last and n_tokens(last[0] + "\n" + piece) <= budget:
                    groups[-1] = (last[0] + "\n" + piece, last[1] + [para["citation"]])
                else:
                    groups.append((piece, [para["citation"]]))
        for i, (body, cits) in enumerate(groups):
            chunks.append(
                Chunk(f"{prov['id']}#{i}", f"{head}\n{body}", tuple(dict.fromkeys(cits)), prov["citation"])
            )
    return chunks


def _fixed(provisions: Iterable[dict], window: int, overlap: int) -> list[Chunk]:
    # Concatenate the raw text and remember each unit's character span.
    parts, spans, pos = [], [], 0
    for prov in provisions:
        for para in prov["paragraphs"]:
            parts.append(para["text"])
            spans.append((pos, pos + len(para["text"]), para["citation"], prov["citation"]))
            pos += len(para["text"]) + 2
    doc = "\n\n".join(parts)
    toks = _enc().encode(doc)
    _, offsets = _enc().decode_with_offsets(toks)
    offsets.append(len(doc))

    chunks, step = [], window - overlap
    for n, start in enumerate(range(0, len(toks), step)):
        end = min(start + window, len(toks))
        c0, c1 = offsets[start], offsets[end]
        hit = [s for s in spans if s[0] < c1 and s[1] > c0]
        provs = {s[3] for s in hit}
        chunks.append(
            Chunk(
                f"fixed#{n}",
                _enc().decode(toks[start:end]),
                tuple(dict.fromkeys(s[2] for s in hit)),
                provs.pop() if len(provs) == 1 else "",
            )
        )
        if end == len(toks):
            break
    return chunks


def chunk(strategy: str, provisions: list[dict] | None = None, max_tokens: int = 400) -> list[Chunk]:
    provisions = provisions if provisions is not None else load_provisions()
    if strategy == "article":
        return _structured(provisions, max_tokens, pack=True)
    if strategy == "paragraph":
        return _structured(provisions, max_tokens, pack=False)
    if strategy == "fixed":
        return _fixed(provisions, window=256, overlap=32)
    raise ValueError(f"unknown strategy {strategy!r}; choose from {STRATEGIES}")


if __name__ == "__main__":
    provs = load_provisions()
    for s in STRATEGIES:
        cs = chunk(s, provs)
        toks = sorted(n_tokens(c.text) for c in cs)
        cits = sum(len(c.citations) for c in cs) / len(cs)
        print(
            f"{s:10s} {len(cs):5d} chunks  tokens p50={toks[len(toks) // 2]} max={toks[-1]}  cits/chunk={cits:.2f}"
        )
