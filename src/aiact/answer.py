"""Grounded answering with forced, verified citations.

    question -> rerank retrieval -> [refuse if evidence is weak] -> LLM over labelled sources
             -> parse [Article X(y)] citations -> verify each against the sources -> one repair turn if needed

Refusal has three independent triggers, cheapest first:
1. low_evidence: the best cross-encoder score is below the threshold (no LLM call, no cost).
2. model: the LLM answers INSUFFICIENT_EVIDENCE (topic is covered but the specific fact is not).
3. uncited: after one repair turn the answer still cites nothing that is in the sources.
A citation the sources don't contain is never shown to the user: it is stripped and counted.
"""

import re
import time
from collections.abc import Generator, Iterator
from dataclasses import asdict, dataclass, field

from aiact.llm.base import LLMClient, LLMResponse, Usage
from aiact.retrieve import Hit

REFUSAL_SENTINEL = "INSUFFICIENT_EVIDENCE"
REFUSAL_TEXT = (
    "I can't answer that from the EU AI Act: the provisions I found don't address it. "
    "Try rephrasing, or ask about a specific obligation, role or article."
)

SYSTEM_PROMPT = f"""You answer questions about Regulation (EU) 2024/1689, the EU AI Act, using only the sources supplied with each question. Each source is a provision of the Regulation, labelled with its citation.

Rules:
1. Use only what the sources state. Do not add outside knowledge or speculate.
2. Support every factual sentence with a citation in square brackets, written exactly as a source label, for example [Article 5(1)] or [Annex III]. You may narrow a citation to a point inside the labelled unit, for example [Article 5(1)(f)]. Separate several citations with semicolons: [Article 26(1); Article 26(2)].
3. Recitals explain the Regulation but are not binding. When an article or annex in the sources states the rule, cite that article or annex; cite a recital only in addition, or when no article or annex in the sources covers the point.
4. If the sources do not contain the answer, reply with exactly {REFUSAL_SENTINEL} and nothing else. Do this also when the sources discuss the topic but not the specific fact asked, such as a fee, a list of companies or a vote count that the Regulation does not state.
5. Answer in one to five short sentences of plain English. No preamble, headings or disclaimers."""

_BRACKET = re.compile(r"\[([^\[\]]{3,160})\]")
_CITATION = re.compile(r"^(Article \d+[a-z]?|Recital \d+|Annex [IVXLC]+)\b")


@dataclass
class Answer:
    question: str
    text: str
    citations: list[str]  # source units the answer cites (resolved, in order of first use)
    refused: bool
    refusal_reason: str | None  # low_evidence | model | uncited | safety
    top_score: float
    sources: list[dict]  # what the model saw: {"label", "citations", "score"}
    invalid_citations: list[str] = field(default_factory=list)  # cited but not in the sources (stripped)
    repaired: bool = False
    llm_calls: int = 0
    usage: Usage = field(default_factory=Usage)
    model: str = ""
    retrieval_ms: float = 0.0
    latency_ms: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def source_label(hit: Hit) -> str:
    return "; ".join(hit.chunk.citations)


def build_user_message(question: str, hits: list[Hit]) -> str:
    blocks = [f'<source label="{source_label(h)}">\n{h.chunk.text}\n</source>' for h in hits]
    return "Sources:\n" + "\n\n".join(blocks) + f"\n\nQuestion: {question}"


def resolve(cited: str, units: list[str]) -> str | None:
    """Map a cited string to the source unit it points into. "Article 5(1)(f)" -> "Article 5(1)"; a coarser
    "Article 5" -> its first unit in the sources. Prefixes must end on a boundary: "Article 50" != "Article 5"."""
    for u in units:
        if cited == u or (cited.startswith(u) and cited[len(u)] in "(, "):
            return u
    return next((u for u in units if u.startswith(cited + "(")), None)


def parse_citations(text: str, units: list[str]) -> tuple[list[str], list[str]]:
    """-> (resolved source units in first-use order, cited strings that are not in the sources)."""
    valid, invalid = [], []
    for group in _BRACKET.findall(text):
        for part in (p.strip() for p in group.split(";")):
            if not _CITATION.match(part):
                continue  # "[sic]", "[...]": not a citation
            unit = resolve(part, units)
            if unit is None:
                invalid.append(part)
            elif unit not in valid:
                valid.append(unit)
    return valid, invalid


def strip_invalid(text: str, units: list[str]) -> str:
    """Remove citations that point outside the sources, keeping valid ones in the same bracket."""

    def fix(m: re.Match) -> str:
        parts = [p.strip() for p in m.group(1).split(";")]
        if not any(_CITATION.match(p) for p in parts):
            return m.group(0)
        kept = [p for p in parts if _CITATION.match(p) and resolve(p, units)]
        return f"[{'; '.join(kept)}]" if kept else ""

    return re.sub(r"\s+([.,;])", r"\1", _BRACKET.sub(fix, text)).strip()


class Answerer:
    """`stream()` is the one code path (CLI, evals and the HTTP API all use it); `ask()` just consumes it.

    Events: {"type": "retrieval", ...}, then {"type": "delta", "text"} while the draft streams, {"type": "repair"}
    if a repair turn starts (a second draft streams after it), and always a last {"type": "final", "answer"}.
    Streamed deltas are a *draft*: the final answer is the verified one (invalid citations stripped, or a refusal),
    and clients replace the draft with it. The refusal sentinel is held back so users never see it stream.
    """

    def __init__(self, retriever, llm: LLMClient, threshold: float, top_k: int = 5):
        self.retriever, self.llm, self.threshold, self.top_k = retriever, llm, threshold, top_k

    def ask(self, question: str) -> Answer:
        for event in self.stream(question):
            if event["type"] == "final":
                return event["answer"]
        raise RuntimeError("stream ended without a final answer")

    def stream(self, question: str) -> Iterator[dict]:
        t0 = time.perf_counter()
        res = self.retriever.search(question, "rerank", k=self.top_k)
        hits, top = res.hits, res.top_score
        ans = Answer(
            question=question,
            text=REFUSAL_TEXT,
            citations=[],
            refused=True,
            refusal_reason="low_evidence",
            top_score=top,
            sources=[
                {"label": source_label(h), "citations": list(h.chunk.citations), "score": h.score}
                for h in hits
            ],
            model=self.llm.model,
            retrieval_ms=res.latency_ms,
        )
        yield {"type": "retrieval", "sources": ans.sources, "top_score": top, "retrieval_ms": res.latency_ms}
        if top < self.threshold:
            yield self._final(ans, t0)
            return

        units = [c for h in hits for c in h.chunk.citations]
        messages = [{"role": "user", "content": build_user_message(question, hits)}]
        resp = yield from self._generate(ans, messages)
        if resp.stop_reason == "refusal":
            ans.refusal_reason = "safety"
            yield self._final(ans, t0)
            return
        text = resp.text.strip()
        valid, invalid = parse_citations(text, units)
        if REFUSAL_SENTINEL not in text and (not valid or invalid):
            # One repair turn: tell the model exactly what was wrong and which labels exist.
            problem = (
                f"cited {', '.join(invalid)}, which is not among the sources"
                if invalid
                else "had no citations"
            )
            labels = "; ".join(dict.fromkeys(units))
            messages += [
                {"role": "assistant", "content": text},
                {
                    "role": "user",
                    "content": f"Your answer {problem}. Rewrite it following the rules, citing only these "
                    f"source labels: {labels}. If the sources do not support an answer, reply "
                    f"{REFUSAL_SENTINEL}.",
                },
            ]
            ans.repaired = True
            ans.invalid_citations = invalid
            yield {"type": "repair", "reason": problem}
            text = (yield from self._generate(ans, messages)).text.strip()
            valid, invalid = parse_citations(text, units)
            ans.invalid_citations += invalid

        if REFUSAL_SENTINEL in text:
            ans.refusal_reason = "model"
        elif not valid:
            ans.refusal_reason = "uncited"
        else:
            ans.refused, ans.refusal_reason = False, None
            ans.text, ans.citations = strip_invalid(text, units), valid
        yield self._final(ans, t0)

    def _generate(self, ans: Answer, messages: list[dict]) -> Generator[dict, None, LLMResponse]:
        """Stream one LLM call as delta events; return its LLMResponse. Text that could still turn out to be the
        refusal sentinel is buffered until it can't."""
        buffer, flushing, resp = "", False, None
        for item in self.llm.stream(SYSTEM_PROMPT, messages):
            if isinstance(item, LLMResponse):
                resp = item
                continue
            if flushing:
                yield {"type": "delta", "text": item}
                continue
            buffer += item
            if not REFUSAL_SENTINEL.startswith(buffer.lstrip()[: len(REFUSAL_SENTINEL)]):
                flushing = True
                yield {"type": "delta", "text": buffer}
        ans.llm_calls += 1
        ans.usage = ans.usage + resp.usage
        ans.model = resp.model
        return resp

    @staticmethod
    def _final(ans: Answer, t0: float) -> dict:
        ans.latency_ms = (time.perf_counter() - t0) * 1000
        return {"type": "final", "answer": ans}
