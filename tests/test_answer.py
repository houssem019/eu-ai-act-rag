"""Answering: prompt, citation parsing/verification, repair turn, and the three refusal triggers. Offline."""

import json

import httpx
import pytest

from aiact.answer import (
    REFUSAL_SENTINEL,
    REFUSAL_TEXT,
    SYSTEM_PROMPT,
    Answerer,
    build_user_message,
    parse_citations,
    resolve,
    strip_invalid,
)
from aiact.chunking import Chunk, expand_chapter_refs
from aiact.evals.answers import check_gate, score
from aiact.llm.base import LLMResponse, Usage
from aiact.llm.ollama_client import OllamaLLM
from aiact.llm.replay import CassetteMiss, ReplayLLM, ScriptedLLM
from aiact.retrieve import Hit, Result, SnapshotRetriever

HITS = [
    Hit(
        Chunk(
            "a",
            "Article 5 - Prohibited AI practices\n1. ...(c) social scoring...",
            ("Article 5(1)",),
            "Article 5",
        ),
        6.0,
    ),
    Hit(
        Chunk("b", "Article 99 - Penalties\n3. ... 35 000 000 EUR ...", ("Article 99(3)",), "Article 99"), 4.0
    ),
    Hit(Chunk("c", "Recital 31\nsocial scoring ...", ("Recital 31",), "Recital 31"), 7.0),
]
UNITS = ["Article 5(1)", "Article 99(3)", "Recital 31"]


class FixedRetriever:
    def __init__(self, hits):
        self.hits, self.calls = hits, []

    def search(self, query, mode="rerank", k=10):
        self.calls.append((query, mode, k))
        return Result(self.hits[:k], 12.0)


def answerer(replies, hits=HITS, threshold=2.0):
    return Answerer(FixedRetriever(hits), ScriptedLLM(replies), threshold=threshold, top_k=5)


# --------------------------------------------------------------------------- citations
@pytest.mark.parametrize(
    ("cited", "unit"),
    [
        ("Article 5(1)", "Article 5(1)"),
        ("Article 5(1)(c)", "Article 5(1)"),  # narrowed to a point inside the unit
        ("Article 5", "Article 5(1)"),  # coarser: resolves to its unit in the sources
        ("Article 50(1)", None),  # prefix of a different article
        ("Article 5(10)", None),
        ("Annex III", None),
    ],
)
def test_resolve(cited, unit):
    assert resolve(cited, UNITS) == unit


def test_parse_citations_handles_groups_and_ignores_non_citations():
    text = "Banned [Article 5(1)(c); Recital 31]. Fines apply [Article 99(3)] [sic]. Also [Article 6(2)]."
    valid, invalid = parse_citations(text, UNITS)
    assert valid == ["Article 5(1)", "Recital 31", "Article 99(3)"]
    assert invalid == ["Article 6(2)"]


def test_strip_invalid_keeps_valid_parts():
    text = "Banned [Article 5(1); Article 6(2)]. High-risk [Article 6(2)]. See [sic]."
    assert strip_invalid(text, UNITS) == "Banned [Article 5(1)]. High-risk. See [sic]."


# --------------------------------------------------------------------------- prompt
def test_user_message_labels_sources_and_ends_with_question():
    msg = build_user_message("Is social scoring banned?", HITS[:2])
    assert '<source label="Article 5(1)">\nArticle 5 - Prohibited AI practices' in msg
    assert msg.endswith("Question: Is social scoring banned?")
    assert REFUSAL_SENTINEL in SYSTEM_PROMPT and "Recitals" in SYSTEM_PROMPT


def test_chapter_refs_are_spelled_out():
    titles = {"I": "GENERAL PROVISIONS", "II": "PROHIBITED AI PRACTICES", "V": "GENERAL-PURPOSE AI MODELS"}
    out = expand_chapter_refs(
        "(a) Chapters I and II shall apply; Chapter V and Article 78; Chapter IX", titles
    )
    assert out == (
        "(a) Chapters I (GENERAL PROVISIONS) and II (PROHIBITED AI PRACTICES) shall apply; "
        "Chapter V (GENERAL-PURPOSE AI MODELS) and Article 78; Chapter IX"
    )


# --------------------------------------------------------------------------- answer flow
def test_grounded_answer():
    a = answerer(["Social scoring is prohibited [Article 5(1)(c)], with fines [Article 99(3)]."]).ask("q")
    assert not a.refused and a.citations == ["Article 5(1)", "Article 99(3)"]
    assert a.llm_calls == 1 and not a.repaired and a.top_score == 7.0  # raw max, not first hit
    assert [s["label"] for s in a.sources] == UNITS


def test_low_evidence_refuses_without_calling_the_llm():
    llm = ScriptedLLM([])
    a = Answerer(FixedRetriever(HITS), llm, threshold=10.0).ask("capital of France?")
    assert a.refused and a.refusal_reason == "low_evidence" and a.text == REFUSAL_TEXT
    assert llm.calls == [] and a.usage.input_tokens == 0


def test_model_refusal():
    a = answerer([REFUSAL_SENTINEL]).ask("How much does registration cost?")
    assert a.refused and a.refusal_reason == "model" and a.text == REFUSAL_TEXT


def test_uncited_answer_is_repaired_once():
    llm = ScriptedLLM(["Social scoring is banned.", "Social scoring is banned [Article 5(1)]."])
    a = Answerer(FixedRetriever(HITS), llm, threshold=2.0).ask("q")
    assert not a.refused and a.repaired and a.llm_calls == 2 and a.citations == ["Article 5(1)"]
    feedback = llm.calls[1]["messages"][-1]["content"]
    assert "had no citations" in feedback and "Article 5(1); Article 99(3); Recital 31" in feedback
    assert a.usage.input_tokens == 2000  # both calls are billed


def test_hallucinated_citation_is_repaired_and_never_shown():
    llm = ScriptedLLM(
        ["Banned [Article 5(1); Article 6(2)].", "Banned [Article 5(1)]; see also [Article 6(2)]."]
    )
    a = Answerer(FixedRetriever(HITS), llm, threshold=2.0).ask("q")
    assert "cited Article 6(2), which is not among the sources" in llm.calls[1]["messages"][-1]["content"]
    assert a.invalid_citations == ["Article 6(2)", "Article 6(2)"]
    assert "6(2)" not in a.text and a.citations == ["Article 5(1)"]


def test_still_uncited_after_repair_refuses():
    a = answerer(["It is banned.", "It is definitely banned."]).ask("q")
    assert a.refused and a.refusal_reason == "uncited" and a.llm_calls == 2


def test_safety_refusal_stop_reason():
    class Refuser:
        model = "m"

        def complete(self, system, messages):
            return LLMResponse("", "refusal", "m", Usage(10, 0))

    a = Answerer(FixedRetriever(HITS), Refuser(), threshold=2.0).ask("q")
    assert a.refused and a.refusal_reason == "safety"


# --------------------------------------------------------------------------- record / replay
def test_llm_cassette_records_then_replays(tmp_path):
    path = tmp_path / "c.json"
    rec = ReplayLLM(path, inner=ScriptedLLM(["Banned [Article 5(1)]."], model="qwen"))
    first = Answerer(FixedRetriever(HITS), rec, threshold=2.0).ask("q")
    replay = Answerer(FixedRetriever(HITS), ReplayLLM(path), threshold=2.0)
    again = replay.ask("q")
    assert again.text == first.text and again.model == "qwen"
    with pytest.raises(CassetteMiss):
        replay.ask("a different question")  # different prompt -> loud failure, not a stale answer


def test_retrieval_snapshot_records_then_replays(tmp_path):
    path = tmp_path / "r.json"
    inner = FixedRetriever(HITS)
    rec = SnapshotRetriever(path, inner=inner).search("q", "rerank", 2)
    rep = SnapshotRetriever(path).search("q", "rerank", 2)
    assert [h.chunk for h in rep.hits] == [h.chunk for h in rec.hits] and rep.top_score == 6.0
    assert len(inner.calls) == 1
    with pytest.raises(KeyError):
        SnapshotRetriever(path).search("other", "rerank", 2)


def test_ollama_client_sends_system_prompt_and_context_size():
    seen = {}

    def handler(request):
        seen.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "message": {"content": "ok"},
                "done_reason": "stop",
                "prompt_eval_count": 7,
                "eval_count": 2,
            },
        )

    llm = OllamaLLM("qwen2.5:7b", "http://ollama")
    llm.http = httpx.Client(base_url="http://ollama", transport=httpx.MockTransport(handler))
    r = llm.complete("SYS", [{"role": "user", "content": "hi"}])
    assert r.text == "ok" and r.usage.input_tokens == 7
    assert seen["messages"][0] == {"role": "system", "content": "SYS"}
    assert seen["options"] == {"temperature": 0, "num_ctx": 8192}


# --------------------------------------------------------------------------- eval scoring
def test_score_and_gate():
    qs = [
        {"id": "a1", "answerable": True, "gold": ["Article 5(1)"]},
        {"id": "a2", "answerable": True, "gold": ["Article 26(6)"]},
        {"id": "u1", "answerable": False, "gold": []},
    ]
    good = answerer(["Banned [Article 5(1)]."]).ask("a1")
    wrong = answerer(["Fines [Recital 31]."]).ask("a2")
    refused = Answerer(FixedRetriever(HITS), ScriptedLLM([]), threshold=99).ask("u1")
    m = score(qs, [good, wrong, refused])["metrics"]
    assert m["answer_rate"] == 1.0 and m["refusal_rate"] == 1.0
    assert m["citation_accuracy"] == 0.5 and m["article_citation_rate"] == 0.5
    assert m["refusal_reasons"]["low_evidence"] == 1
    assert check_gate(m, {"min": {"citation_accuracy": 0.6}, "max": {"invalid_citation_rate": 0.1}}) == [
        "FAIL citation_accuracy=0.500 < baseline 0.6"
    ]
