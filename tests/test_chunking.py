from collections import Counter

import pytest

from aiact.chunking import STRATEGIES, chunk, header, load_provisions, n_tokens


@pytest.fixture(scope="module")
def provisions():
    return load_provisions()


@pytest.fixture(scope="module")
def units(provisions):
    return [p["citation"] for prov in provisions for p in prov["paragraphs"]]


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_every_unit_is_retrievable(provisions, units, strategy):
    covered = {c for ch in chunk(strategy, provisions) for c in ch.citations}
    assert covered == set(units)


@pytest.mark.parametrize("strategy", ["article", "paragraph"])
def test_structured_chunks_fit_budget_and_stay_in_one_provision(provisions, strategy):
    for ch in chunk(strategy, provisions, max_tokens=400):
        assert n_tokens(ch.text) <= 400, ch.id
        assert ch.provision and all(c.startswith(ch.provision) for c in ch.citations), ch.id


def test_header_carries_title_and_chapter(provisions):
    art5 = next(p for p in provisions if p["id"] == "art_5")
    assert header(art5) == "Article 5 - Prohibited AI practices (Chapter II: PROHIBITED AI PRACTICES)"
    first = next(c for c in chunk("paragraph", [art5]))
    assert first.text.startswith(header(art5) + "\n1. The following AI practices shall be prohibited")


def test_paragraph_strategy_is_one_unit_per_chunk(provisions):
    for ch in chunk("paragraph", provisions):
        assert len(ch.citations) == 1


def test_article_strategy_packs_short_definitions(provisions):
    art3 = [c for c in chunk("article", provisions) if c.provision == "Article 3"]
    assert len(art3) < 68 / 3  # 68 definitions packed several per chunk
    assert "Article 3(1)" in art3[0].citations and "Article 3(2)" in art3[0].citations


def test_oversized_unit_is_split_but_keeps_its_citation(provisions):
    # Annex III is one citable unit far larger than the budget.
    anx3 = [c for c in chunk("paragraph", provisions) if c.provision == "Annex III"]
    assert len(anx3) > 1 and all(c.citations == ("Annex III",) for c in anx3)
    assert all(c.text.startswith("Annex III - High-risk AI systems") for c in anx3)


def test_fixed_windows_map_back_to_units_by_offset(provisions):
    chunks = chunk("fixed", provisions)
    assert all(n_tokens(c.text) <= 257 for c in chunks)  # decode/re-encode can merge at most one boundary
    assert not any(c.text.startswith("Article ") for c in chunks[:5])  # baseline has no headers
    text_of = {p["citation"]: p["text"] for prov in provisions for p in prov["paragraphs"]}
    for c in chunks[::25]:
        first = c.citations[0]
        # the window shares text with every unit it claims, checked on its first and last unit
        assert c.text[:40] in text_of[first] or text_of[first][-40:] in c.text, c.id
        assert c.text[-40:] in text_of[c.citations[-1]] or text_of[c.citations[-1]][:40] in c.text, c.id


def test_chunk_ids_unique(provisions):
    for s in STRATEGIES:
        ids = Counter(c.id for c in chunk(s, provisions))
        assert max(ids.values()) == 1


def test_unknown_strategy():
    with pytest.raises(ValueError):
        chunk("semantic", [])
