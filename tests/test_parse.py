import json

import pytest

from aiact.config import settings
from aiact.corpus.parse import parse


@pytest.fixture(scope="module")
def provisions():
    return {p.id: p for p in parse(settings.raw_html.read_text(encoding="utf-8"))}


def test_complete_regulation(provisions):
    kinds = [p.kind for p in provisions.values()]
    assert kinds.count("article") == 113
    assert kinds.count("recital") == 180
    assert kinds.count("annex") == 13
    assert all(p.text.strip() for p in provisions.values())


def test_article_structure_and_citations(provisions):
    art5 = provisions["art_5"]
    assert art5.title == "Prohibited AI practices" and art5.chapter == "II"
    assert art5.paragraphs[0].citation == "Article 5(1)"
    assert "(a) the placing on the market" in art5.paragraphs[0].text  # point label kept with its text

    art9 = provisions["art_9"]
    assert (art9.chapter, art9.section) == ("III", "2")
    assert art9.section_title == "Requirements for high-risk AI systems"


def test_definitions_are_individually_citable(provisions):
    defs = {p.citation: p.text for p in provisions["art_3"].paragraphs}
    assert len(defs) == 69  # intro + 68 definitions
    assert defs["Article 3(1)"].startswith("(1) ‘AI system’ means")
    assert "publicly accessible space" in defs["Article 3(44)"]


def test_unnumbered_article_keeps_point_labels(provisions):
    text = provisions["art_113"].text
    assert "It shall apply from 2 August 2026." in text
    assert "(a) Chapters I and II shall apply from 2 February 2025;" in text


def test_annex_nesting(provisions):
    anx = provisions["anx_III"]
    assert anx.title == "High-risk AI systems referred to in Article 6(2)"
    assert "1. Biometrics" in anx.text and "  (a) remote biometric identification systems." in anx.text


def test_committed_corpus_matches_parser(provisions):
    lines = settings.corpus_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == len(provisions)
    first = json.loads(lines[0])
    assert first["text"] == provisions[first["id"]].text


def test_exponents_kept_and_footnote_markers_dropped(provisions):
    assert "greater than 10^25" in provisions["art_51"].paragraphs[1].text
    assert all("( 1 )" not in p.text for p in provisions.values())
