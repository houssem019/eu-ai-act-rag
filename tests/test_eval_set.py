"""The curated eval set is only as good as its gold labels: verify them against the Official Journal text."""

import json

import pytest
import yaml

from aiact.config import PROJECT_ROOT, settings

QUESTIONS = yaml.safe_load((PROJECT_ROOT / "data" / "eval" / "questions.yaml").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def units() -> dict[str, str]:
    out = {}
    for line in settings.corpus_path.read_text(encoding="utf-8").splitlines():
        prov = json.loads(line)
        for p in prov["paragraphs"]:
            out[p["citation"]] = p["text"]
    return out


def test_ids_unique_and_balanced():
    ids = [q["id"] for q in QUESTIONS]
    assert len(ids) == len(set(ids))
    assert sum(q["answerable"] for q in QUESTIONS) >= 45
    assert sum(not q["answerable"] for q in QUESTIONS) >= 6


def test_every_gold_citation_exists(units):
    for q in QUESTIONS:
        assert bool(q["gold"]) == q["answerable"], q["id"]
        for c in q["gold"]:
            assert c in units, f"{q['id']}: {c} not in corpus"


# Key facts in the reference answers, checked against the cited text (catches wrong paragraph numbers).
FACTS = [
    ("Article 3(60)", "deep fake"),
    ("Article 3(49)", "serious incident"),
    ("Article 5(1)", "social behaviour"),
    ("Article 5(1)", "emotions of a natural person in the areas of workplace"),
    ("Article 5(3)", "24 hours"),
    ("Article 2(3)", "military, defence or national security"),
    ("Article 2(6)", "sole purpose of scientific research"),
    ("Article 2(12)", "free and open-source"),
    ("Article 6(3)", "profiling of natural persons"),
    ("Annex III", "creditworthiness"),
    ("Annex III", "analyse and filter job applications"),
    ("Article 14(5)", "at least two natural persons"),
    ("Article 22(1)", "authorised representative"),
    ("Article 26(6)", "at least six months"),
    ("Article 27(1)", "points 5 (b) and (c) of Annex III"),
    ("Article 43(2)", "does not provide for the involvement of a notified body"),
    ("Article 50(4)", "deep fake"),
    ("Article 51(2)", "10^25"),
    ("Article 52(1)", "within two weeks"),
    ("Article 53(1)", "copyright"),
    ("Article 57(1)", "2 August 2026"),
    ("Article 73(2)", "15 days"),
    ("Article 73(4)", "10 days"),
    ("Article 99(3)", "35 000 000"),
    ("Article 99(5)", "7 500 000"),
    ("Article 101(1)", "15 000 000"),
    ("Article 111(3)", "2 August 2027"),
    ("Article 113", "2 February 2025"),
]


@pytest.mark.parametrize(("citation", "phrase"), FACTS)
def test_reference_facts_are_in_cited_text(units, citation, phrase):
    assert phrase in units[citation]
