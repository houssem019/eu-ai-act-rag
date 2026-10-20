"""Parse the Official Journal XHTML of the AI Act into citable provisions.

Output: one JSON line per provision (113 articles, 180 recitals, 13 annexes), keeping the legal
structure that citations need:

    {"id": "art_5", "kind": "article", "number": "5", "citation": "Article 5",
     "title": "Prohibited AI practices", "chapter": "II", "chapter_title": "PROHIBITED AI PRACTICES",
     "section": null, "section_title": null,
     "paragraphs": [{"number": "1", "citation": "Article 5(1)", "text": "1. The following ... (a) ..."}],
     "text": "<full provision text>"}

Points such as (a), (b) stay inside their paragraph, written on their own lines with their labels, so
any chunk built from a paragraph is still quotable and citable as "Article 5(1)(a)".
"""

from __future__ import annotations

import argparse
import json
import re
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup, Tag, XMLParsedAsHTMLWarning

from aiact.config import settings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)  # OJ serves XHTML; lxml handles it
_WS = re.compile(r"[ \t\u00a0\u2009\u202f]+")


def _clean(text: str) -> str:
    return re.sub(r"(\d) \^(\d)", r"\1^\2", _WS.sub(" ", text)).strip()


@dataclass
class Paragraph:
    number: str | None  # "1", "2"... ; None for unnumbered text
    citation: str
    text: str


@dataclass
class Provision:
    id: str
    kind: str  # article | recital | annex
    number: str
    citation: str
    title: str
    chapter: str | None = None
    chapter_title: str | None = None
    section: str | None = None
    section_title: str | None = None
    paragraphs: list[Paragraph] = field(default_factory=list)
    text: str = ""


def render(el: Tag, depth: int = 0) -> list[str]:
    """Flatten an element to lines, keeping (a)/(i)/1. labels next to their text, nested tables indented."""
    lines: list[str] = []
    for child in el.children:
        if not isinstance(child, Tag):
            continue
        if child.name == "table":
            for tr in child.find_all("tr", recursive=False) or child.select(":scope > tbody > tr"):
                cells = tr.find_all("td", recursive=False)
                if len(cells) == 2:
                    label = _clean(cells[0].get_text(" "))
                    body = render(cells[1], depth + 1)
                    if body:
                        lines.append("  " * depth + f"{label} {body[0].strip()}")
                        lines.extend(body[1:])
                else:
                    lines.extend(render(tr, depth))
        elif child.name == "p":
            text = _clean(child.get_text(" "))
            if text:
                lines.append("  " * depth + text)
        else:
            lines.extend(render(child, depth))
    return lines


def _title_of(soup: BeautifulSoup, container_id: str) -> str | None:
    t = soup.find(id=f"{container_id}.tit_1")
    return _clean(t.get_text(" ")) if t else None


def _split_numbered_points(article: str, lines: list[str]) -> list[Paragraph]:
    """Unnumbered articles made of top-level '(1) ...', '(2) ...' items (Article 3's 68 definitions) are
    cited per item, e.g. 'Article 3(44)', so each item becomes its own paragraph."""
    top = [i for i, line in enumerate(lines) if re.match(r"^\(\d+\) ", line)]
    if len(top) < 3:
        return [Paragraph(None, f"Article {article}", "\n".join(lines))]
    out = []
    if top[0] > 0:
        out.append(Paragraph(None, f"Article {article}", "\n".join(lines[: top[0]])))
    for start, end in zip(top, top[1:] + [len(lines)], strict=True):
        num = re.match(r"^\((\d+)\)", lines[start]).group(1)
        out.append(Paragraph(num, f"Article {article}({num})", "\n".join(lines[start:end])))
    return out


def parse_articles(soup: BeautifulSoup) -> list[Provision]:
    out = []
    for div in soup.find_all("div", id=re.compile(r"^art_\d+$")):
        number = div["id"].split("_")[1]
        title = _title_of(soup, div["id"]) or ""
        chapter = section = None
        for parent in div.parents:
            pid = parent.get("id", "") if isinstance(parent, Tag) else ""
            if re.fullmatch(r"cpt_[IVXL]+\.sct_\d+", pid) and section is None:
                section = pid
            elif re.fullmatch(r"cpt_[IVXL]+", pid):
                chapter = pid
                break
        paragraphs = []
        numbered = div.find_all("div", id=re.compile(r"^\d{3}\.\d{3}$"), recursive=False)
        if numbered:
            for pdiv in numbered:
                pnum = str(int(pdiv["id"].split(".")[1]))
                paragraphs.append(Paragraph(pnum, f"Article {number}({pnum})", "\n".join(render(pdiv))))
        else:  # single unnumbered body (e.g. Article 3 definitions): render a copy without the headings
            body = BeautifulSoup(str(div), "lxml").find("div", id=div["id"])
            for heading in body.find_all(class_="oj-ti-art") + body.find_all(id=re.compile(r"\.tit_1$")):
                heading.decompose()
            paragraphs.extend(_split_numbered_points(number, render(body)))
        out.append(
            Provision(
                id=div["id"],
                kind="article",
                number=number,
                citation=f"Article {number}",
                title=title,
                chapter=chapter.split("_")[1] if chapter else None,
                chapter_title=_title_of(soup, chapter) if chapter else None,
                section=section.split("sct_")[1] if section else None,
                section_title=_title_of(soup, section) if section else None,
                paragraphs=paragraphs,
                text="\n".join(p.text for p in paragraphs),
            )
        )
    return out


def parse_recitals(soup: BeautifulSoup) -> list[Provision]:
    out = []
    for div in soup.find_all("div", id=re.compile(r"^rct_\d+$")):
        number = div["id"].split("_")[1]
        lines = render(div)
        text = re.sub(r"^\(\d+\)\s*", "", "\n".join(lines))
        out.append(
            Provision(
                id=div["id"],
                kind="recital",
                number=number,
                citation=f"Recital {number}",
                title="",
                paragraphs=[Paragraph(None, f"Recital {number}", text)],
                text=text,
            )
        )
    return out


def parse_annexes(soup: BeautifulSoup) -> list[Provision]:
    out = []
    for div in soup.find_all("div", id=re.compile(r"^anx_[IVXL]+$")):
        number = div["id"].split("_")[1]
        heads = [_clean(p.get_text(" ")) for p in div.find_all("p", class_="oj-doc-ti", recursive=False)]
        title = heads[1] if len(heads) > 1 else ""
        body = [line for line in render(div) if _clean(line) not in heads]
        text = "\n".join(body)
        out.append(
            Provision(
                id=div["id"],
                kind="annex",
                number=number,
                citation=f"Annex {number}",
                title=title,
                paragraphs=[Paragraph(None, f"Annex {number}", text)],
                text=text,
            )
        )
    return out


def _fix_superscripts(soup: BeautifulSoup) -> None:
    """'10<sup>25</sup>' would flatten to '10 25' (a meaning-changing error for the GPAI compute threshold).
    Keep exponents as '10^25'; drop footnote reference markers like '( 1 )', which are noise in chunks."""
    for el in soup.find_all(class_="oj-super"):
        prev = el.previous_sibling
        if "oj-note-tag" in el.get("class", []):
            el.decompose()
        elif el.get_text().isdigit() and isinstance(prev, str) and prev.rstrip().endswith("10"):
            el.replace_with("^" + el.get_text())


def parse(html: str) -> list[Provision]:
    soup = BeautifulSoup(html, "lxml")
    _fix_superscripts(soup)
    return parse_recitals(soup) + parse_articles(soup) + parse_annexes(soup)


def main() -> None:
    p = argparse.ArgumentParser(description="Parse the AI Act XHTML into provisions JSONL")
    p.add_argument("--html", type=Path, default=settings.raw_html)
    p.add_argument("--out", type=Path, default=settings.corpus_path)
    args = p.parse_args()
    provisions = parse(args.html.read_text(encoding="utf-8"))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for prov in provisions:
            f.write(json.dumps(asdict(prov), ensure_ascii=False) + "\n")
    kinds = {k: sum(p.kind == k for p in provisions) for k in ("article", "recital", "annex")}
    words = sum(len(p.text.split()) for p in provisions)
    print(f"Wrote {len(provisions)} provisions {kinds}, {words:,} words -> {args.out}")


if __name__ == "__main__":
    main()
