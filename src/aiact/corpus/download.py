"""Download the official English text of Regulation (EU) 2024/1689 (the AI Act).

EUR-Lex's HTML pages sit behind a bot challenge (HTTP 202 with an empty body), so we use the EU
Publications Office's content-negotiated CELEX endpoint, which serves the same Official Journal XHTML.
EU legislation is free to reuse (Commission Decision 2011/833/EU).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import httpx

from aiact.config import settings

CELEX_URL = "https://publications.europa.eu/resource/celex/32024R1689"


def download(out: Path = settings.raw_html) -> Path:
    r = httpx.get(
        CELEX_URL,
        headers={"Accept": "application/xhtml+xml, text/html", "Accept-Language": "en"},
        follow_redirects=True,
        timeout=120,
    )
    r.raise_for_status()
    if 'id="art_1"' not in r.text:
        raise RuntimeError(f"Unexpected document from {r.url} ({len(r.content)} bytes)")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=settings.raw_html)
    print(f"Saved {download(p.parse_args().out)}")
