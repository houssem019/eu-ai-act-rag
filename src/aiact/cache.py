"""Answer cache for the API: exact match on the normalized question, then a guarded semantic match.

Semantic caching is risky for legal QA: questions that differ by one term ("provider" vs "deployer",
"real-time" vs "post" remote biometric identification, "point 4" vs "point 5") embed almost identically
(cosine up to 0.96 on data/eval/cache_pairs.yaml) but have different answers. A false hit serves a wrong
answer with valid-looking citations. So a semantic hit needs BOTH
  - cosine >= threshold (tuned on labelled paraphrase / look-alike pairs: aiact.evals.cache_bench), and
  - the same discriminating terms: the Act's own defined terms (Article 3), plus numbers and
    Article / Annex / point references.
Entries are keyed by a version string (prompt + retrieval config + model), so a config change never serves stale
answers. In-process and size-bounded (LRU); at scale this would be Redis with a vector index.
"""

import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

import numpy as np

from aiact.config import settings

# Defined terms too generic to tell questions apart ("AI system" appears in paraphrases of almost everything).
_NEUTRAL_TERMS = {"ai system", "risk", "operator"}
_REF = re.compile(r"\b(?:article|annex|point|paragraph|recital|chapter)\s+[0-9ivxlc]+(?:\s*\(\w+\))*", re.I)
_NUMBER = re.compile(r"\b\d+(?:[.,]\d+)?\b")


def normalize(question: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s()]", " ", question.lower())).strip()


@cache
def defined_terms(corpus_path: Path | None = None) -> tuple[str, ...]:
    """Article 3's defined terms, longest first so 'real-time remote biometric identification system' wins
    over 'remote biometric identification system'."""
    for line in (corpus_path or settings.corpus_path).read_text(encoding="utf-8").splitlines():
        prov = json.loads(line)
        if prov["id"] == "art_3":
            terms = []
            for p in prov["paragraphs"]:
                if m := re.match(r"\(\d+\)\s*[‘'\"](.+?)[’'\"]\s+means", p["text"]):
                    terms.append(m.group(1).lower().replace("-", " "))
            return tuple(sorted(set(terms) - _NEUTRAL_TERMS, key=len, reverse=True))
    raise ValueError("Article 3 not found in corpus")


def discriminators(question: str, terms: tuple[str, ...] | None = None) -> frozenset[str]:
    """Terms that change the answer if swapped: defined terms (plural- and 'system'-tolerant), legal references
    and numbers. Two questions may share a cached answer only if these sets are equal."""
    text = " " + re.sub(r"[^\w\s()]", " ", question.lower().replace("-", " ")) + " "
    text = re.sub(r"\s+", " ", text)
    found = set()
    terms = terms if terms is not None else defined_terms()
    # Full terms plus "... system" terms without the word "system" ("real-time remote biometric identification"),
    # matched longest pattern first: "general-purpose AI model" must beat the stem "general-purpose AI", and
    # "real-time remote biometric identification" must beat the defined term "biometric identification".
    variants = [(t, t) for t in terms] + [
        (t, t.removesuffix(" system")) for t in terms if t.endswith(" system")
    ]
    for term, stem in sorted(variants, key=lambda v: len(v[1]), reverse=True):
        stem = re.escape(stem)
        pattern = rf" {stem}(?:s|es)? "
        if re.search(pattern, text):
            found.add(term)
            text = re.sub(pattern, " ", text)  # consume, so a shorter term inside it isn't matched again
    found |= {re.sub(r"\s+", " ", r.lower()) for r in _REF.findall(question)}
    found |= set(_NUMBER.findall(_REF.sub(" ", question)))
    return frozenset(found)


@dataclass
class CacheStats:
    exact_hits: int = 0
    semantic_hits: int = 0
    guard_rejections: int = 0  # cosine was above threshold but the discriminating terms differed
    misses: int = 0


@dataclass
class _Entry:
    question: str
    vector: np.ndarray
    terms: frozenset[str]
    value: dict
    created: float = field(default_factory=time.time)


class AnswerCache:
    def __init__(
        self, encoder, version: str, threshold: float, max_entries: int = 1000, ttl_s: float = 86400
    ):
        self.encoder, self.version, self.threshold = encoder, version, threshold
        self.max_entries, self.ttl_s = max_entries, ttl_s
        self.entries: OrderedDict[str, _Entry] = OrderedDict()
        self.stats = CacheStats()
        self._lock = threading.Lock()

    def _vec(self, question: str) -> np.ndarray:
        v = np.asarray(self.encoder.embed_query(question), dtype=np.float32)
        return v / (np.linalg.norm(v) or 1.0)

    def _key(self, question: str) -> str:
        return f"{self.version}|{normalize(question)}"

    def get(self, question: str) -> tuple[dict, str] | None:
        """-> (cached answer dict, "exact" | "semantic") or None."""
        with self._lock:
            self._expire()
            key = self._key(question)
            if key in self.entries:
                self.entries.move_to_end(key)
                self.stats.exact_hits += 1
                return self.entries[key].value, "exact"
            if not self.entries:
                self.stats.misses += 1
                return None
            vec, terms = self._vec(question), discriminators(question)
            keys = [
                k for k in self.entries if k.startswith(f"{self.version}|")
            ]  # never match across versions
            if not keys:
                self.stats.misses += 1
                return None
            sims = np.stack([self.entries[k].vector for k in keys]) @ vec
            best = int(np.argmax(sims))
            if sims[best] >= self.threshold:
                entry = self.entries[keys[best]]
                if entry.terms == terms:
                    self.entries.move_to_end(keys[best])
                    self.stats.semantic_hits += 1
                    return entry.value, "semantic"
                self.stats.guard_rejections += 1
            self.stats.misses += 1
            return None

    def put(self, question: str, value: dict) -> None:
        with self._lock:
            key = self._key(question)
            self.entries[key] = _Entry(question, self._vec(question), discriminators(question), value)
            self.entries.move_to_end(key)
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)

    def _expire(self) -> None:
        now = time.time()
        for k in [k for k, e in self.entries.items() if now - e.created >= self.ttl_s]:
            del self.entries[k]
