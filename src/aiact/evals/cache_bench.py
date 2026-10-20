"""Tune the semantic cache on labelled question pairs (data/eval/cache_pairs.yaml).

For each cosine threshold, with and without the discriminating-term guard: how many paraphrases would be served
from cache (hit rate, saves an LLM call) and how many look-alikes would wrongly be (false hits: a wrong answer).
The shipped threshold is the lowest one with zero false hits under the guard.

    uv run python -m aiact.evals.cache_bench
"""

import json
from pathlib import Path

import numpy as np
import yaml

from aiact.cache import discriminators
from aiact.config import settings
from aiact.index import FastEmbedDense

PAIRS = settings.eval_path.parent / "cache_pairs.yaml"
OUT = settings.cassette_dir.parent.parent / "bench" / "cache.json"
THRESHOLDS = [0.80, 0.82, 0.84, 0.86, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]


def evaluate(pairs: list[dict], sims: list[float]) -> dict:
    guard_ok = [discriminators(p["a"]) == discriminators(p["b"]) for p in pairs]
    n_pos = sum(p["same"] for p in pairs)
    n_neg = len(pairs) - n_pos
    rows = []
    for t in THRESHOLDS:
        for guard in (False, True):
            hit = [s >= t and (ok or not guard) for s, ok in zip(sims, guard_ok, strict=True)]
            rows.append(
                {
                    "threshold": t,
                    "guard": guard,
                    "hit_rate": sum(h and p["same"] for h, p in zip(hit, pairs, strict=True)) / n_pos,
                    "false_hits": sum(h and not p["same"] for h, p in zip(hit, pairs, strict=True)),
                }
            )
    safe = [r for r in rows if r["guard"] and r["false_hits"] == 0]
    chosen = min(safe, key=lambda r: r["threshold"]) if safe else None
    blocked = [
        {
            "a": p["a"],
            "b": p["b"],
            "a_terms": sorted(discriminators(p["a"])),
            "b_terms": sorted(discriminators(p["b"])),
        }
        for p, ok in zip(pairs, guard_ok, strict=True)
        if not ok
    ]
    return {
        "n_paraphrases": n_pos,
        "n_lookalikes": n_neg,
        "rows": rows,
        "chosen": chosen,
        "guard_blocked": blocked,
    }


def main(path: Path = PAIRS) -> None:
    pairs = yaml.safe_load(path.read_text(encoding="utf-8"))
    enc = FastEmbedDense(settings.embedder)

    def unit(q):
        v = enc.embed_query(q)
        return v / np.linalg.norm(v)

    sims = [float(unit(p["a"]) @ unit(p["b"])) for p in pairs]
    result = evaluate(pairs, sims) | {"similarities": [round(s, 4) for s in sims]}
    print(f"{result['n_paraphrases']} paraphrase pairs, {result['n_lookalikes']} look-alike pairs\n")
    print("threshold | no guard: hit rate / false hits | guard: hit rate / false hits")
    for t in THRESHOLDS:
        a, b = (r for r in result["rows"] if r["threshold"] == t)
        print(
            f"   {t:.2f}   |   {a['hit_rate']:.2f} / {a['false_hits']:2d}   |   {b['hit_rate']:.2f} / {b['false_hits']:2d}"
        )
    print(f"\nchosen: {result['chosen']}")
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
