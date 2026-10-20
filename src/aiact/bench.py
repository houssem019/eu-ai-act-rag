"""Retrieval benchmark: chunking x embedder x mode over the curated eval set.

A question is a hit at k if any of its gold citation units is among the citations of the top-k chunks.
Also records context size (tokens of the top-5 chunks, i.e. what the LLM would have to read), latency, and the
top rerank score on answerable vs unanswerable questions to choose the refusal threshold.

    uv run python -m aiact.bench                     # full grid (~50 min on a 12-core laptop CPU; the reranker dominates)
    uv run python -m aiact.bench --embedders bge-small --chunkings article paragraph
"""

import argparse
import json
import statistics
from pathlib import Path

import yaml

from aiact.chunking import STRATEGIES, chunk, load_provisions, n_tokens
from aiact.config import PROJECT_ROOT, settings
from aiact.index import DENSE_MODELS, FastEmbedBM25, FastEmbedDense, FastEmbedReranker, build_index
from aiact.retrieve import MODES, Hit, Retriever

OUT_DIR = PROJECT_ROOT / "data" / "bench"


def load_questions(path: Path | None = None) -> list[dict]:
    return yaml.safe_load((path or settings.eval_path).read_text(encoding="utf-8"))


def first_hit_rank(hits: list[Hit], gold: set[str]) -> int | None:
    """1-based rank of the first chunk that contains a gold citation."""
    for rank, h in enumerate(hits, 1):
        if gold & set(h.chunk.citations):
            return rank
    return None


def summarize(ranks: list[int | None], ks: tuple[int, ...] = (1, 5, 10)) -> dict[str, float]:
    out = {f"recall@{k}": sum(r is not None and r <= k for r in ranks) / len(ranks) for k in ks}
    out["mrr@10"] = sum(1 / r for r in ranks if r is not None and r <= 10) / len(ranks)
    return out


def choose_threshold(pos: list[float], neg: list[float]) -> dict[str, float]:
    """Refuse when the top rerank score < threshold. Pick the cut maximizing balanced accuracy;
    ties go to the lower threshold (prefer answering over refusing)."""
    best = None
    for t in sorted(set(pos + neg)):
        tpr = sum(s >= t for s in pos) / len(pos)  # answerable questions we still answer
        tnr = sum(s < t for s in neg) / len(neg)  # unanswerable questions we refuse
        if best is None or (tpr + tnr) / 2 > best["balanced_acc"]:
            best = {"threshold": t, "answer_rate": tpr, "refusal_rate": tnr, "balanced_acc": (tpr + tnr) / 2}
    return best


def run(chunkings, embedders, modes) -> dict:
    questions = load_questions()
    answerable = [q for q in questions if q["answerable"]]
    unanswerable = [q for q in questions if not q["answerable"]]
    provisions = load_provisions()
    sparse, reranker = FastEmbedBM25(), FastEmbedReranker()
    rows, refusal = [], []
    for emb in embedders:
        dense = FastEmbedDense(emb)
        for strat in chunkings:
            chunks = chunk(strat, provisions)
            index = build_index(chunks, dense, sparse, f"{strat}_{emb}")
            retr = Retriever(index, dense, sparse, reranker)
            retr.search("warm-up query", "rerank" if "rerank" in modes else modes[0])
            for mode in modes:
                ranks, lat, ctx, top = [], [], [], {}
                # One pass over all questions: unanswerable ones only feed the refusal threshold
                # (reranking is the slow step, ~2-6 s/query on CPU, so never run it twice).
                for q in questions:
                    res = retr.search(q["question"], mode, k=10)
                    top[q["id"]] = res.hits[0].score
                    if not q["answerable"]:
                        continue
                    ranks.append(first_hit_rank(res.hits, set(q["gold"])))
                    lat.append(res.latency_ms)
                    ctx.append(sum(n_tokens(h.chunk.text) for h in res.hits[:5]))
                row = {"chunking": strat, "embedder": emb, "mode": mode, **summarize(ranks)}
                row |= {"ctx_tokens@5": statistics.mean(ctx), "p50_ms": statistics.median(lat)}
                row["misses@10"] = [q["id"] for q, r in zip(answerable, ranks, strict=True) if r is None]
                rows.append(row)
                print(_fmt(row), flush=True)
                if mode == "rerank":
                    pos = [top[q["id"]] for q in answerable]
                    neg = [top[q["id"]] for q in unanswerable]
                    refusal.append(
                        {"chunking": strat, "embedder": emb, **choose_threshold(pos, neg), "top_scores": top}
                    )
    return {
        "n_answerable": len(answerable),
        "n_unanswerable": len(unanswerable),
        "rows": rows,
        "refusal": refusal,
    }


def _fmt(r: dict) -> str:
    return (
        f"{r['chunking']:9s} {r['embedder']:9s} {r['mode']:6s} R@1={r['recall@1']:.2f} R@5={r['recall@5']:.2f} "
        f"R@10={r['recall@10']:.2f} MRR={r['mrr@10']:.3f} ctx={r['ctx_tokens@5']:.0f} p50={r['p50_ms']:.0f}ms"
    )


def to_markdown(result: dict) -> str:
    lines = [
        f"Answerable questions: {result['n_answerable']}. Hit = any gold citation in the top-k chunks. "
        "ctx@5 = mean tokens in the top-5 chunks. Latency is per query on CPU (laptop), including query embedding.",
        "",
        "| Chunking | Embedder | Mode | R@1 | R@5 | R@10 | MRR@10 | ctx@5 tokens | p50 ms |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in result["rows"]:
        lines.append(
            f"| {r['chunking']} | {r['embedder']} | {r['mode']} | {r['recall@1']:.2f} | {r['recall@5']:.2f} | "
            f"{r['recall@10']:.2f} | {r['mrr@10']:.3f} | {r['ctx_tokens@5']:.0f} | {r['p50_ms']:.0f} |"
        )
    if result["refusal"]:
        lines += [
            "",
            f"Refusal threshold on the top rerank score ({result['n_answerable']} answerable vs "
            f"{result['n_unanswerable']} unanswerable questions):",
            "",
            "| Chunking | Embedder | Threshold | Answer rate (answerable) | Refusal rate (unanswerable) |",
            "|---|---|---|---|---|",
        ]
        for r in result["refusal"]:
            lines.append(
                f"| {r['chunking']} | {r['embedder']} | {r['threshold']:.2f} | {r['answer_rate']:.2f} | "
                f"{r['refusal_rate']:.2f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--chunkings", nargs="+", default=list(STRATEGIES), choices=STRATEGIES)
    p.add_argument("--embedders", nargs="+", default=list(DENSE_MODELS), choices=list(DENSE_MODELS))
    p.add_argument("--modes", nargs="+", default=list(MODES), choices=MODES)
    p.add_argument("--out", type=Path, default=OUT_DIR / "retrieval.json")
    args = p.parse_args()
    result = run(args.chunkings, args.embedders, args.modes)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    args.out.with_suffix(".md").write_text(to_markdown(result), encoding="utf-8")
    print(f"\nWrote {args.out} and {args.out.with_suffix('.md')}")


if __name__ == "__main__":
    main()
