"""Answer-level evaluation over the curated eval set (deterministic metrics; LLM-judged metrics come later).

Record once with a real model (retrieval snapshot + LLM cassette), then replay for free in CI:

    uv run python -m aiact.evals.answers --provider ollama --record          # ~1 min/question on CPU
    uv run python -m aiact.evals.answers --provider fake --gate data/eval/answer_baseline.json

Metrics (answerable = 55 questions with gold citations, unanswerable = 8):
- answer_rate:        answerable questions that got an answer (1 - false refusals)
- refusal_rate:       unanswerable questions that were refused
- citation_accuracy:  answered questions citing at least one gold unit
- invalid_citation_rate: answers whose first draft cited something not in the sources (stripped before display)
- article_citation_rate: answered questions citing at least one article or annex (recitals are not binding)
"""

import argparse
import json
import statistics
from pathlib import Path

from aiact.answer import Answer, Answerer
from aiact.bench import load_questions
from aiact.config import settings
from aiact.llm import get_llm
from aiact.retrieve import SnapshotRetriever

OUT_DIR = Path("eval-results")


def score(questions: list[dict], answers: list[Answer]) -> dict:
    rows = []
    for q, a in zip(questions, answers, strict=True):
        rows.append(
            {
                "id": q["id"],
                "answerable": q["answerable"],
                "refused": a.refused,
                "refusal_reason": a.refusal_reason,
                "citations": a.citations,
                "gold_hit": bool(set(q["gold"]) & set(a.citations)),
                "cites_article": any(not c.startswith("Recital") for c in a.citations),
                "invalid": a.invalid_citations,
                "repaired": a.repaired,
                "top_score": round(a.top_score, 2),
                "latency_ms": round(a.latency_ms),
                "input_tokens": a.usage.input_tokens,
                "output_tokens": a.usage.output_tokens,
                "cost_usd": a.usage.cost_usd(a.model),
                "answer": a.text,
            }
        )
    pos = [r for r in rows if r["answerable"]]
    neg = [r for r in rows if not r["answerable"]]
    answered = [r for r in pos if not r["refused"]]
    called = [r for r in rows if r["input_tokens"]]

    def frac(xs, pred):
        return sum(map(pred, xs)) / len(xs) if xs else 0.0

    metrics = {
        "answer_rate": frac(pos, lambda r: not r["refused"]),
        "refusal_rate": frac(neg, lambda r: r["refused"]),
        "citation_accuracy": frac(answered, lambda r: r["gold_hit"]),
        "invalid_citation_rate": frac(called, lambda r: bool(r["invalid"])),
        "article_citation_rate": frac(answered, lambda r: r["cites_article"]),
        "repair_rate": frac(called, lambda r: r["repaired"]),
        "p50_latency_ms": statistics.median(r["latency_ms"] for r in called) if called else 0.0,
        "mean_cost_usd": statistics.mean(r["cost_usd"] for r in rows),
        "refusal_reasons": {
            k: sum(r["refusal_reason"] == k for r in rows)
            for k in ("low_evidence", "model", "uncited", "safety")
        },
    }
    return {"n_answerable": len(pos), "n_unanswerable": len(neg), "metrics": metrics, "rows": rows}


def to_markdown(result: dict, model: str) -> str:
    m = result["metrics"]
    lines = [
        f"Model: `{model}`. {result['n_answerable']} answerable + {result['n_unanswerable']} unanswerable questions.",
        "",
        "| Answer rate | Refusal rate | Citation accuracy | Invalid citations (draft) | Cites article/annex | p50 latency | Cost/question |",
        "|---|---|---|---|---|---|---|",
        f"| {m['answer_rate']:.2f} | {m['refusal_rate']:.2f} | {m['citation_accuracy']:.2f} | "
        f"{m['invalid_citation_rate']:.2f} | {m['article_citation_rate']:.2f} | {m['p50_latency_ms'] / 1000:.1f}s | "
        f"${m['mean_cost_usd']:.4f} |",
        "",
        f"Refusals by trigger: {m['refusal_reasons']}",
        "",
        "| id | refused | cited | gold hit |",
        "|---|---|---|---|",
    ]
    for r in result["rows"]:
        refused = r["refusal_reason"] if r["refused"] else ""
        hit = "" if not r["answerable"] or r["refused"] else ("yes" if r["gold_hit"] else "**no**")
        lines.append(f"| {r['id']} | {refused} | {'; '.join(r['citations'])} | {hit} |")
    return "\n".join(lines) + "\n"


def check_gate(metrics: dict, baseline: dict) -> list[str]:
    fails = []
    for k, floor in baseline.get("min", {}).items():
        if metrics[k] < floor:
            fails.append(f"FAIL {k}={metrics[k]:.3f} < baseline {floor}")
    for k, ceiling in baseline.get("max", {}).items():
        if metrics[k] > ceiling:
            fails.append(f"FAIL {k}={metrics[k]:.3f} > baseline {ceiling}")
    return fails


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["fake", "ollama", "anthropic"], default=settings.llm_provider)
    p.add_argument("--cassette", type=Path, help="LLM cassette (default: data/eval/cassettes/<model>.json)")
    p.add_argument("--snapshot", type=Path, default=settings.cassette_dir / "retrieval.json")
    p.add_argument("--record", action="store_true", help="run real retrieval + LLM and save both")
    p.add_argument("--ids", nargs="+", help="only these question ids")
    p.add_argument("--gate", type=Path, help="baseline JSON; exit 1 on regression")
    p.add_argument("--out", type=Path, default=OUT_DIR)
    args = p.parse_args()

    cfg = settings.model_copy(update={"llm_provider": args.provider})
    model = {"ollama": cfg.ollama_model, "anthropic": cfg.llm_model}.get(args.provider)
    cassette = args.cassette or cfg.cassette_dir / f"{(model or 'qwen2.5:7b').replace(':', '-')}.json"
    if args.record:
        from aiact.pipeline import build_retriever

        if args.provider == "fake":
            p.error("--record needs a real provider")
        retriever = SnapshotRetriever(args.snapshot, inner=build_retriever(cfg))
    else:
        retriever = SnapshotRetriever(args.snapshot)
    llm = get_llm(cfg, cassette=cassette, record=args.record)
    answerer = Answerer(retriever, llm, threshold=cfg.refusal_threshold, top_k=cfg.top_k)

    questions = [q for q in load_questions() if not args.ids or q["id"] in args.ids]
    answers = []
    for q in questions:
        a = answerer.ask(q["question"])
        answers.append(a)
        flag = f"refused/{a.refusal_reason}" if a.refused else "; ".join(a.citations)
        print(f"{q['id']:5s} {a.latency_ms / 1000:6.1f}s {flag}", flush=True)

    result = score(questions, answers)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "answers.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.out / "answers.md").write_text(to_markdown(result, llm.model), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
    if args.gate:
        fails = check_gate(result["metrics"], json.loads(args.gate.read_text(encoding="utf-8")))
        print("\n".join(fails) or "Answer gate passed.")
        raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
