"""LLM-judged answer quality with Ragas, on top of the recorded answer eval.

For every answerable question that was answered (data/eval/results/answers.json), with the sources the model
actually saw (the retrieval snapshot) and the curated reference answer:

- faithfulness:       share of the answer's atomic statements supported by its retrieved sources (Ragas
                      Faithfulness: statement extraction + NLI). Catches hallucination beyond the context.
- answer_correctness: factual agreement with the reference answer (Ragas AnswerCorrectness, factuality only:
                      TP/FP/FN statement classification -> F1; no embedding similarity term).

Refusals are graded deterministically in aiact.evals.answers, not here. Record once, replay in CI:

    JUDGE_PROVIDER=anthropic uv run python -m aiact.evals.judged --record     # Opus judge (needs a key)
    uv run python -m aiact.evals.judged --provider ollama --record            # free local judge, slow on CPU
    uv run python -m aiact.evals.judged --provider fake --gate data/eval/judged_baseline.json
"""

import argparse
import asyncio
import json
import math
import random
import statistics
from pathlib import Path

from ragas.metrics.collections import AnswerCorrectness, Faithfulness

from aiact.bench import load_questions
from aiact.config import settings
from aiact.evals.judge import JudgeError, RagasJudge
from aiact.llm.base import LLMClient
from aiact.llm.replay import ReplayLLM

RESULTS = settings.cassette_dir.parent / "results"


def build_judge_llm(provider: str, cassette: Path, record: bool) -> LLMClient:
    if provider == "fake":
        return ReplayLLM(cassette)
    if provider == "anthropic":
        from aiact.llm.anthropic_client import AnthropicLLM

        inner: LLMClient = AnthropicLLM(settings.judge_model, effort="medium", max_tokens=8000)
    else:
        from aiact.llm.ollama_client import OllamaLLM

        inner = OllamaLLM(settings.ollama_model, settings.ollama_host)
    return ReplayLLM(cassette, inner=inner) if record else inner


def judge_samples(samples: list[dict], judge: RagasJudge) -> list[dict]:
    faith = Faithfulness(llm=judge)
    correct = AnswerCorrectness(llm=judge, weights=[1.0, 0.0])  # factuality only: no embedding model needed
    out = []
    for s in samples:
        row = {"id": s["id"], "faithfulness": math.nan, "answer_correctness": math.nan, "error": None}
        try:
            row["faithfulness"] = asyncio.run(
                faith.ascore(user_input=s["question"], response=s["answer"], retrieved_contexts=s["contexts"])
            ).value
            row["answer_correctness"] = asyncio.run(
                correct.ascore(user_input=s["question"], response=s["answer"], reference=s["reference"])
            ).value
        except JudgeError as e:
            row["error"] = str(e)[:300]
        out.append(row)
        print(
            f"{s['id']:5s} faith={row['faithfulness']:.2f} correct={row['answer_correctness']:.2f}",
            flush=True,
        )
    return out


def load_samples(answers_path: Path, snapshot_path: Path, ids: list[str] | None = None) -> list[dict]:
    questions = {q["id"]: q for q in load_questions()}
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    rows = json.loads(answers_path.read_text(encoding="utf-8"))["rows"]
    samples = []
    for r in rows:
        q = questions[r["id"]]
        if not q["answerable"] or r["refused"] or (ids and r["id"] not in ids):
            continue
        hits = snapshot[f"rerank|{settings.top_k}|{q['question']}"]["hits"]
        samples.append(
            {
                "id": r["id"],
                "category": q["category"],
                "question": q["question"],
                "answer": r["answer"],
                "reference": q["reference"],
                "contexts": [h["text"] for h in hits],
            }
        )
    return samples


def stratified_sample(samples: list[dict], n: int, seed: int = 0) -> list[dict]:
    """n samples allocated proportionally across eval categories (at least one each), chosen with a fixed seed:
    reproducible, and not hand-picked towards easy or hard questions. Keeps the eval-set order."""
    if n >= len(samples):
        return samples
    by_cat: dict[str, list[dict]] = {}
    for s in samples:
        by_cat.setdefault(s["category"], []).append(s)
    quota = {c: max(1, round(n * len(v) / len(samples))) for c, v in by_cat.items()}
    while sum(quota.values()) > n:  # rounding overshoot: trim the largest categories first
        quota[max(quota, key=lambda c: (quota[c], c))] -= 1
    while sum(quota.values()) < n:
        quota[max(by_cat, key=lambda c: (len(by_cat[c]) - quota[c], c))] += 1
    rng = random.Random(seed)
    chosen = {s["id"] for c, v in by_cat.items() for s in rng.sample(v, quota[c])}
    return [s for s in samples if s["id"] in chosen]


def summarize(rows: list[dict]) -> dict:
    def mean(key):
        vals = [r[key] for r in rows if not math.isnan(r[key])]
        return statistics.mean(vals) if vals else math.nan

    return {
        "n_judged": len(rows),
        "faithfulness": mean("faithfulness"),
        "answer_correctness": mean("answer_correctness"),
        "unfaithful_answers": sum(r["faithfulness"] < 1.0 for r in rows if not math.isnan(r["faithfulness"])),
        "judge_failures": sum(
            math.isnan(r["faithfulness"]) or math.isnan(r["answer_correctness"]) for r in rows
        ),
    }


def to_markdown(result: dict) -> str:
    m = result["metrics"]
    lines = [
        f"Judge: `{result['judge']}` over `{result['generator']}` answers. {m['n_judged']} answered questions"
        + (f" (stratified sample of {result['sample']}, seed 0)." if result.get("sample") else "."),
        "",
        "| Faithfulness | Answer correctness | Answers with an unsupported statement | Judge failures |",
        "|---|---|---|---|",
        f"| {m['faithfulness']:.3f} | {m['answer_correctness']:.3f} | {m['unfaithful_answers']} | {m['judge_failures']} |",
        "",
        "| id | faithfulness | answer correctness |",
        "|---|---|---|",
    ]
    lines += [
        f"| {r['id']} | {r['faithfulness']:.2f} | {r['answer_correctness']:.2f} |" for r in result["rows"]
    ]
    return "\n".join(lines) + "\n"


def check_gate(metrics: dict, baseline: dict) -> list[str]:
    fails = [
        f"FAIL {k}={metrics[k]:.3f} < baseline {v}"
        for k, v in baseline.get("min", {}).items()
        if not metrics[k] >= v
    ]
    fails += [
        f"FAIL {k}={metrics[k]} > baseline {v}" for k, v in baseline.get("max", {}).items() if metrics[k] > v
    ]
    return fails


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=["fake", "ollama", "anthropic"], default=settings.judge_provider)
    p.add_argument(
        "--cassette", type=Path, help="judge cassette (default: data/eval/cassettes/judge-<model>.json)"
    )
    p.add_argument("--answers", type=Path, default=RESULTS / "answers.json")
    p.add_argument("--snapshot", type=Path, default=settings.cassette_dir / "retrieval.json")
    p.add_argument("--record", action="store_true")
    p.add_argument("--ids", nargs="+")
    p.add_argument("--sample", type=int, help="stratified sample of N answered questions (fixed seed)")
    p.add_argument("--gate", type=Path)
    p.add_argument("--out", type=Path, default=Path("eval-results"))
    args = p.parse_args()

    if args.cassette:
        cassette = args.cassette
    elif args.provider == "fake":  # replay whichever judge was recorded
        found = sorted(settings.cassette_dir.glob("judge-*.json"))
        if len(found) != 1:
            p.error(
                f"expected exactly one judge cassette in {settings.cassette_dir}, found {len(found)}: pass --cassette"
            )
        cassette = found[0]
    else:
        name = settings.judge_model if args.provider == "anthropic" else settings.ollama_model
        cassette = settings.cassette_dir / f"judge-{name.replace(':', '-')}.json"
    judge = RagasJudge(build_judge_llm(args.provider, cassette, args.record))

    baseline = json.loads(args.gate.read_text(encoding="utf-8")) if args.gate else {}
    # A baseline can pin the exact judged question ids (e.g. a recording that was stopped early), so replay never
    # asks the cassette for a sample it doesn't hold.
    samples = load_samples(args.answers, args.snapshot, args.ids or baseline.get("ids"))
    if args.sample:
        samples = stratified_sample(samples, args.sample)
    rows = judge_samples(samples, judge)
    result = {
        "judge": judge.llm.model,
        "generator": json.loads(args.answers.read_text(encoding="utf-8")).get("model", "unknown"),
        "sample": args.sample,
        "metrics": summarize(rows) | {"judge_calls": judge.calls, "parse_retries": judge.parse_retries},
        "cost_usd": judge.usage.cost_usd(judge.llm.model),
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "judged.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (args.out / "judged.md").write_text(to_markdown(result), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2), f"\njudge cost ${result['cost_usd']:.4f}")
    if args.gate:
        fails = check_gate(result["metrics"], baseline)
        print("\n".join(fails) or "Judged gate passed.")
        raise SystemExit(1 if fails else 0)


if __name__ == "__main__":
    main()
