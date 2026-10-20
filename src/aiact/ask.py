"""Ask the EU AI Act assistant from the command line.

LLM_PROVIDER=ollama uv run python -m aiact.ask "Is social scoring banned?"
LLM_PROVIDER=anthropic uv run python -m aiact.ask --json "Who is a deployer?"
"""

import argparse
import json
import sys

from aiact.config import settings
from aiact.pipeline import build_answerer


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("question")
    p.add_argument("--json", action="store_true", help="print the full Answer record as JSON")
    args = p.parse_args()
    if settings.llm_provider == "fake":
        sys.exit("Set LLM_PROVIDER=ollama (free, local) or LLM_PROVIDER=anthropic to ask live questions.")
    ans = build_answerer().ask(args.question)
    if args.json:
        print(json.dumps(ans.to_dict(), indent=2, ensure_ascii=False))
        return
    print(ans.text)
    print(f"\nSources: {', '.join(s['label'] for s in ans.sources)}")
    cost = ans.usage.cost_usd(ans.model)
    status = f"refused ({ans.refusal_reason})" if ans.refused else f"cites {', '.join(ans.citations)}"
    print(f"[{status} | top score {ans.top_score:.2f} | {ans.latency_ms / 1000:.1f}s | ${cost:.4f}]")


if __name__ == "__main__":
    main()
