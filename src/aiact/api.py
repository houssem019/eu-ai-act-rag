"""HTTP API for the EU AI Act assistant.

    LLM_PROVIDER=ollama uv run uvicorn aiact.api:app --port 8000

- POST /ask          {"question": ...} -> the verified Answer (+ "cached": null | "exact" | "semantic")
- POST /ask/stream   Server-Sent Events: retrieval -> delta* -> [repair -> delta*] -> final. Deltas are a draft;
                     `final` carries the verified answer (invalid citations stripped, or a refusal) and replaces it.
- GET  /metrics      latency p50/p95 (retrieval vs LLM), cache hit rate, refusals by trigger, tokens, cost
- GET  /health
"""

import hashlib
import json
import logging
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse
from starlette.concurrency import iterate_in_threadpool

from aiact.answer import SYSTEM_PROMPT, Answer, Answerer
from aiact.cache import AnswerCache
from aiact.config import settings
from aiact.metrics import Metrics

log = logging.getLogger("aiact.api")


class AskRequest(BaseModel):
    question: str = Field(
        min_length=3, max_length=1000, examples=["Is social scoring banned under the AI Act?"]
    )


def answer_dict(ans: Answer) -> dict:
    return ans.to_dict() | {"cost_usd": ans.usage.cost_usd(ans.model)}


def config_version(model: str) -> str:
    """Cache namespace: any change to the prompt, retrieval config, corpus or model invalidates cached answers."""
    parts = [
        SYSTEM_PROMPT,
        model,
        settings.chunking,
        settings.embedder,
        str(settings.top_k),
        str(settings.recital_penalty),
        str(settings.refusal_threshold),
        hashlib.sha1(settings.corpus_path.read_bytes()).hexdigest(),
    ]
    return hashlib.sha1("\x00".join(parts).encode()).hexdigest()[:12]


def create_app(answerer: Answerer | None = None, cache: AnswerCache | None = None) -> FastAPI:
    state: dict = {"answerer": answerer, "cache": cache, "metrics": Metrics()}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        if state["answerer"] is None:  # production wiring; tests inject fakes
            from aiact.pipeline import build_answerer

            state["answerer"] = build_answerer()
        if state["cache"] is None and settings.cache_enabled:
            a = state["answerer"]
            state["cache"] = AnswerCache(
                a.retriever.dense,
                version=config_version(a.llm.model),
                threshold=settings.cache_threshold,
                max_entries=settings.cache_max_entries,
                ttl_s=settings.cache_ttl_s,
            )
        yield

    app = FastAPI(title="EU AI Act Assistant", version="0.1.0", lifespan=lifespan)

    def _cached(question: str) -> tuple[dict, str] | None:
        return state["cache"].get(question) if state["cache"] else None

    def _finish(question: str, answer: dict, t0: float, cached: str | None) -> dict:
        total_ms = (time.perf_counter() - t0) * 1000
        state["metrics"].record(answer, total_ms, cached)
        if not cached and state["cache"] and answer["refusal_reason"] != "safety":
            state["cache"].put(question, answer)
        log.info(
            json.dumps(
                {
                    "event": "ask",
                    "cached": cached,
                    "refused": answer["refused"],
                    "reason": answer["refusal_reason"],
                    "citations": answer["citations"],
                    "top_score": round(answer["top_score"], 2),
                    "total_ms": round(total_ms),
                    "llm_calls": 0 if cached else answer["llm_calls"],
                    "cost_usd": 0.0 if cached else answer["cost_usd"],
                }
            )
        )
        return answer | {"cached": cached}

    @app.get("/health")
    def health() -> dict:
        a: Answerer = state["answerer"]
        return {
            "status": "ok",
            "llm_provider": settings.llm_provider,
            "model": a.llm.model,
            "chunks": len(getattr(getattr(a.retriever, "index", None), "chunks", [])),
            "cache_entries": len(state["cache"].entries) if state["cache"] else None,
        }

    @app.get("/metrics")
    def metrics() -> dict:
        return state["metrics"].snapshot() | {
            "cache_guard_rejections": state["cache"].stats.guard_rejections if state["cache"] else 0
        }

    @app.post("/ask")
    def ask(req: AskRequest) -> dict:
        t0 = time.perf_counter()
        if hit := _cached(req.question):
            return _finish(req.question, hit[0], t0, hit[1])
        try:
            ans = state["answerer"].ask(req.question)
        except Exception as e:
            state["metrics"].record_error()
            log.exception("ask failed")
            raise HTTPException(502, f"answering failed: {type(e).__name__}") from e
        return _finish(req.question, answer_dict(ans), t0, None)

    @app.post("/ask/stream")
    def ask_stream(req: AskRequest) -> EventSourceResponse:
        t0 = time.perf_counter()

        def events() -> Iterator[dict]:
            if hit := _cached(req.question):
                yield {"type": "final", "answer": _finish(req.question, hit[0], t0, hit[1])}
                return
            try:
                for e in state["answerer"].stream(req.question):
                    if e["type"] == "final":
                        e = {
                            "type": "final",
                            "answer": _finish(req.question, answer_dict(e["answer"]), t0, None),
                        }
                    yield e
            except Exception as e:  # headers are already sent: report the failure as an event
                state["metrics"].record_error()
                log.exception("stream failed")
                yield {"type": "error", "message": f"answering failed: {type(e).__name__}"}

        async def gen():
            async for e in iterate_in_threadpool(events()):
                yield {"event": e["type"], "data": json.dumps(e, default=str)}

        return EventSourceResponse(gen())

    return app


app = create_app()
