# syntax=docker/dockerfile:1
# The assistant's HTTP API (FastAPI + SSE). Retrieval models and corpus embeddings are baked in at build time, so
# the container needs no downloads to start.
#   docker build -t aiact .
#   docker run --rm -p 8000:8000 -e LLM_PROVIDER=ollama -e OLLAMA_HOST=http://host.docker.internal:11434 aiact
#   docker run --rm -p 8000:8000 -e LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY aiact
#   CLI instead of the API: docker run --rm -e LLM_PROVIDER=... --entrypoint python aiact -m aiact.ask "question"
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.1 /uv /bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    AIACT_CACHE_DIR=/app/data/cache

WORKDIR /app
# Dependencies first (cached layer), project code second
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev --no-install-project
COPY src ./src
COPY data/corpus ./data/corpus
RUN --mount=type=cache,target=/root/.cache/uv uv sync --locked --no-dev

# ~160 MB of ONNX models (bge-small, bm25, MiniLM reranker) + the corpus embeddings
RUN python -m aiact.warmup
# Models are baked in: never contact the Hugging Face hub at runtime.
ENV HF_HUB_OFFLINE=1

RUN useradd --create-home app && chown -R app /app/data/cache
USER app

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1
# One worker: the index lives in-process (Qdrant local mode). Scale with replicas, or a Qdrant server + workers.
CMD ["uvicorn", "aiact.api:app", "--host", "0.0.0.0", "--port", "8000"]
