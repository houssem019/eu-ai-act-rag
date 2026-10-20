# syntax=docker/dockerfile:1
# The assistant as a CLI image (an HTTP API comes with the serving milestone). Retrieval models and corpus
# embeddings are baked in at build time, so the container needs no downloads to start.
#   docker build -t aiact .
#   docker run --rm -e LLM_PROVIDER=ollama -e OLLAMA_HOST=http://host.docker.internal:11434 aiact "Is social scoring banned?"
#   docker run --rm -e LLM_PROVIDER=anthropic -e ANTHROPIC_API_KEY aiact "Who is a deployer?"
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
# Models are baked in: never contact the Hugging Face hub at runtime (also makes --network none work).
ENV HF_HUB_OFFLINE=1

RUN useradd --create-home app && chown -R app /app/data/cache
USER app

ENTRYPOINT ["python", "-m", "aiact.ask"]
CMD ["--help"]
