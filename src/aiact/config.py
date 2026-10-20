from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _env(name: str) -> AliasChoices:
    """Accept both the shared un-prefixed name (e.g. LLM_PROVIDER) and AIACT_<name>."""
    return AliasChoices(name, f"AIACT_{name}")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_prefix="AIACT_", extra="ignore", populate_by_name=True
    )

    raw_html: Path = PROJECT_ROOT / "data" / "raw" / "ai_act_en.html"
    corpus_path: Path = PROJECT_ROOT / "data" / "corpus" / "ai_act.jsonl"
    eval_path: Path = PROJECT_ROOT / "data" / "eval" / "questions.yaml"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"  # model weights + embedding cache, gitignored
    cassette_dir: Path = PROJECT_ROOT / "data" / "eval" / "cassettes"

    # Retrieval defaults chosen by the benchmark (data/bench/retrieval.md, docs/BUILD_LOG.md step 5).
    chunking: str = "paragraph"
    embedder: str = "bge-small"
    retrieval_mode: str = "rerank"
    top_k: int = 5
    recital_penalty: float = 2.0  # rank recitals below binding text (CHALLENGES #6, BUILD_LOG step 7)
    # Refuse when the best raw cross-encoder score is below this: mid-gap cut tuned on 55 answerable vs 8
    # unanswerable questions (data/bench/production_p2.md).
    refusal_threshold: float = 1.37

    # API answer cache: exact match always; semantic match only at cosine >= cache_threshold AND identical
    # discriminating terms. 0.96 is the lowest threshold with zero false hits on data/eval/cache_pairs.yaml
    # (data/bench/cache.json) - near-off by design: a false hit serves a wrong answer with valid citations.
    cache_enabled: bool = True
    cache_threshold: float = 0.96
    cache_max_entries: int = 1000
    cache_ttl_s: float = 86400.0

    # LLM provider: anthropic (Claude API) | ollama (free, local) | fake (replay a recorded cassette)
    llm_provider: Literal["anthropic", "ollama", "fake"] = Field(
        "fake", validation_alias=_env("LLM_PROVIDER")
    )
    llm_model: str = Field("claude-sonnet-5-5", validation_alias=_env("LLM_MODEL"))
    llm_effort: Literal["low", "medium", "high", "xhigh", "max"] = Field(
        "medium", validation_alias=_env("LLM_EFFORT")
    )
    ollama_host: str = Field("http://localhost:11434", validation_alias=_env("OLLAMA_HOST"))
    ollama_model: str = Field("qwen2.5:7b", validation_alias=_env("OLLAMA_MODEL"))

    # LLM judge for the Ragas metrics. Defaults to Opus (judge stronger than the generator); `ollama` uses
    # OLLAMA_MODEL, which is then also the generator - a self-evaluation caveat reported with the numbers.
    judge_provider: Literal["anthropic", "ollama", "fake"] = Field(
        "fake", validation_alias=_env("JUDGE_PROVIDER")
    )
    judge_model: str = Field("claude-opus-5-5", validation_alias=_env("JUDGE_MODEL"))


settings = Settings()
