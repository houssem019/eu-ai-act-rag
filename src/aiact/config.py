from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AIACT_", extra="ignore")

    raw_html: Path = PROJECT_ROOT / "data" / "raw" / "ai_act_en.html"
    corpus_path: Path = PROJECT_ROOT / "data" / "corpus" / "ai_act.jsonl"
    eval_path: Path = PROJECT_ROOT / "data" / "eval" / "questions.yaml"
    cache_dir: Path = PROJECT_ROOT / "data" / "cache"  # model weights + embedding cache, gitignored

    # Retrieval defaults chosen by the benchmark (data/bench/retrieval.md, docs/BUILD_LOG.md step 5).
    chunking: str = "paragraph"
    embedder: str = "bge-small"
    retrieval_mode: str = "rerank"
    top_k: int = 5
    # Refuse when the best cross-encoder score is below this (tuned on 55 answerable vs 8 unanswerable).
    refusal_threshold: float = 2.36


settings = Settings()
