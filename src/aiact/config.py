from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="AIACT_", extra="ignore")

    raw_html: Path = PROJECT_ROOT / "data" / "raw" / "ai_act_en.html"
    corpus_path: Path = PROJECT_ROOT / "data" / "corpus" / "ai_act.jsonl"


settings = Settings()
