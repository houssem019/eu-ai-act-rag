from pathlib import Path

from aiact.config import Settings, settings
from aiact.llm.base import LLMClient, LLMResponse, Usage

__all__ = ["LLMClient", "LLMResponse", "Usage", "get_llm"]


def get_llm(cfg: Settings = settings, cassette: Path | None = None, record: bool = False) -> LLMClient:
    """Build the configured LLM. `fake` replays `cassette`; record=True wraps the real provider and saves."""
    from aiact.llm.replay import ReplayLLM

    if cfg.llm_provider == "fake":
        return ReplayLLM(cassette or cfg.cassette_dir / "default.json")
    if cfg.llm_provider == "anthropic":
        from aiact.llm.anthropic_client import AnthropicLLM

        llm: LLMClient = AnthropicLLM(cfg.llm_model, effort=cfg.llm_effort)
    else:
        from aiact.llm.ollama_client import OllamaLLM

        llm = OllamaLLM(cfg.ollama_model, cfg.ollama_host)
    return ReplayLLM(cassette, inner=llm) if record and cassette else llm
